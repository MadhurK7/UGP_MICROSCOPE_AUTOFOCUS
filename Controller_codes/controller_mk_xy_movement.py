"""
controller.py
=============
Serial Controller — OpenFlexure Motor Board Edition

PURPOSE:
    Bridge between the CV tracking pipeline and the physical XYZ stage,
    speaking the OpenFlexure motor board firmware protocol over USB serial.

ARCHITECTURE:
    Tracker position (x, y)
         ↓
    _compute_displacement()   ← dead zone + proportional gain
         ↓
    move_relative(dx, dy, dz) ← serial write of "mr <x> <y> <z>\n"
         ↓
    Arduino (OpenFlexure FW) → ULN2003 drivers → 3 × 28BYJ-48 motors

═══════════════════════════════════════════════════════════════════════
WHY THIS REPLACES THE SINGLE-BYTE PROTOCOL
═══════════════════════════════════════════════════════════════════════
The original controller sent single bytes ('L','R','U','D','S') for
fixed-step moves. The OpenFlexure firmware does not understand these —
it expects ASCII commands terminated by '\n':

    mr <dx> <dy> <dz>     — relative move on all 3 axes
    mrx <d>               — relative move x only
    mry <d>               — relative move y only
    mrz <d>               — relative move z only
    p?                    — query current position
    release               — de-energise all motor coils
    zero                  — reset internal coordinates to (0,0,0)
    dt <us>               — set min step delay (speed)
    ramp_time <us>        — set accel/decel ramp time
    version               — firmware version string
    help                  — list all commands

Replies are text (e.g. "done.", "1234 -567 0", or version banner) — we
read the line-terminated response after each command.

═══════════════════════════════════════════════════════════════════════
KEY DESIGN DECISIONS
═══════════════════════════════════════════════════════════════════════
1. PROPORTIONAL CONTROL instead of fixed-step bang-bang.
   Error vector (dx, dy) → step displacement scaled by GAIN.
   Larger error → larger move. Reduces oscillation near centre.

2. DEAD ZONE preserved (10% of frame, configurable).
   No move is sent if particle is inside the dead zone — saves bandwidth
   and avoids motor hunting on tracker noise.

3. MAX STEP CLAMP per command.
   Caps the step count per command so a sudden tracking glitch can't
   send the stage flying. Default 200 steps ≈ small visible move.

4. ALL 3 AXES SUPPORTED.
   Uses `mr <dx> <dy> <dz>` so future Z autofocus can hook in trivially.

5. BAUD = 115200 (OpenFlexure firmware default), not 9600.

6. WAITS FOR REPLY after each move ("done.\n").
   Prevents queueing commands faster than the motor can execute.
   Includes timeout protection so a hung firmware can't lock the pipeline.

7. AXIS SIGN INVERSION configurable per axis.
   If your stage moves the wrong way, flip axis_invert_x/y/z in __init__
   without rewiring or reflashing firmware.

8. RELEASE on close + final zero-displacement halt.
   Sends `release` to de-energise coils on shutdown — protects motors
   from heating and reduces electrical noise to imaging sensor.
"""

import time
import logging
from typing import Optional, Tuple

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logging.warning(
        "[Controller] pyserial not installed. "
        "Running in DRY-RUN mode. Install with: pip install pyserial"
    )

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# CONTROLLER CLASS
# ══════════════════════════════════════════════════════════════════════

class Controller:
    """
    OpenFlexure-protocol serial controller for particle centering.

    Converts tracker (x, y) error → relative-move command and sends it
    to the OpenFlexure motor board over USB serial.

    Usage:
        ctl = Controller(port='/dev/ttyACM0', baud=115200)
        ctl.update(position, frame_width, frame_height)
        ctl.close()

    Tuning:
        gain                  - steps per pixel of error (default 2)
        max_steps_per_command - safety clamp (default 200)
        dead_zone_fraction    - inside this, no move (default 0.10)
    """

    # ── Tuning constants (override in __init__ if needed) ─────────────
    DEFAULT_BAUD = 115200            # OpenFlexure firmware uses 115200
    BOOT_DELAY_S = 2.0               # Arduino bootloader delay
    REPLY_TIMEOUT_S = 5.0            # Wait this long for "done." reply
    MIN_SEND_INTERVAL_S = 0.02       # 20ms — protects serial buffer

    def __init__(
        self,
        port: str = '/dev/ttyACM0',
        baud: int = DEFAULT_BAUD,
        gain: float = 2.0,
        max_steps_per_command: int = 200,
        dead_zone_fraction: float = 0.10,
        axis_invert_x: bool = False,
        axis_invert_y: bool = False,
        axis_invert_z: bool = False,
        min_step_delay_us: Optional[int] = None,   # if set, sent on connect
        ramp_time_us: Optional[int] = None,        # if set, sent on connect
    ):
        """
        Initialize serial connection to the OpenFlexure motor board.

        Args:
            port:                 Serial device. Linux: '/dev/ttyACM0' /
                                  '/dev/ttyUSB0', Windows: 'COM3' etc.
            baud:                 Baud rate. Must match firmware
                                  (OpenFlexure default: 115200).
            gain:                 Steps per pixel of error. Larger = more
                                  aggressive centering. Start at 2.0,
                                  reduce if oscillating, raise if sluggish.
            max_steps_per_command: Safety clamp on step magnitude per
                                  command. Prevents runaway moves on
                                  bad tracking data.
            dead_zone_fraction:   Half-width of dead zone as fraction of
                                  frame dim (0.10 = ±10% from centre).
            axis_invert_x/y/z:    Flip sign on each axis if the stage
                                  moves opposite to the expected
                                  direction. Easier than rewiring.
            min_step_delay_us:    If provided, sent as `dt <us>` on
                                  connect. Lower = faster motor.
                                  Typical 28BYJ-48: 1000–3000 us.
            ramp_time_us:         If provided, sent as `ramp_time <us>`
                                  on connect. Adds accel/decel for
                                  smoother moves. 0 = instant max speed.
        """
        self._port = port
        self._baud = baud
        self._serial: Optional['serial.Serial'] = None
        self._connected = False
        self._dry_run = False

        # Control parameters
        self._gain = gain
        self._max_steps = max_steps_per_command
        self._dead_zone = dead_zone_fraction
        self._invert = (
            -1 if axis_invert_x else 1,
            -1 if axis_invert_y else 1,
            -1 if axis_invert_z else 1,
        )
        self._min_step_delay_us = min_step_delay_us
        self._ramp_time_us = ramp_time_us

        # Rate-limit / state tracking
        self._last_send_time = 0.0
        self._last_displacement = (0, 0, 0)

        # Statistics
        self._total_updates = 0
        self._total_moves = 0
        self._total_holds = 0   # times we decided not to move (in dead zone)
        self._firmware_version = None

        self._connect()

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────

    def update(
        self,
        position: Optional[Tuple[float, float]],
        width: int,
        height: int,
    ) -> Tuple[int, int, int]:
        """
        Main update call — run once per tracking frame.

        Computes the required (dx, dy) displacement from the particle's
        current position relative to frame centre, then sends an `mr`
        command to the firmware if the error exceeds the dead zone.

        Args:
            position: (x, y) tuple from tracker — pixel coordinates of
                      the tracked particle. Pass None if particle lost.
            width:    Frame width in pixels.
            height:   Frame height in pixels.

        Returns:
            (dx, dy, dz) — the displacement actually sent (steps).
                           (0,0,0) means we held position.

        Notes:
            On lost particle (position is None) we hold position. Sending
            random moves on stale data is dangerous; the firmware will
            simply continue holding the last position.
        """
        self._total_updates += 1

        if position is None:
            # Lost particle → hold (do not send anything)
            self._total_holds += 1
            return (0, 0, 0)

        dx, dy, dz = self._compute_displacement(position, width, height)

        if dx == 0 and dy == 0 and dz == 0:
            # Inside dead zone — nothing to do
            self._total_holds += 1
            return (0, 0, 0)

        # Apply the same axis-inversion + clamping that move_relative will,
        # so the returned tuple reflects what was actually requested
        clamped = self._clamp_and_invert(dx, dy, dz)

        # Attempt to send; if rate-limited we still report the intent
        # (caller can use the return value for display/logging).
        self.move_relative(dx, dy, dz)
        return clamped

    def _clamp_and_invert(self, dx: int, dy: int, dz: int) -> Tuple[int, int, int]:
        """Apply axis inversion and step clamping (used by both update and move_relative)."""
        dx = int(dx) * self._invert[0]
        dy = int(dy) * self._invert[1]
        dz = int(dz) * self._invert[2]
        dx = max(-self._max_steps, min(self._max_steps, dx))
        dy = max(-self._max_steps, min(self._max_steps, dy))
        dz = max(-self._max_steps, min(self._max_steps, dz))
        return (dx, dy, dz)

    def move_relative(self, dx: int, dy: int, dz: int = 0) -> bool:
        """
        Send a relative move command: `mr <dx> <dy> <dz>\\n`.

        Applies axis inversion and step clamping before sending.
        Waits for the firmware's "done." reply with a timeout.

        Args:
            dx, dy, dz: signed integer step counts.

        Returns:
            True if command sent and acknowledged; False otherwise.
        """
        # Apply axis inversion + safety clamp
        dx, dy, dz = self._clamp_and_invert(dx, dy, dz)

        # Rate limit
        now = time.monotonic()
        if now - self._last_send_time < self.MIN_SEND_INTERVAL_S:
            return False

        cmd = f"mr {dx} {dy} {dz}"
        reply = self._send_and_read(cmd, expect_done=True)

        self._last_displacement = (dx, dy, dz)
        self._last_send_time = now

        if reply is None:
            return False

        self._total_moves += 1
        return True

    def get_position(self) -> Optional[Tuple[int, int, int]]:
        """
        Query current stage position via `p?` command.

        Returns:
            (x, y, z) tuple of integer step counts, or None on failure.
        """
        reply = self._send_and_read("p?", expect_done=False)
        if reply is None:
            return None
        try:
            parts = reply.strip().split()
            if len(parts) >= 3:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError) as e:
            logger.warning(f"[Controller] Could not parse position: {reply!r}: {e}")
        return None

    def zero_position(self) -> bool:
        """Reset firmware's internal position counter to (0,0,0)."""
        return self._send_and_read("zero", expect_done=False) is not None

    def release_motors(self) -> bool:
        """De-energise all motor coils. Use when stage should idle."""
        return self._send_and_read("release", expect_done=False) is not None

    def set_min_step_delay(self, us: int) -> bool:
        """Set minimum delay between steps in microseconds (motor speed)."""
        return self._send_and_read(f"dt {int(us)}", expect_done=True) is not None

    def set_ramp_time(self, us: int) -> bool:
        """Set acceleration/deceleration ramp time in microseconds."""
        return self._send_and_read(
            f"ramp_time {int(us)}", expect_done=True
        ) is not None

    def close(self):
        """
        Safely close serial connection.

        - Releases motor coils (firmware `release` command).
        - Closes the serial port.
        - Prints session statistics.
        """
        logger.info("[Controller] Closing...")

        if self._connected and self._serial:
            try:
                self.release_motors()
                time.sleep(0.1)
                logger.info("[Controller] Motors released.")
            except Exception:
                pass

        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
                logger.info(f"[Controller] Port {self._port} closed.")
            except Exception as e:
                logger.warning(f"[Controller] Error closing port: {e}")

        self._connected = False
        self._print_stats()

    # ──────────────────────────────────────────────────────────────────
    # CORE LOGIC
    # ──────────────────────────────────────────────────────────────────

    def _compute_displacement(
        self,
        position: Tuple[float, float],
        width: int,
        height: int,
    ) -> Tuple[int, int, int]:
        """
        Convert particle position → (dx, dy, dz) step displacement.

        Logic:
            error = position − frame_center
            if |error_x| < dead_zone_x and |error_y| < dead_zone_y:
                return (0, 0, 0)
            dx = round(error_x · gain)   ← proportional move
            dy = round(error_y · gain)

        STAGE COORDINATE CONVENTION (OpenFlexure standard):
            Particle drifts RIGHT in image  → stage moves RIGHT to recenter
                            → +x command
            Particle drifts DOWN in image   → stage moves DOWN
                            → +y command
            Z is unused for centering (autofocus would set it separately).

        If your stage rig moves opposite, flip axis_invert_x/y in __init__.
        """
        x, y = position
        cx = width / 2.0
        cy = height / 2.0

        err_x = x - cx
        err_y = y - cy

        thresh_x = width * self._dead_zone
        thresh_y = height * self._dead_zone

        # Inside dead zone → no move
        if abs(err_x) < thresh_x and abs(err_y) < thresh_y:
            return (0, 0, 0)

        # Proportional control
        dx = int(round(err_x * self._gain))
        dy = int(round(err_y * self._gain))
        dz = 0

        return (dx, dy, dz)

    # ──────────────────────────────────────────────────────────────────
    # SERIAL I/O
    # ──────────────────────────────────────────────────────────────────

    def _send_and_read(
        self,
        command: str,
        expect_done: bool = True,
    ) -> Optional[str]:
        """
        Send a command (newline appended) and read a reply line.

        The OpenFlexure firmware terminates most replies with "done."
        Movement commands always do; query commands (p?) return the
        data line directly. Set expect_done=False for those.

        Args:
            command:     Command string without trailing newline.
            expect_done: If True, read until a line equal to "done."
                         is received. If False, read just one line.

        Returns:
            The reply string, or None on failure / timeout / dry-run.
        """
        if self._dry_run or not self._connected:
            logger.debug(f"[Controller DRY-RUN] would send: {command!r}")
            return ""  # non-None so calling code knows it "succeeded"

        try:
            self._serial.reset_input_buffer()
            self._serial.write((command + "\n").encode("ascii"))
            self._serial.flush()
            logger.debug(f"[Controller] sent: {command!r}")

            if expect_done:
                # Read lines until "done." or timeout
                deadline = time.monotonic() + self.REPLY_TIMEOUT_S
                last_data_line = None
                while time.monotonic() < deadline:
                    line = self._serial.readline().decode(
                        "ascii", errors="replace"
                    ).strip()
                    if not line:
                        continue
                    logger.debug(f"[Controller] recv: {line!r}")
                    if line.lower().startswith("done"):
                        return last_data_line if last_data_line else line
                    last_data_line = line
                logger.warning(
                    f"[Controller] timeout waiting for 'done.' reply to {command!r}"
                )
                return None
            else:
                # Read a single reply line
                line = self._serial.readline().decode(
                    "ascii", errors="replace"
                ).strip()
                logger.debug(f"[Controller] recv: {line!r}")
                return line if line else None

        except serial.SerialException as e:
            logger.error(f"[Controller] Serial error on {command!r}: {e}")
            self._connected = False
            return None
        except Exception as e:
            logger.error(f"[Controller] Unexpected error on {command!r}: {e}")
            return None

    def _connect(self):
        """
        Open serial port, wait for boot, read version banner, apply
        any user-specified motor parameters (dt / ramp_time).

        Falls back to dry-run if anything goes wrong — pipeline keeps
        running so you can debug without hardware.
        """
        if not SERIAL_AVAILABLE:
            logger.warning("[Controller] pyserial unavailable → dry-run.")
            self._dry_run = True
            return

        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=1.0,
                write_timeout=1.0,
            )
            logger.info(
                f"[Controller] Waiting for firmware boot "
                f"(port={self._port}, baud={self._baud})..."
            )
            time.sleep(self.BOOT_DELAY_S)

            # Drain any boot output then ask for version
            self._serial.reset_input_buffer()
            self._connected = True

            version = self._send_and_read("version", expect_done=False)
            if version:
                self._firmware_version = version
                logger.info(f"[Controller] Firmware: {version}")
            else:
                logger.warning(
                    "[Controller] No version reply — firmware may be wrong "
                    "or baud mismatched. Continuing anyway."
                )

            # Apply optional motor parameters
            if self._min_step_delay_us is not None:
                self.set_min_step_delay(self._min_step_delay_us)
                logger.info(
                    f"[Controller] min_step_delay → {self._min_step_delay_us} us"
                )
            if self._ramp_time_us is not None:
                self.set_ramp_time(self._ramp_time_us)
                logger.info(f"[Controller] ramp_time → {self._ramp_time_us} us")

        except serial.SerialException as e:
            logger.error(
                f"[Controller] Cannot open {self._port}: {e}\n"
                f"  → Check Arduino plugged in?\n"
                f"  → Try /dev/ttyUSB0 / COM3 / etc.\n"
                f"  → Running in dry-run mode."
            )
            self._dry_run = True
        except Exception as e:
            logger.error(
                f"[Controller] Unexpected connection error: {e}\n"
                f"  → Running in dry-run mode."
            )
            self._dry_run = True

    # ──────────────────────────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────────────────────────

    def _print_stats(self):
        if self._total_updates == 0:
            return
        print("\n[Controller] Session Statistics:")
        print(f"  Firmware              : {self._firmware_version or 'unknown'}")
        print(f"  Frames processed      : {self._total_updates}")
        print(f"  Moves sent            : {self._total_moves}")
        print(f"  Holds (dead zone/lost): {self._total_holds}")
        if self._total_updates > 0:
            move_pct = 100 * self._total_moves / self._total_updates
            print(f"  Move rate             : {move_pct:5.1f}%")

    # ──────────────────────────────────────────────────────────────────
    # PROPERTIES
    # ──────────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run

    @property
    def firmware_version(self) -> Optional[str]:
        return self._firmware_version

    @property
    def last_displacement(self) -> Tuple[int, int, int]:
        return self._last_displacement


# ══════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST (dry-run only)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    print("=" * 60)
    print("  CONTROLLER (OpenFlexure protocol) — DRY-RUN TEST")
    print("=" * 60)

    # Force dry-run by passing a non-existent port
    ctl = Controller(
        port="/dev/null_nonexistent",
        gain=2.0,
        max_steps_per_command=200,
        dead_zone_fraction=0.10,
    )

    W, H = 640, 480

    print("\n[T1] Particle at centre → expect (0,0,0)")
    out = ctl.update((320, 240), W, H);  print(f"   got {out}")
    assert out == (0, 0, 0)

    print("\n[T2] Particle far right → expect +dx")
    out = ctl.update((600, 240), W, H);  print(f"   got {out}")
    assert out[0] > 0 and out[1] == 0

    print("\n[T3] Particle far down-left → expect -dx, +dy")
    out = ctl.update((50, 450), W, H);   print(f"   got {out}")
    assert out[0] < 0 and out[1] > 0

    print("\n[T4] Particle just outside dead zone → small move")
    out = ctl.update((320 + int(0.11 * W), 240), W, H)
    print(f"   got {out}")
    assert out[0] > 0

    print("\n[T5] Particle inside dead zone → no move")
    out = ctl.update((320 + int(0.05 * W), 240), W, H)
    print(f"   got {out}")
    assert out == (0, 0, 0)

    print("\n[T6] Lost particle → hold")
    out = ctl.update(None, W, H);        print(f"   got {out}")
    assert out == (0, 0, 0)

    print("\n[T7] Step clamp test — huge error should clamp")
    out = ctl.update((10000, 10000), W, H)
    print(f"   got {out}  (should be clamped to ±200)")
    assert abs(out[0]) <= 200 and abs(out[1]) <= 200

    ctl.close()
    print("\n  ALL TESTS PASSED ✓")
