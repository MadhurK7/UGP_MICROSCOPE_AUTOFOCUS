"""
autofocus/stage_iface.py
========================
Thin, OpenFlexure-firmware-compatible Z-axis stage abstraction.

PROTOCOL (lines, '\\n' terminated, ASCII):
    "mr <dx> <dy> <dz>"   relative move on three axes
    "p?"                  query position → "x y z"
    "release"             de-energise motor coils
    "zero"                set internal position to 0,0,0
    "dt <us>"             min step delay (motor speed)
    "ramp_time <us>"      accel / decel ramp time
    "version"             firmware banner

Replies finishing with "done." indicate command completion.

DESIGN
──────
Two backends in one class:
    SerialStage    talks to an Arduino over USB
    NullStage      records moves only — for testing / dry-run

Both expose IDENTICAL interfaces:
    move_z(dz)          → blocking, waits for "done."
    move_xyz(dx,dy,dz)  → blocking
    get_position()      → (x,y,z) tuple
    release()           → de-energise coils
    close()             → release + close port

The Coarse and Fine controllers only need `move_z()`, so they remain
hardware-agnostic.

THREAD SAFETY
─────────────
A single `_lock` serialises all serial I/O. The thread doing autofocus
and any thread that wants to nudge the stage manually must both
acquire this lock. (Most autofocus runs single-threaded; the lock
exists so the camera capture thread can be added later without
risking interleaved bytes on the wire.)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

try:
    import serial as _pyserial
    SERIAL_OK = True
except ImportError:
    _pyserial = None
    SERIAL_OK = False


# ──────────────────────────────────────────────────────────────────────
class StageError(RuntimeError):
    """Raised on serial / protocol failure."""


# ──────────────────────────────────────────────────────────────────────
class BaseStage:
    """Common interface."""
    def move_z(self, dz: int) -> bool:                   raise NotImplementedError
    def move_xyz(self, dx: int, dy: int, dz: int) -> bool: raise NotImplementedError
    def get_position(self) -> Optional[Tuple[int, int, int]]: raise NotImplementedError
    def release(self) -> bool:                            raise NotImplementedError
    def close(self) -> None:                              raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
# SERIAL BACKEND
# ══════════════════════════════════════════════════════════════════════
class SerialStage(BaseStage):
    """OpenFlexure-firmware-compatible serial stage."""

    BOOT_DELAY_S      = 2.0
    REPLY_TIMEOUT_S   = 5.0
    PER_CMD_CLAMP     = 600        # max steps per single mr command

    def __init__(
        self,
        port: str           = "/dev/ttyACM0",
        baud: int           = 115200,
        invert_z: bool      = False,
        min_step_delay_us:  Optional[int] = None,
        ramp_time_us:       Optional[int] = None,
    ):
        if not SERIAL_OK:
            raise StageError("pyserial not installed")
        self._port = port
        self._baud = baud
        self._invert_z = -1 if invert_z else 1
        self._lock = threading.RLock()
        self._ser: "Optional[_pyserial.Serial]" = None
        self._connect()

        if min_step_delay_us is not None:
            self._send(f"dt {int(min_step_delay_us)}")
        if ramp_time_us is not None:
            self._send(f"ramp_time {int(ramp_time_us)}")

    # ── connection ────────────────────────────────────────────────────
    def _connect(self) -> None:
        try:
            self._ser = _pyserial.Serial(
                self._port, self._baud, timeout=1.0, write_timeout=1.0
            )
        except Exception as e:
            raise StageError(f"could not open {self._port}: {e}")
        time.sleep(self.BOOT_DELAY_S)
        self._ser.reset_input_buffer()
        # ask for version banner — non-fatal if absent
        ver = self._send("version", expect_done=False)
        logger.info(f"[SerialStage] firmware: {ver!r}")

    # ── public API ────────────────────────────────────────────────────
    def move_z(self, dz: int) -> bool:
        return self.move_xyz(0, 0, dz)

    def move_xyz(self, dx: int, dy: int, dz: int) -> bool:
        dx, dy, dz = int(dx), int(dy), int(dz) * self._invert_z
        # break into chunks to respect firmware safety
        c = self.PER_CMD_CLAMP
        ok = True
        while dx or dy or dz:
            chunk_x = max(-c, min(c, dx)); dx -= chunk_x
            chunk_y = max(-c, min(c, dy)); dy -= chunk_y
            chunk_z = max(-c, min(c, dz)); dz -= chunk_z
            ok = ok and self._send(f"mr {chunk_x} {chunk_y} {chunk_z}",
                                   expect_done=True) is not None
        return ok

    def get_position(self) -> Optional[Tuple[int, int, int]]:
        reply = self._send("p?", expect_done=False)
        if not reply:
            return None
        try:
            parts = reply.strip().split()
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            return None

    def release(self) -> bool:
        return self._send("release", expect_done=False) is not None

    def close(self) -> None:
        try:
            self.release()
        except Exception:
            pass
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass

    # ── transport ────────────────────────────────────────────────────
    def _send(self, cmd: str, expect_done: bool = True) -> Optional[str]:
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
                self._ser.flush()
            except Exception as e:
                logger.error(f"[SerialStage] write error on {cmd!r}: {e}")
                return None

            if expect_done:
                deadline = time.monotonic() + self.REPLY_TIMEOUT_S
                last_data: Optional[str] = None
                while time.monotonic() < deadline:
                    try:
                        line = self._ser.readline().decode("ascii", errors="replace").strip()
                    except Exception:
                        return None
                    if not line:
                        continue
                    if line.lower().startswith("done"):
                        return last_data if last_data else line
                    last_data = line
                logger.warning(f"[SerialStage] timeout on {cmd!r}")
                return None
            else:
                try:
                    line = self._ser.readline().decode("ascii", errors="replace").strip()
                except Exception:
                    return None
                return line if line else None


# ══════════════════════════════════════════════════════════════════════
# NULL BACKEND
# ══════════════════════════════════════════════════════════════════════
class NullStage(BaseStage):
    """In-memory stage. Records every move; never touches hardware."""

    def __init__(self, invert_z: bool = False):
        self.invert_z = -1 if invert_z else 1
        self._x = self._y = self._z = 0
        self.history: List[Tuple[int, int, int]] = []

    def move_z(self, dz: int) -> bool:
        return self.move_xyz(0, 0, dz)

    def move_xyz(self, dx: int, dy: int, dz: int) -> bool:
        dz = int(dz) * self.invert_z
        self._x += int(dx); self._y += int(dy); self._z += dz
        self.history.append((self._x, self._y, self._z))
        return True

    def get_position(self): return (self._x, self._y, self._z)
    def release(self):      return True
    def close(self):        pass
