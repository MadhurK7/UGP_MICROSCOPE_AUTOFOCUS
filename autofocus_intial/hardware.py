"""
hardware.py
============
Real hardware adapters that drop into your openflexure_autofocus_fixed.py
in place of MockStage / MockCamera. They expose the SAME methods:

    stage.position           → {"z": int}
    stage.move_relative(z)   → blocking, returns when motor done
    camera.grab_frame()      → BGR uint8 ndarray (H,W,3)

So your existing OpenFlexureAutofocus class works untouched.

THREE THINGS THIS MODULE PROVIDES
─────────────────────────────────
1. RealCamera         — OpenCV VideoCapture wrapper for USB / Pi cameras
2. SerialStage        — talks the OpenFlexure firmware protocol over USB
3. BacklashAwareStage — wraps any stage and adds backlash compensation,
                         settling delays, and direction-aware moves

WHY BACKLASH MATTERS
────────────────────
A 28BYJ-48 stepper driving an OpenFlexure flexure stage through a
plastic gear train has 30–200 steps of mechanical play. When you
reverse direction, the FIRST steps after the reversal don't move the
sample at all — the gears are taking up slack. Then the sample jumps.

Symptoms if you ignore this:
  * autofocus appears to "skip" steps after a direction change
  * peak position depends on which side you approached from
  * fine controller hunts forever because every reversal swallows
    the first N steps of the new direction

BacklashAwareStage fixes this by always APPROACHING the target from
the same direction. To go backward, it OVERSHOOTS by `backlash` then
steps forward to the target.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

try:
    import serial as _pyserial
    SERIAL_OK = True
except ImportError:
    _pyserial = None
    SERIAL_OK = False

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# REAL CAMERA
# ══════════════════════════════════════════════════════════════════════
class RealCamera:
    """
    OpenCV-based camera adapter.

    Works with USB webcams and (via the V4L2 backend) with the Raspberry
    Pi camera on a Pi running the libcamera-shim. For best results on a
    Pi 4 install the official picamera2 package and substitute
    `PiCamera2Adapter` (below) instead.
    """

    def __init__(self,
                 index: int = 0,
                 width:  Optional[int] = 640,
                 height: Optional[int] = 480,
                 target_fps: Optional[int] = 30,
                 warmup_frames: int = 5):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera index {index}")
        if width:      self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        if height:     self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if target_fps: self._cap.set(cv2.CAP_PROP_FPS,          target_fps)
        # First USB-cam frames are often black or auto-exposing; warm up
        for _ in range(warmup_frames):
            self._cap.read()
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"[RealCamera] index={index}  {self.width}x{self.height}")

    def grab_frame(self) -> np.ndarray:
        """Return one BGR frame. Raises if the camera disconnects."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("camera frame read failed")
        return frame

    def close(self) -> None:
        try: self._cap.release()
        except Exception: pass

    # context manager sugar
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# ══════════════════════════════════════════════════════════════════════
# SERIAL STAGE
# ══════════════════════════════════════════════════════════════════════
class SerialStage:
    """
    OpenFlexure-firmware-compatible stage on USB serial.

    PROTOCOL
    ────────
    Commands are ASCII, terminated with '\\n':
        mr <dx> <dy> <dz>     relative move
        p?                    query position → "x y z"
        release               de-energise motor coils
        zero                  reset position to (0,0,0)
        dt <us>               minimum step delay (motor speed)
        ramp_time <us>        accel/decel ramp time

    Each command is acknowledged by a "done." line; we wait for it
    before issuing the next command. This naturally rate-limits us.

    EXPOSED INTERFACE (drop-in for MockStage)
    ─────────────────────────────────────────
        stage.position           → {"x": int, "y": int, "z": int}
        stage.move_relative(z=N)
        stage.move_relative(x=N, y=M, z=K)
        stage.release()
        stage.close()
    """

    BOOT_DELAY_S      = 2.0     # Arduino bootloader wait
    REPLY_TIMEOUT_S   = 5.0
    PER_CMD_CLAMP     = 600     # max steps in a single mr command

    def __init__(self,
                 port: str   = "/dev/ttyACM0",
                 baud: int   = 115200,
                 invert_z:   bool = False,
                 invert_x:   bool = False,
                 invert_y:   bool = False,
                 min_step_delay_us: Optional[int] = None,
                 ramp_time_us:      Optional[int] = None):
        if not SERIAL_OK:
            raise RuntimeError("pyserial not installed (`pip install pyserial`)")
        self._port = port; self._baud = baud
        self._inv_x = -1 if invert_x else 1
        self._inv_y = -1 if invert_y else 1
        self._inv_z = -1 if invert_z else 1
        self._lock = threading.RLock()
        self._x = self._y = self._z = 0

        self._ser = _pyserial.Serial(
            port, baud, timeout=1.0, write_timeout=1.0,
        )
        time.sleep(self.BOOT_DELAY_S)
        self._ser.reset_input_buffer()
        # Read version banner if the firmware prints one
        ver = self._send("version", expect_done=False)
        logger.info(f"[SerialStage] firmware: {ver!r}")

        if min_step_delay_us is not None:
            self._send(f"dt {int(min_step_delay_us)}")
        if ramp_time_us is not None:
            self._send(f"ramp_time {int(ramp_time_us)}")

    # ── interface that matches MockStage ────────────────────────────
    @property
    def position(self) -> dict:
        return {"x": self._x, "y": self._y, "z": self._z}

    def move_relative(self,
                      x: int = 0, y: int = 0, z: int = 0) -> bool:
        """Issue `mr <dx> <dy> <dz>\\n`. Blocks until firmware says done."""
        # Apply axis inversions
        dx = int(x) * self._inv_x
        dy = int(y) * self._inv_y
        dz = int(z) * self._inv_z
        # Update internal counters using user's coordinate convention
        # (so stage.position["z"] reflects what the user requested, not
        # what the firmware thinks).
        self._x += int(x); self._y += int(y); self._z += int(z)
        # Break large moves into firmware-safe chunks
        c = self.PER_CMD_CLAMP
        ok = True
        while dx or dy or dz:
            chunk_x = max(-c, min(c, dx)); dx -= chunk_x
            chunk_y = max(-c, min(c, dy)); dy -= chunk_y
            chunk_z = max(-c, min(c, dz)); dz -= chunk_z
            ok = ok and self._send(f"mr {chunk_x} {chunk_y} {chunk_z}",
                                   expect_done=True) is not None
        return ok

    def release(self) -> bool:
        """De-energise motor coils. Reduces heat + electrical noise."""
        return self._send("release", expect_done=False) is not None

    def close(self) -> None:
        try:    self.release()
        except Exception: pass
        try:    self._ser.close()
        except Exception: pass

    # context manager sugar
    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    # ── transport ───────────────────────────────────────────────────
    def _send(self, cmd: str, expect_done: bool = True) -> Optional[str]:
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
                self._ser.flush()
            except Exception as e:
                logger.error(f"[SerialStage] write {cmd!r} failed: {e}")
                return None

            if expect_done:
                deadline = time.monotonic() + self.REPLY_TIMEOUT_S
                last_data: Optional[str] = None
                while time.monotonic() < deadline:
                    try:
                        line = self._ser.readline().decode(
                            "ascii", errors="replace"
                        ).strip()
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
                    return self._ser.readline().decode(
                        "ascii", errors="replace"
                    ).strip() or None
                except Exception:
                    return None


# ══════════════════════════════════════════════════════════════════════
# BACKLASH-AWARE STAGE WRAPPER
# ══════════════════════════════════════════════════════════════════════
class BacklashAwareStage:
    """
    Wrap any stage (Mock, Serial, ...) and add:

        - backlash compensation: always approach a target from one
          direction, overshooting by `backlash_steps` when reversing
        - settling delay: short pause after every move so the
          mechanism stops vibrating before the next frame is grabbed
        - safe absolute move: move_absolute(z) which an autofocus
          loop can call without thinking about backlash

    Wraps the inner stage transparently — `position`, `move_relative`,
    `release`, `close` all forward through, but motion paths are
    rewritten to honour backlash.

    USAGE
    ─────
        raw = SerialStage(port="/dev/ttyACM0")
        stage = BacklashAwareStage(raw, backlash_steps=80,
                                   approach="from_below",
                                   settle_s=0.20)

        af = OpenFlexureAutofocus(stage=stage, camera=cam)
        af.autofocus()
    """

    def __init__(self,
                 inner,
                 backlash_steps: int = 80,
                 approach: str   = "from_below",   # or "from_above"
                 settle_s: float = 0.15):
        self._inner = inner
        self.backlash = int(backlash_steps)
        if approach not in ("from_below", "from_above"):
            raise ValueError("approach must be 'from_below' or 'from_above'")
        self.dir_sign = +1 if approach == "from_below" else -1
        self.settle_s = float(settle_s)
        # Track last commanded direction so we know when a reversal happens
        self._last_dir_z = 0

    # ── pass-through state ──────────────────────────────────────────
    @property
    def position(self):
        return self._inner.position

    def release(self):
        return self._inner.release() if hasattr(self._inner, "release") else None

    def close(self):
        return self._inner.close() if hasattr(self._inner, "close") else None

    # ── relative move (backlash-aware on z) ─────────────────────────
    def move_relative(self, x: int = 0, y: int = 0, z: int = 0) -> bool:
        """
        Issue a relative move. If the z direction reverses since last
        time, eat the backlash by overshooting and approaching.
        """
        ok = True
        if z != 0:
            new_dir = +1 if z > 0 else -1
            if self._last_dir_z != 0 and new_dir != self._last_dir_z:
                # reversal — eat backlash by overshooting in the OPPOSITE
                # of the chosen approach direction, then re-approach
                overshoot = -self.dir_sign * self.backlash
                ok = ok and self._inner.move_relative(z=overshoot)
                self._settle()
                ok = ok and self._inner.move_relative(z=self.dir_sign * self.backlash)
                self._settle()
            ok = ok and self._inner.move_relative(z=int(z))
            self._last_dir_z = new_dir
            self._settle()

        if x != 0:
            ok = ok and self._inner.move_relative(x=int(x))
        if y != 0:
            ok = ok and self._inner.move_relative(y=int(y))
        if x or y:
            self._settle()
        return ok

    # ── absolute move with backlash compensation ────────────────────
    def move_absolute_z(self, target_z: int) -> bool:
        """
        Move so that stage.position['z'] == target_z, ALWAYS approaching
        from `approach` direction. Use this in autofocus sweeps.

        Two cases:

        1. We're going IN the approach direction (e.g. approach='from_below'
           → moving up, dz>0). No reversal, no backlash penalty. Just go.

        2. We're going AGAINST the approach direction. We must overshoot
           past target by `backlash`, then come back from the approach
           side, eating the slack on the way.
        """
        cur = self.position["z"]
        delta = target_z - cur
        if delta == 0:
            return True

        going_with_approach = (delta > 0) == (self.dir_sign > 0)

        if going_with_approach:
            # Same direction as last approach — no reversal needed
            ok = self._inner.move_relative(z=int(delta))
            self._last_dir_z = +1 if delta > 0 else -1
            self._settle()
            return ok

        # Reversal: overshoot past target in the opposite of approach,
        # then come back to target from the approach direction.
        # (`approach`-direction means dir_sign>0 ⇒ final move is +z;
        #  so overshoot is to target - backlash * dir_sign.)
        overshoot_pt   = target_z - self.dir_sign * self.backlash
        overshoot_dz   = overshoot_pt - cur
        approach_dz    = self.dir_sign * self.backlash
        ok = self._inner.move_relative(z=overshoot_dz)
        self._settle()
        ok = ok and self._inner.move_relative(z=approach_dz)
        self._last_dir_z = +1 if self.dir_sign > 0 else -1
        self._settle()
        return ok

    def _settle(self) -> None:
        if self.settle_s > 0:
            time.sleep(self.settle_s)
