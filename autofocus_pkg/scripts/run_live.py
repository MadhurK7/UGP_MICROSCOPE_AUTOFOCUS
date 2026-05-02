"""
scripts/run_live.py
===================
Live entry point — runs the full autofocus system on a real camera +
OpenFlexure-firmware-compatible Arduino.

Usage:
    python scripts/run_live.py --port /dev/ttyACM0 --camera 0
    python scripts/run_live.py --simulate     # no hardware needed

The simulated mode runs against the synthetic z-rig used in the
validation tests, useful for testing the full software stack on a
machine without a microscope.
"""
import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autofocus import (
    AutofocusSystem, FineFocusController, NullStage, SerialStage, SERIAL_OK,
)


# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",     default="/dev/ttyACM0",
                   help="serial port to OpenFlexure firmware Arduino")
    p.add_argument("--baud",     type=int, default=115200)
    p.add_argument("--camera",   type=int, default=0,
                   help="OpenCV camera index")
    p.add_argument("--simulate", action="store_true",
                   help="use the synthetic rig (no hardware required)")
    p.add_argument("--coarse",   action="store_true",
                   help="run a coarse sweep before fine tracking")
    p.add_argument("--frames",   type=int, default=300,
                   help="stop after this many frames (-1 = forever)")
    p.add_argument("--no-display", action="store_true",
                   help="don't open an OpenCV window")
    p.add_argument("--invert-z", action="store_true",
                   help="invert Z direction (some rigs are mounted upside-down)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
class _SimRig:
    """Synthetic stage + camera for --simulate."""
    def __init__(self):
        self.true   = 35
        self.mech_z = -50
        self.last_dir = 0; self.slack = 0; self.backlash = 25
        self._h, self._w = 480, 640
        rng = np.random.RandomState(123)
        self._cx = rng.randint(50, self._w-50, size=40)
        self._cy = rng.randint(50, self._h-50, size=40)
        self._rr = rng.randint(4, 9, size=40)
    def grab(self):
        img = np.full((self._h, self._w), 200, dtype=np.float32)
        yy, xx = np.indices((self._h, self._w))
        r2 = ((xx - self._w/2)**2 + (yy - self._h/2)**2) / (max(self._w, self._h)**2 / 4)
        img *= (1.0 - 0.30 * np.clip(r2, 0, 1))
        for x, y, r in zip(self._cx, self._cy, self._rr):
            cv2.circle(img, (int(x), int(y)), int(r), 20.0, -1)
        sigma = 0.5 + min(8.0, abs(self.mech_z - self.true) * 0.06)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        img += np.random.normal(0, 1.5, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    def move(self, dz):
        if dz == 0: return
        nd = +1 if dz > 0 else -1
        if self.last_dir != 0 and nd != self.last_dir: self.slack = self.backlash
        r = abs(dz)
        if self.slack > 0:
            c = min(self.slack, r); self.slack -= c; r -= c
        self.mech_z += nd * r; self.last_dir = nd


class _SimStage(NullStage):
    """Bridges _SimRig into BaseStage."""
    def __init__(self, rig): super().__init__(); self.rig = rig
    def move_z(self, dz): self.rig.move(int(dz)); return super().move_z(dz)
    def move_xyz(self, dx, dy, dz): self.rig.move(int(dz)); return super().move_xyz(dx, dy, dz)


# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    log = logging.getLogger("run_live")

    # ── Camera + stage setup ────────────────────────────────────────
    if args.simulate:
        log.info("Running in SIMULATE mode (no hardware).")
        rig = _SimRig()
        grab = rig.grab
        stage = _SimStage(rig)
    else:
        log.info(f"Opening camera index {args.camera}")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            log.error("Cannot open camera.")
            return 1
        def grab():
            ok, frame = cap.read()
            return frame if ok else None

        if not SERIAL_OK:
            log.error("pyserial missing — install with `pip install pyserial`")
            return 1
        log.info(f"Connecting to Arduino on {args.port}@{args.baud}")
        stage = SerialStage(port=args.port, baud=args.baud,
                            invert_z=args.invert_z)

    # ── Autofocus system ─────────────────────────────────────────────
    af = AutofocusSystem(
        grab_frame=grab,
        stage=stage,
        # Reasonable defaults for 28BYJ-48-class steppers with belt
        coarse_range_steps = 400,
        coarse_n_samples   = 21,
        coarse_settle_s    = 0.20,
        coarse_backlash    = 80,
        fine_settle_s      = 0.05,
        fine = FineFocusController(
            coarse_step=8, fine_step=2, min_step=1,
            deadband=0.01, hysteresis_frames=3, stable_required=5,
            smooth_window=5,
        ),
    )

    # ── Optional coarse sweep ───────────────────────────────────────
    if args.coarse:
        log.info("Running coarse sweep…")
        cr = af.coarse_focus()
        log.info(f"  success={cr.success}  best_z={cr.best_z}  "
                 f"snr={cr.snr:.2f}  prom={cr.prominence:.2f}  diag={cr.diagnostic}")

    # ── Display callback ────────────────────────────────────────────
    palette = {
        "init": (200, 200, 200), "climb": (0, 255, 0),
        "refine": (0, 200, 255), "locked": (255, 100, 100),
        "hold": (50, 50, 200),   "probe": (0, 255, 255),
    }

    def on_frame(rec, info):
        if args.no_display:
            return
        enhanced = info["preproc"].enhanced
        disp = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        col = palette.get(rec.state, (255, 255, 255))
        for i, line in enumerate([
            f"score:  {rec.score:.3f}",
            f"conf :  {rec.confidence:.2f}",
            f"state:  {rec.state}",
            f"cum_z:  {rec.cum_z:+d}",
            f"cond :  {rec.condition}",
        ]):
            cv2.putText(disp, line, (15, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imshow("Autofocus", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt()

    # ── Track loop ───────────────────────────────────────────────────
    log.info("Entering tracking loop. Press 'q' or Ctrl-C to stop.")
    try:
        af.track(
            stop_after_frames=args.frames if args.frames > 0 else None,
            on_frame=on_frame,
        )
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        if not args.simulate:
            try: cap.release()
            except Exception: pass
        try: stage.close()
        except Exception: pass

        if af.session_log:
            log.info(
                f"Session: {len(af.session_log)} frames  "
                f"final state={af.session_log[-1].state}  "
                f"cum_z={af.session_log[-1].cum_z}  "
                f"best score={af.fine.best_score:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
