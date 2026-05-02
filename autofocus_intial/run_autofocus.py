"""
run_autofocus.py
================
Runnable entry-point for OpenFlexureAutofocusV2.

EXAMPLES
────────

    # Real hardware: USB camera + Arduino on /dev/ttyACM0
    python run_autofocus.py \\
        --port /dev/ttyACM0 --camera 0 \\
        --csv run.csv --duration 300

    # Hardware test without an Arduino — coarse sweep simulated only,
    # motor commands are logged but never sent
    python run_autofocus.py --camera 0 --simulate-stage --csv test.csv

    # Pure simulation, no hardware needed at all
    python run_autofocus.py --simulate --csv simrun.csv --duration 30
"""
import argparse
import logging
import sys
import time

import cv2
import numpy as np

# Local modules
from hardware import RealCamera, SerialStage, BacklashAwareStage, SERIAL_OK
from openflexure_autofocus_full import OpenFlexureAutofocusV2


# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    # source
    p.add_argument("--camera",  type=int, default=0,
                   help="OpenCV camera index (ignored with --simulate)")
    p.add_argument("--simulate", action="store_true",
                   help="use the synthetic stage+camera (no hardware)")
    # serial
    p.add_argument("--port",    default="/dev/ttyACM0")
    p.add_argument("--baud",    type=int, default=115200)
    p.add_argument("--simulate-stage", action="store_true",
                   help="real camera, but stage commands logged not sent")
    p.add_argument("--invert-z", action="store_true")
    # autofocus
    p.add_argument("--sweep-range",  type=int, default=3000)
    p.add_argument("--sweep-step",   type=int, default=150)
    p.add_argument("--backlash",     type=int, default=80,
                   help="extra steps to overshoot when reversing direction")
    p.add_argument("--settle",       type=float, default=0.20)
    p.add_argument("--fine-step",    type=int, default=2)
    p.add_argument("--fine-deadband", type=float, default=0.01)
    p.add_argument("--track-period", type=float, default=0.10)
    # session
    p.add_argument("--no-coarse", action="store_true",
                   help="skip the coarse sweep, go straight to fine")
    p.add_argument("--no-fine",   action="store_true",
                   help="run only the coarse sweep, no fine tracking")
    p.add_argument("--duration",  type=float, default=120.0,
                   help="fine tracking duration in seconds")
    p.add_argument("--csv",       default=None,
                   help="path for telemetry CSV (none → don't log)")
    p.add_argument("--display",   action="store_true",
                   help="show live OpenCV window during tracking")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# SIMULATION RIG (used by --simulate)
# ──────────────────────────────────────────────────────────────────────
class _SimMechStage:
    def __init__(self, backlash=60):
        self.backlash = backlash; self._cmd_z = 0; self._mech_z = 0
        self._last_dir = 0; self._slack = 0
    @property
    def position(self): return {"x": 0, "y": 0, "z": self._cmd_z}
    @property
    def mech_z(self):   return self._mech_z
    def move_relative(self, x=0, y=0, z=0):
        if z == 0: return True
        nd = +1 if z > 0 else -1
        if self._last_dir != 0 and nd != self._last_dir:
            self._slack = self.backlash
        self._cmd_z += int(z); rem = abs(z)
        if self._slack > 0:
            c = min(self._slack, rem); self._slack -= c; rem -= c
        self._mech_z += nd * rem; self._last_dir = nd
        return True
    def release(self): pass
    def close(self):   pass


class _SimCamera:
    def __init__(self, stage, true_focus_z=50):
        self.stage = stage; self.true = true_focus_z
        h, w = 480, 640
        rng = np.random.RandomState(7)
        base = np.full((h, w), 230, dtype=np.float32)
        yy, xx = np.indices((h, w))
        r2 = ((xx - w/2)**2 + (yy - h/2)**2) / (max(w, h)**2 / 4)
        base *= (1.0 - 0.30 * np.clip(r2, 0, 1))
        for _ in range(200):
            cx = rng.randint(20, w - 20); cy = rng.randint(20, h - 20)
            r  = rng.randint(2, 6)
            cv2.circle(base, (cx, cy), r, 20.0, -1)
        tex = rng.normal(0, 6, (h, w)).astype(np.float32)
        base = base + cv2.GaussianBlur(tex, (0, 0), 0.7)
        self._base = np.clip(base, 0, 255)
    def grab_frame(self):
        sigma = 0.4 + min(6.0, abs(self.stage.mech_z - self.true) * 0.04)
        img = cv2.GaussianBlur(self._base, (0, 0), sigma)
        img += np.random.normal(0, 1.5, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    def close(self): pass


class _LoggingNullStage:
    """Used by --simulate-stage. Logs commands instead of sending them."""
    def __init__(self):
        self._z = 0
    @property
    def position(self): return {"x": 0, "y": 0, "z": self._z}
    def move_relative(self, x=0, y=0, z=0):
        self._z += int(z)
        logging.getLogger("stage").info(f"[STAGE] mr 0 0 {z}  → cum_z={self._z}")
        return True
    def release(self):
        logging.getLogger("stage").info("[STAGE] release")
        return True
    def close(self): pass


# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_autofocus")

    # ── Build the camera + stage ────────────────────────────────────
    if args.simulate:
        log.info("SIMULATE: synthetic stage + camera")
        raw_stage = _SimMechStage(backlash=60)
        camera    = _SimCamera(raw_stage, true_focus_z=50)
    else:
        log.info(f"opening camera {args.camera}")
        camera = RealCamera(index=args.camera)

        if args.simulate_stage:
            log.info("SIMULATE STAGE: motor commands logged, not sent")
            raw_stage = _LoggingNullStage()
        else:
            if not SERIAL_OK:
                log.error("pyserial missing — install with `pip install pyserial`")
                return 2
            log.info(f"opening serial stage {args.port}@{args.baud}")
            raw_stage = SerialStage(
                port=args.port, baud=args.baud, invert_z=args.invert_z,
            )

    # Wrap with backlash compensation
    stage = BacklashAwareStage(
        raw_stage,
        backlash_steps = args.backlash,
        approach       = "from_below",
        settle_s       = args.settle,
    )

    # ── Build the autofocus ─────────────────────────────────────────
    af = OpenFlexureAutofocusV2(
        stage  = stage,
        camera = camera,
        sweep_range      = args.sweep_range,
        sweep_step       = args.sweep_step,
        fine_step        = args.fine_step,
        fine_deadband    = args.fine_deadband,
        track_period_s   = args.track_period,
        csv_path         = args.csv,
        debug            = args.verbose,
    )

    # ── Optional live display ───────────────────────────────────────
    palette = {"init": (200,200,200), "climb": (0,255,0),
               "refine": (0,200,255), "locked": (255,100,100),
               "hold": (50,50,200)}
    on_frame = None
    if args.display:
        def _cb(z, score, conf, state, metrics):
            frame = camera.grab_frame()
            disp = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            col = palette.get(state, (255,255,255))
            for i, line in enumerate([
                f"score: {score:.3f}",
                f"conf : {conf:.2f}",
                f"state: {state}",
                f"z    : {z:+d}",
            ]):
                cv2.putText(disp, line, (12, 26 + i*24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            cv2.imshow("Autofocus", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
        on_frame = _cb

    # ── Run ──────────────────────────────────────────────────────────
    try:
        if not args.no_coarse:
            log.info("─── COARSE SWEEP ───")
            peak = af.autofocus()
            log.info(f"coarse peak at z={peak}")

        if not args.no_fine:
            log.info(f"─── FINE TRACKING ({args.duration:.0f} s) ───")
            n = af.track(duration_s=args.duration, on_frame=on_frame)
            log.info(f"tracked {n} frames; final state={af.fine.state}; "
                     f"final z={stage.position['z']}")
    finally:
        if args.display:
            cv2.destroyAllWindows()
        af.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
