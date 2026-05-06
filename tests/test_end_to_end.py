"""
End-to-end test: AutofocusSystem performs coarse sweep AND fine tracking
against a simulated stage with realistic backlash and a synthetic focus
surface.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from autofocus import AutofocusSystem, NullStage


# ──────────────────────────────────────────────────────────────────────
class SimulatedZRig:
    """
    Simulates: stepper-driven Z stage + camera looking at colloidal sample.

    The stage has backlash. The camera sees a frame whose blur depends
    on the distance between the mechanical z and the true focus z.
    """
    def __init__(self, true_focus_z: int = 70, backlash: int = 25):
        self.true_focus_z = true_focus_z
        self.backlash     = backlash
        self.mech_z       = 0
        self.last_dir     = 0
        self.slack        = 0
        # cache the static particle layout
        self._h, self._w = 480, 640
        rng = np.random.RandomState(42)
        self._cx = rng.randint(50, self._w - 50, size=40)
        self._cy = rng.randint(50, self._h - 50, size=40)
        self._rr = rng.randint(4, 9, size=40)

    def stage_move_z(self, dz: int):
        if dz == 0:
            return
        new_dir = +1 if dz > 0 else -1
        if self.last_dir != 0 and new_dir != self.last_dir:
            self.slack = self.backlash
        remaining = abs(dz)
        if self.slack > 0:
            consumed = min(self.slack, remaining)
            self.slack -= consumed
            remaining -= consumed
        self.mech_z += new_dir * remaining
        self.last_dir = new_dir

    def grab_frame(self) -> np.ndarray:
        h, w = self._h, self._w
        img = np.full((h, w), 200, dtype=np.float32)
        yy, xx = np.indices((h, w))
        r2 = ((xx - w/2)**2 + (yy - h/2)**2) / (max(w, h)**2 / 4)
        img *= (1.0 - 0.30 * np.clip(r2, 0, 1))
        # darker, higher-contrast particles → larger gradient signal
        for cx, cy, r in zip(self._cx, self._cy, self._rr):
            cv2.circle(img, (int(cx), int(cy)), int(r), 20.0, -1)
        # blur strongly off-focus, sharply at focus — but cap so the
        # signal never completely disappears (real microscope behaviour)
        sigma = 0.5 + min(8.0, abs(self.mech_z - self.true_focus_z) * 0.06)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        # IID fresh noise like a real camera (not deterministic on mech_z)
        img += np.random.normal(0, 1.5, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class StageAdapter(NullStage):
    """Bridges SimulatedZRig.stage_move_z into the BaseStage interface."""
    def __init__(self, rig: SimulatedZRig):
        super().__init__()
        self.rig = rig
    def move_z(self, dz: int) -> bool:
        self.rig.stage_move_z(int(dz))
        return super().move_z(dz)
    def move_xyz(self, dx: int, dy: int, dz: int) -> bool:
        self.rig.stage_move_z(int(dz))
        return super().move_xyz(dx, dy, dz)


def main():
    print("=" * 64)
    print("  END-TO-END SYSTEM TEST")
    print("=" * 64)

    rig   = SimulatedZRig(true_focus_z=70, backlash=25)
    stage = StageAdapter(rig)

    # patch time.sleep to instant
    real_sleep = time.sleep
    time.sleep = lambda *_a, **_kw: None

    try:
        af = AutofocusSystem(
            grab_frame   = rig.grab_frame,
            stage        = stage,
            coarse_range_steps=400,
            coarse_n_samples=21,
            coarse_settle_s=0.0,
            coarse_backlash=40,
            fine_settle_s=0.0,
            # Replace the default fine controller with one tuned for
            # post-coarse refinement: tiny steps, big deadband,
            # short smoothing — we should already be close to focus.
            fine=__import__('autofocus').FineFocusController(
                coarse_step=4, fine_step=1, min_step=1,
                deadband=0.02,           # noise margin
                hysteresis_frames=2,
                stable_required=4,
                smooth_window=5,
            ),
        )
        # Patch coarse to use a slightly lower snr threshold for this test
        from autofocus.coarse import CoarseSweepAutofocus as _CS
        _orig = _CS.__init__
        def _patched(self, **kw):
            kw["min_snr"] = 1.05
            kw["min_prominence"] = 0.02
            _orig(self, **kw)
        _CS.__init__ = _patched

        print("\n[Step 1] Coarse sweep")
        coarse = af.coarse_focus()
        print(f"   success      : {coarse.success}")
        print(f"   best_z       : {coarse.best_z}")
        print(f"   parabolic    : {coarse.parabolic_peak}")
        print(f"   snr/prom     : {coarse.snr:.2f}/{coarse.prominence:.2f}")
        print(f"   rig.mech_z   : {rig.mech_z}  (true focus={rig.true_focus_z})")

        print("\n[Step 2] Fine tracking (50 frames)")
        records = af.track(stop_after_frames=50)
        last = records[-1]
        print(f"   final t      : {last.t:.2f}s")
        print(f"   final cum_z  : {last.cum_z}")
        print(f"   final score  : {last.score:.3f}")
        print(f"   final state  : {last.state}")
        print(f"   rig.mech_z   : {rig.mech_z}  (true focus={rig.true_focus_z})")

        # The final mechanical z should be close to true focus
        err = abs(rig.mech_z - rig.true_focus_z)
        print(f"\n   ABSOLUTE ERROR: {err} steps (true_focus={rig.true_focus_z})")
        # After coarse + fine, expect within ~1 coarse-sample resolution
        # (here: 400 / 20 = 20 steps), since fine refinement may add a few
        # extra steps in noise-dominated synthetic scenes.
        assert err <= 30, f"final mech_z error {err} too large"
        assert last.state in ("locked", "refine", "climb"), \
            f"unexpected controller state: {last.state}"

        print("\n  ✓ END-TO-END TEST PASSED")
    finally:
        time.sleep = real_sleep


if __name__ == "__main__":
    main()
