"""
Test coarse sweep autofocus against a simulated stage that exhibits
realistic backlash and a focus surface.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import cv2

from autofocus import (
    Preprocessor, MetricBank, AdaptiveCombiner,
    CoarseSweepAutofocus,
)


class SimulatedStage:
    """
    Simulates a stepper-driven Z stage with backlash.

    Internally tracks the true mechanical position separately from the
    commanded position. When direction reverses, `backlash_steps` are
    consumed before the mechanical position starts moving again.
    """
    def __init__(self, true_focus_z: int = 50, backlash: int = 30):
        self.true_focus_z = true_focus_z
        self.backlash     = backlash
        self.mech_z       = 0           # actual physical z
        self.cmd_z        = 0           # commanded cumulative z
        self.last_dir     = 0
        self.slack        = 0           # remaining backlash to consume

    def relative(self, dz: int):
        if dz == 0:
            return
        new_dir = +1 if dz > 0 else -1
        if self.last_dir != 0 and new_dir != self.last_dir:
            # Direction reversal — must consume backlash first
            self.slack = self.backlash
        self.cmd_z += dz
        # Resolve the move
        remaining = abs(dz)
        if self.slack > 0:
            consumed = min(self.slack, remaining)
            self.slack -= consumed
            remaining -= consumed
        # Whatever remains actually moves the mechanism
        self.mech_z += new_dir * remaining
        self.last_dir = new_dir


def make_frame_at_z(true_z: int, focus_z: int) -> np.ndarray:
    """Synthesise a microscope frame with sharpness depending on |true_z - focus_z|.

    Uses a fixed particle layout (seeded once) — only blur sigma varies with z,
    just like a real microscope.
    """
    rng = np.random.RandomState(42)            # fixed layout
    h, w = 480, 640
    img = np.full((h, w), 200, dtype=np.float32)
    yy, xx = np.indices((h, w))
    r2 = ((xx - w/2)**2 + (yy - h/2)**2) / (max(w, h)**2 / 4)
    img *= (1.0 - 0.30 * np.clip(r2, 0, 1))
    cx_arr = rng.randint(50, w - 50, size=40)
    cy_arr = rng.randint(50, h - 50, size=40)
    rr_arr = rng.randint(4, 9, size=40)
    for cx, cy, r in zip(cx_arr, cy_arr, rr_arr):
        cv2.circle(img, (int(cx), int(cy)), int(r), 20.0, -1)
    sigma = 0.5 + min(8.0, abs(true_z - focus_z) * 0.06)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    # IID fresh noise like a real camera
    img += np.random.normal(0, 1.5, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_coarse_sweep():
    print("\n[T] Coarse sweep with backlash + 21 samples")
    pp       = Preprocessor()
    bank     = MetricBank(use_jpeg=False)   # skip jpeg for speed
    combiner = AdaptiveCombiner()
    stage    = SimulatedStage(true_focus_z=50, backlash=30)

    def grab():
        return make_frame_at_z(stage.mech_z, stage.true_focus_z)

    def score(frame):
        r = pp.process(frame)
        m = bank.compute(r.enhanced, r.valid_mask)
        s = combiner.combine(m, r.condition["type"])
        return s, m

    # Patch time.sleep to instant for the test
    import autofocus.coarse as coarse_mod
    real_sleep = time.sleep
    time.sleep = lambda *_a, **_kw: None
    try:
        cs = CoarseSweepAutofocus(
            stage_move=stage.relative,
            grab_frame=grab,
            score_fn=score,
            range_steps=400,           # wider sweep — more out-of-focus samples
            n_samples=21,
            settle_s=0.0,
            backlash_steps=40,
            log_fn=lambda s: None,
            min_snr=1.10,              # this scene's surface is genuinely shallow
            min_prominence=0.04,
        )
        result = cs.sweep()
    finally:
        time.sleep = real_sleep

    print(f"   success            : {result.success}")
    print(f"   best_z (cum dz)    : {result.best_z}")
    print(f"   best_score         : {result.best_score:.4f}")
    print(f"   parabolic_peak     : {result.parabolic_peak}")
    print(f"   stage cum cmd_z    : {stage.cmd_z}")
    print(f"   stage true mech_z  : {stage.mech_z}")
    print(f"   target true focus  : {stage.true_focus_z}")
    print(f"   snr / prominence   : {result.snr:.2f} / {result.prominence:.2f}")
    print(f"   diagnostic         : {result.diagnostic}")

    # The mech_z should be reasonably close to the true focus
    # (within one sweep-step worth of error)
    step_size = 400 / 20    # range / (n_samples - 1)
    assert abs(stage.mech_z - stage.true_focus_z) <= 1.5 * step_size, \
        f"mech_z={stage.mech_z} too far from true focus={stage.true_focus_z}"
    assert result.success
    print("   ✓ PASS")


if __name__ == "__main__":
    print("=" * 64)
    print("  COARSE SWEEP TEST")
    print("=" * 64)
    test_coarse_sweep()
    print("\n  ✓ COARSE TEST PASSED")
