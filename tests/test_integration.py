"""
Integration tests for preprocessor + metrics + fine controller.
Uses synthetic frames at varying simulated focus offsets.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from autofocus import (
    Preprocessor, MetricBank, AdaptiveCombiner, ConfidenceEstimator,
    FineFocusController, FineState,
)


def make_microscope_frame(z_off: float, n_particles: int = 40,
                          h: int = 480, w: int = 640,
                          seed: int = 0) -> np.ndarray:
    """
    Synthesise a bright-field colloidal frame with realistic vignette.

    z_off:   "defocus" — particles get blurred more as |z_off| grows.
             At z_off = 0 we are at focus.
    """
    rng = np.random.RandomState(seed)
    img = np.full((h, w), 200, dtype=np.float32)        # bright background
    # add vignetting (radial intensity falloff)
    yy, xx = np.indices((h, w))
    r2 = ((xx - w/2)**2 + (yy - h/2)**2) / (max(w, h)**2 / 4)
    img *= (1.0 - 0.35 * np.clip(r2, 0, 1))
    # particles
    cx = rng.randint(50, w - 50, size=n_particles)
    cy = rng.randint(50, h - 50, size=n_particles)
    rr = rng.randint(4, 10, size=n_particles)
    for x, y, r in zip(cx, cy, rr):
        cv2.circle(img, (int(x), int(y)), int(r), 60.0, -1)
    # blur proportional to defocus + small fixed PSF
    sigma = 0.7 + abs(z_off)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    # noise
    img += rng.normal(0, 1.5, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_preprocessor_basic():
    print("\n[T1] Preprocessor on synthetic frame")
    pp = Preprocessor()
    frame = make_microscope_frame(z_off=0.5, n_particles=30)
    r = pp.process(frame)
    print(f"   condition: {r.condition['type']}")
    print(f"   particle ROIs: {len(r.particle_rois)}")
    print(f"   centre ROI shape: {r.centre_roi.shape}")
    assert r.gray.shape == (480, 640)
    assert r.normalised.shape == (480, 640)
    assert r.condition["type"] in {"ok", "low_contrast", "low_intensity"}
    assert len(r.particle_rois) > 0, "should detect some particles"
    print("   ✓ PASS")


def test_metrics_monotonic_around_focus():
    print("\n[T2] Metrics — score peaks near in-focus frame")
    pp = Preprocessor()
    bank = MetricBank(use_jpeg=True)
    combiner = AdaptiveCombiner()

    z_offs = np.linspace(-3.0, 3.0, 13)
    scores = []
    for z in z_offs:
        frame = make_microscope_frame(z_off=z, n_particles=40, seed=42)
        r = pp.process(frame)
        m = bank.compute(r.enhanced, r.valid_mask)
        s = combiner.combine(m, r.condition["type"])
        scores.append(s)
        print(f"   z={z:+.2f}  score={s:.4f}  cond={r.condition['type']}")

    peak_idx = int(np.argmax(scores))
    peak_z = z_offs[peak_idx]
    print(f"   peak at z={peak_z:+.2f}  (true peak z=0.0)")
    assert abs(peak_z) <= 0.5, f"peak {peak_z} too far from 0"
    print("   ✓ PASS  (peak found within 0.5 of true focus)")


def test_confidence_low_on_empty():
    print("\n[T3] Confidence drops on empty / saturated frames")
    pp = Preprocessor()
    bank = MetricBank()
    conf_est = ConfidenceEstimator()

    # empty grey frame
    blank = np.full((480, 640, 3), 50, dtype=np.uint8)
    r = pp.process(blank)
    m = bank.compute(r.enhanced, r.valid_mask)
    c_blank = conf_est.estimate(m, r.condition)

    # saturated frame
    sat = np.full((480, 640, 3), 255, dtype=np.uint8)
    r = pp.process(sat)
    m = bank.compute(r.enhanced, r.valid_mask)
    c_sat = conf_est.estimate(m, r.condition)

    # good frame
    good = make_microscope_frame(z_off=0.5, n_particles=40)
    r = pp.process(good)
    m = bank.compute(r.enhanced, r.valid_mask)
    c_good = conf_est.estimate(m, r.condition)

    print(f"   blank conf  : {c_blank:.3f}")
    print(f"   saturated   : {c_sat:.3f}")
    print(f"   good frame  : {c_good:.3f}")
    assert c_blank <= 0.4
    assert c_sat   <= 0.4
    assert c_good  >  0.5
    print("   ✓ PASS")


def test_fine_controller_climbs_synthetic_curve():
    print("\n[T4] FineFocusController — climbs simulated focus surface")
    ctl = FineFocusController(coarse_step=4, fine_step=1, min_step=1,
                              deadband=0.005, hysteresis_frames=2,
                              stable_required=4, smooth_window=3)
    z_star = 18
    cum_z = 0
    rng = np.random.RandomState(7)

    for i in range(80):
        # focus surface: sharper near z_star with noise
        score = 1.0 / (1 + ((cum_z - z_star) / 10.0) ** 2)
        score += rng.normal(0, 0.005)
        d = ctl.update(score=score, confidence=0.9)
        cum_z += d.dz
        if i % 10 == 0:
            print(f"   i={i:3d}  z={cum_z:+4d}  score={score:.3f}  "
                  f"state={d.state.value}  dz={d.dz:+d}  "
                  f"reason='{d.reason}'")

    print(f"   final z={cum_z}  target z*={z_star}  state={ctl.state.value}")
    assert abs(cum_z - z_star) <= 5, "did not converge near peak"
    print("   ✓ PASS")


def test_fine_low_confidence_holds():
    print("\n[T5] FineFocusController — holds when confidence is low")
    ctl = FineFocusController()
    d = ctl.update(score=1.0, confidence=0.10)
    assert d.dz == 0
    assert d.state == FineState.HOLD
    print(f"   state={d.state.value}  dz={d.dz}  reason='{d.reason}'")
    print("   ✓ PASS")


if __name__ == "__main__":
    print("=" * 64)
    print("  AUTOFOCUS — INTEGRATION TESTS (preprocessor + metrics + fine)")
    print("=" * 64)
    test_preprocessor_basic()
    test_metrics_monotonic_around_focus()
    test_confidence_low_on_empty()
    test_fine_controller_climbs_synthetic_curve()
    test_fine_low_confidence_holds()
    print("\n" + "=" * 64)
    print("  ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 64)
