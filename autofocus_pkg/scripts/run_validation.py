"""
scripts/run_validation.py
=========================
Runnable scientific validation of the autofocus system.

Builds a synthetic z-stack with a known ground-truth peak, runs every
benchmark in validation/validator.py, and writes plots + a JSON report
to ./validation_out/.

Usage:
    python scripts/run_validation.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from autofocus import (
    Preprocessor, MetricBank, AdaptiveCombiner, ConfidenceEstimator,
    FineFocusController, AutofocusSystem, NullStage,
)
from validation.validator import Validator, ValidationReport


# ──────────────────────────────────────────────────────────────────────
# Synthetic z-stack generator (matches end-to-end test fixture)
# ──────────────────────────────────────────────────────────────────────
def make_z_stack(true_focus_z: int = 0,
                 z_min: int = -100, z_max: int = 100, dz: int = 10,
                 seed: int = 42) -> list:
    """Build a list of (z_int, frame_bgr) covering the focus surface."""
    rng = np.random.RandomState(seed)
    h, w = 480, 640
    cx = rng.randint(50, w - 50, size=40)
    cy = rng.randint(50, h - 50, size=40)
    rr = rng.randint(4, 9, size=40)

    stack = []
    for z in range(z_min, z_max + 1, dz):
        img = np.full((h, w), 200, dtype=np.float32)
        yy, xx = np.indices((h, w))
        r2 = ((xx - w/2)**2 + (yy - h/2)**2) / (max(w, h)**2 / 4)
        img *= (1.0 - 0.30 * np.clip(r2, 0, 1))
        for x, y, r in zip(cx, cy, rr):
            cv2.circle(img, (int(x), int(y)), int(r), 20.0, -1)
        sigma = 0.5 + min(8.0, abs(z - true_focus_z) * 0.06)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        img += np.random.normal(0, 1.5, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        stack.append((z, bgr))
    return stack


# ──────────────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "validation_out")
    out_dir = os.path.abspath(out_dir)
    print(f"Writing outputs to: {out_dir}")

    # ── 1. Build the z-stack ─────────────────────────────────────────
    np.random.seed(0)
    true_z = 0
    stack = make_z_stack(true_focus_z=true_z, z_min=-100, z_max=100, dz=10)
    print(f"Built z-stack with {len(stack)} frames, true peak z={true_z}")

    # ── 2. Build the focus-score function ────────────────────────────
    pp       = Preprocessor()
    bank     = MetricBank(use_jpeg=True)
    combiner = AdaptiveCombiner()
    conf_est = ConfidenceEstimator()

    def score_fn(frame):
        r = pp.process(frame)
        m = bank.compute(r.enhanced, r.valid_mask)
        s = combiner.combine(m, r.condition["type"])
        c = conf_est.estimate(m, r.condition)
        return s, {"metrics": m, "condition": r.condition, "confidence": c}

    # ── 3. Validator ─────────────────────────────────────────────────
    val = Validator(z_stack=stack, true_peak_z=true_z,
                    score_fn=score_fn, out_dir=out_dir)

    # ── 4. Per-metric accuracy ───────────────────────────────────────
    print("\n[1/4] Per-metric accuracy")
    curves = val.per_metric_accuracy(
        metric_names=["tenengrad", "laplacian", "brenner", "jpeg", "counting"]
    )
    for c in curves:
        print(f"   {c.name:<10}  pred={c.pred_peak:+7.2f}  "
              f"true={c.true_peak:+5.1f}  MAE={c.mae:5.2f}  "
              f"snr={c.snr:.2f}  prom={c.prominence:.2f}")
    p1 = val.plot_focus_curves(curves, "focus_curves.png")
    p2 = val.plot_peak_error(curves, "peak_error_bar.png")
    print(f"   wrote {p1}")
    print(f"   wrote {p2}")

    # ── 5. Repeatability ─────────────────────────────────────────────
    print("\n[2/4] Repeatability (8 trials, fresh noise each)")
    rep_std, rep_n = val.repeatability(n_trials=8, noise_sigma=1.5)
    print(f"   std of predicted peak across trials: {rep_std:.3f} steps  (n={rep_n})")

    # ── 6. Pipeline timing ───────────────────────────────────────────
    print("\n[3/4] Pipeline timing")
    def time_preproc(frame): return pp.process(frame)
    def time_metrics(frame):
        r = pp.process(frame)
        return bank.compute(r.enhanced, r.valid_mask)
    def time_combine(frame):
        r = pp.process(frame)
        m = bank.compute(r.enhanced, r.valid_mask)
        return combiner.combine(m, r.condition["type"])
    def time_full(frame):
        return score_fn(frame)
    timing = val.timing_breakdown(
        pipeline_stages={
            "preproc": time_preproc,
            "metrics": time_metrics,
            "combine": time_combine,
            "full":    time_full,
        },
        n_repeats=20,
    )
    for t in timing:
        print(f"   {t.stage:<10}  mean={t.mean_ms:6.2f} ms   p95={t.p95_ms:6.2f} ms")
    p3 = val.plot_timing(timing, "timing_breakdown.png")
    print(f"   wrote {p3}")

    # ── 7. Tracking trace from a closed-loop run ─────────────────────
    print("\n[4/4] Tracking trace (closed-loop on simulated stage)")
    # simple simulator: set rig.mech_z, regenerate frame
    class _Rig:
        def __init__(self):
            self.mech_z = -50    # start off-focus
            self.true   = 0
            self._h, self._w = 480, 640
            rng = np.random.RandomState(42)
            self._cx = rng.randint(50, self._w-50, size=40)
            self._cy = rng.randint(50, self._h-50, size=40)
            self._rr = rng.randint(4, 9, size=40)
            self.last_dir = 0
            self.slack = 0
            self.backlash = 25
        def move(self, dz):
            if dz == 0: return
            new_dir = +1 if dz > 0 else -1
            if self.last_dir != 0 and new_dir != self.last_dir:
                self.slack = self.backlash
            r = abs(dz)
            if self.slack > 0:
                c = min(self.slack, r); self.slack -= c; r -= c
            self.mech_z += new_dir * r
            self.last_dir = new_dir
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

    rig = _Rig()
    class StageBridge(NullStage):
        def __init__(self, rig): super().__init__(); self.rig = rig
        def move_z(self, dz):     self.rig.move(int(dz)); return super().move_z(dz)
        def move_xyz(self, dx, dy, dz): self.rig.move(int(dz)); return super().move_xyz(dx, dy, dz)

    stage = StageBridge(rig)
    real_sleep = time.sleep; time.sleep = lambda *_a, **_kw: None
    try:
        # Patch coarse for synthetic curve
        from autofocus.coarse import CoarseSweepAutofocus as _CS
        _orig = _CS.__init__
        def _patched(self, **kw):
            kw["min_snr"] = 1.05
            kw["min_prominence"] = 0.02
            _orig(self, **kw)
        _CS.__init__ = _patched

        af = AutofocusSystem(
            grab_frame=rig.grab, stage=stage,
            coarse_range_steps=300, coarse_n_samples=21,
            coarse_settle_s=0.0, coarse_backlash=40,
            fine_settle_s=0.0,
            fine=FineFocusController(
                coarse_step=4, fine_step=1, min_step=1,
                deadband=0.02, hysteresis_frames=2, stable_required=4,
                smooth_window=5,
            ),
        )
        coarse_res = af.coarse_focus()
        print(f"   coarse: success={coarse_res.success}  best_z={coarse_res.best_z}  "
              f"snr={coarse_res.snr:.2f}  rig.mech_z={rig.mech_z}")
        records = af.track(stop_after_frames=80)
    finally:
        time.sleep = real_sleep

    osc = Validator.tracking_oscillation([r.cum_z for r in records])
    print(f"   tracking: final cum_z={records[-1].cum_z}  "
          f"final state={records[-1].state}  rig.mech_z={rig.mech_z}")
    print(f"   oscillation: reversals={osc['reversals']:.0f}  "
          f"rms_drift={osc['rms_drift']:.2f}")
    p4 = val.plot_tracking_trace(records, "tracking_trace.png")
    print(f"   wrote {p4}")

    # ── 8. Build report ──────────────────────────────────────────────
    report = ValidationReport(
        accuracy_per_metric=curves,
        repeatability_std=rep_std,
        repeatability_n=rep_n,
        timing=timing,
        tracking_oscillation=osc,
        notes=[
            f"true_peak_z={true_z}",
            f"z_stack range -100..100 step 10 (n={len(stack)})",
            f"coarse parked at rig.mech_z={rig.mech_z} (vs true={rig.true})",
        ],
    )
    json_path = os.path.join(out_dir, "report.json")
    report.save_json(json_path)
    print(f"\nWrote report to: {json_path}")
    print("\n  ✓ VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
