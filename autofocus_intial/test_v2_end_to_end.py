"""
End-to-end test of OpenFlexureAutofocusV2.

Wires:
    SimMechStage  ──► BacklashAwareStage  ──► OpenFlexureAutofocusV2
    SimCamera     ───────────────────────────►

Verifies:
    1. Coarse sweep finds the true focus z within tolerance.
    2. Continuous fine tracking holds focus (no oscillation).
    3. CSV output is generated.
    4. close() releases motor coils.
"""
import os, sys, time, tempfile, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from hardware import BacklashAwareStage
from openflexure_autofocus_full import OpenFlexureAutofocusV2


# ──────────────────────────────────────────────────────────────────────
class SimMechStage:
    """Stepper stage with mechanical backlash."""
    def __init__(self, backlash: int = 60):
        self.backlash = backlash
        self._cmd_z   = 0
        self._mech_z  = 0
        self._last_dir = 0
        self._slack    = 0
        self._released = False

    @property
    def position(self): return {"x": 0, "y": 0, "z": self._cmd_z}
    @property
    def mech_z(self):   return self._mech_z

    def move_relative(self, x=0, y=0, z=0):
        self._released = False
        if z == 0: return True
        new_dir = +1 if z > 0 else -1
        if self._last_dir != 0 and new_dir != self._last_dir:
            self._slack = self.backlash
        self._cmd_z += int(z)
        rem = abs(z)
        if self._slack > 0:
            c = min(self._slack, rem); self._slack -= c; rem -= c
        self._mech_z += new_dir * rem
        self._last_dir = new_dir
        return True

    def release(self):
        self._released = True
        return True

    def close(self): pass


class SimCamera:
    """
    Camera looking at a richly-textured colloidal sample.

    We use many small particles + procedural noise pattern to give
    the metric engine plenty of edge content. Sharpness depends on
    |mech_z - true_focus_z|.
    """
    def __init__(self, stage: SimMechStage, true_focus_z: int = 50):
        self.stage = stage
        self.true_focus_z = true_focus_z
        h, w = 480, 640
        self._h, self._w = h, w
        # Pre-render a high-detail base image (sharp version of the sample)
        rng = np.random.RandomState(7)
        base = np.full((h, w), 230, dtype=np.float32)
        # Vignette
        yy, xx = np.indices((h, w))
        r2 = ((xx - w/2)**2 + (yy - h/2)**2) / (max(w, h)**2 / 4)
        base *= (1.0 - 0.30 * np.clip(r2, 0, 1))
        # 200 small dark colloids
        for _ in range(200):
            cx = rng.randint(20, w - 20)
            cy = rng.randint(20, h - 20)
            r  = rng.randint(2, 6)
            cv2.circle(base, (cx, cy), r, 20.0, -1)
        # Sub-pixel-ish texture so blur really kills sharpness
        texture = rng.normal(0, 6, (h, w)).astype(np.float32)
        base = base + cv2.GaussianBlur(texture, (0, 0), 0.7)
        self._base = np.clip(base, 0, 255)

    def grab_frame(self):
        sigma = 0.4 + min(6.0, abs(self.stage.mech_z - self.true_focus_z) * 0.04)
        img = cv2.GaussianBlur(self._base, (0, 0), sigma)
        img += np.random.normal(0, 1.5, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def close(self): pass


# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  OpenFlexureAutofocusV2 — END-TO-END TEST")
    print("=" * 64)

    # ── Build sim rig + autofocus ────────────────────────────────────
    raw   = SimMechStage(backlash=60)
    stage = BacklashAwareStage(raw, backlash_steps=80,
                               approach="from_below", settle_s=0.0)
    cam   = SimCamera(raw, true_focus_z=50)

    # patch time.sleep to instant for the test
    real_sleep = time.sleep
    time.sleep = lambda *_a, **_kw: None

    csv_path = os.path.join(tempfile.gettempdir(), "afv2_test.csv")
    if os.path.exists(csv_path): os.unlink(csv_path)

    af = OpenFlexureAutofocusV2(
        stage=stage, camera=cam,
        sweep_range=600, sweep_step=20,    # wide enough that even with
                                           # backlash offset, focus is sampled
        fine_coarse_step=4, fine_step=1,
        fine_deadband=0.02, fine_conf_floor=0.30,
        track_period_s=0.0,
        csv_path=csv_path,
        debug=False,
    )

    try:
        # ── Step 1: Coarse sweep ─────────────────────────────────────
        print("\n[1] Coarse sweep")
        peak_z = af.autofocus()
        print(f"   detected peak z (cmd) : {peak_z}")
        print(f"   stage cmd z           : {raw.position['z']}")
        print(f"   stage mech z          : {raw.mech_z}")
        print(f"   true focus            : {cam.true_focus_z}")

        coarse_err = abs(raw.mech_z - cam.true_focus_z)
        print(f"   coarse error          : {coarse_err} steps")
        # Accept within 1 sweep-step worth of error (20)
        assert coarse_err <= 25, f"coarse failed: error {coarse_err}"

        # ── Step 2: Fine tracking ────────────────────────────────────
        print("\n[2] Fine tracking (60 frames)")
        n = af.track(max_frames=60)
        print(f"   frames processed      : {n}")
        print(f"   final fine state      : {af.fine.state}")
        print(f"   stage cmd z           : {raw.position['z']}")
        print(f"   stage mech z          : {raw.mech_z}")

        fine_err = abs(raw.mech_z - cam.true_focus_z)
        print(f"   final mech error      : {fine_err} steps")
        assert fine_err <= 25, f"fine drifted away: error {fine_err}"
        assert af.fine.state in ("locked", "refine", "climb"), \
            f"unexpected state: {af.fine.state}"

        # ── Step 3: CSV exists and has content ───────────────────────
        print("\n[3] CSV output")
        assert os.path.isfile(csv_path), "CSV not written"
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        print(f"   csv path              : {csv_path}")
        print(f"   csv rows              : {len(rows)}  (1 header + N data)")
        # Header
        assert rows[0][0:3] == ["phase", "t", "stage_z"]
        coarse_rows = [r for r in rows[1:] if r[0] == "coarse"]
        fine_rows   = [r for r in rows[1:] if r[0] == "fine"]
        print(f"   coarse rows           : {len(coarse_rows)}")
        print(f"   fine rows             : {len(fine_rows)}")
        assert len(coarse_rows) > 5
        assert len(fine_rows)   > 30

        # ── Step 4: close() releases motors ──────────────────────────
        print("\n[4] close() releases motors")
        assert not raw._released
        af.close()
        assert raw._released, "stage.release() not called by af.close()"

        print("\n" + "=" * 64)
        print("  ALL CHECKS PASSED ✓")
        print(f"  coarse error: {coarse_err}  fine error: {fine_err}")
        print(f"  csv: {csv_path}")
        print("=" * 64)
    finally:
        time.sleep = real_sleep


if __name__ == "__main__":
    main()
