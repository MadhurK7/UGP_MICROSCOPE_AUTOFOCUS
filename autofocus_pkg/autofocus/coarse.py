"""
autofocus/coarse.py
===================
Coarse autofocus: OpenFlexure-style symmetric z-sweep.

WHY SWEEP, NOT HILL-CLIMB?
──────────────────────────
Hill-climbing fails on real microscopes because the focus surface
F(z) is NOT monotonic on either side of the peak:

    1. Diffraction halos: translucent colloids show secondary local
       maxima at ±λ/(NA²) on either side of true focus.
    2. Sensor-noise plateaus: well below focus the score is dominated
       by Poisson noise — score is roughly constant for a range of z,
       and a hill-climber will report convergence at the wrong place.
    3. Backlash: stepper motors with belt drive lose 50–500 steps when
       reversing direction. A naive climber that flips direction near
       a noise plateau will believe it has found the peak when it has
       only consumed backlash.

A SWEEP (uniform z-grid scan) avoids all three: it samples the entire
plausible focus range, then we fit a smooth curve and pick the global
maximum. The cost is more motion, but on a calibrated rig this only
takes a few seconds and is run once per session.

OPENFLEXURE'S APPROACH
──────────────────────
OpenFlexure's autofocus_v2 (in microscope.py) implements:
    1. Capture-and-store JPEG sharpness at each z position.
    2. Move via a SINGLE direction to avoid backlash; if the search
       must reverse, it overshoots and approaches from the original
       direction.
    3. After the sweep, the index of the maximum sharpness is selected
       and the stage is moved there from below (again, one direction).
    4. A settle delay is inserted after every move so the camera AGC
       and the mechanical stage have stabilised before sharpness is
       sampled.

This module reproduces those behaviours and adds:
    a. Multi-metric scoring (delegated to metrics.py).
    b. Parabolic peak fit on the three samples around the argmax for
       sub-step resolution.
    c. Curve-quality checks: peak prominence and SNR. If the sweep
       returned a noisy curve, we report failure rather than locking
       onto a phantom peak.

BACKLASH COMPENSATION DETAILS
─────────────────────────────
We always ENTER the sweep range from below. If the current stage z is
above the lower bound, we first move down past it by `backlash_steps`,
then climb up. Every sample point is reached in the same direction.
The final move-to-peak is also from below. This guarantees consistent
mechanical conditions across the entire sweep.

If the user passes a known `direction_of_approach` to the constructor
(say, the rig is mounted upside down), the logic is mirrored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

# Type alias for the user-supplied stage callback
StageMoveFn  = Callable[[int], None]   # (dz_signed_steps) -> None
FrameGrabFn  = Callable[[], np.ndarray]
ScoreFn      = Callable[[np.ndarray], Tuple[float, dict]]
# ScoreFn returns (score, info_dict) where info_dict can hold metrics


@dataclass
class CoarseResult:
    """Outcome of a coarse sweep."""
    success:           bool
    best_z:            int                  # cumulative dz at peak
    best_score:        float
    z_samples:         List[int]            # sampled z positions (cumulative dz)
    score_samples:     List[float]          # focus score at each sample
    parabolic_peak:    Optional[float]      # sub-step peak estimate (cumulative dz)
    sweep_range:       Tuple[int, int]      # (z_min, z_max) used
    snr:               float                # peak / median ratio
    prominence:        float                # peak − second-peak gap (normalised)
    diagnostic:        str = ""             # human-readable status


class CoarseSweepAutofocus:
    """
    Sweep-based autofocus controller.

    Workflow:
        cs = CoarseSweepAutofocus(
            stage_move=stage.relative,
            grab_frame=camera.read,
            score_fn=score_one,            # uses metrics.py + preprocessor
            range_steps=400, n_samples=21,
            settle_s=0.20, backlash_steps=120,
        )
        result = cs.sweep()
        if result.success:
            print(f"focused at cumulative z={result.best_z}")

    The class holds NO long-term state besides what's needed for one sweep.
    """

    def __init__(
        self,
        stage_move:        StageMoveFn,
        grab_frame:        FrameGrabFn,
        score_fn:          ScoreFn,
        # ── sweep geometry ────────────────────────────────────────────
        range_steps:       int   = 400,    # total z extent covered (steps)
        n_samples:         int   = 21,     # number of samples in the sweep
        # ── timing ────────────────────────────────────────────────────
        settle_s:          float = 0.20,   # delay after each move
        # ── backlash ──────────────────────────────────────────────────
        backlash_steps:    int   = 120,    # over-travel distance
        direction_of_approach: int = +1,   # ±1; +1 means approach from below
        # ── peak-quality gates ────────────────────────────────────────
        min_snr:           float = 1.20,   # peak must exceed median by 20 %
        min_prominence:    float = 0.05,   # 5 % gap above 2nd-best
        # ── safety ────────────────────────────────────────────────────
        per_step_clamp:    int   = 600,    # max single move issued
        log_fn:            Optional[Callable[[str], None]] = None,
    ):
        self.stage_move   = stage_move
        self.grab_frame   = grab_frame
        self.score_fn     = score_fn
        self.range_steps  = int(range_steps)
        self.n_samples    = int(n_samples)
        self.settle_s     = float(settle_s)
        self.backlash     = int(backlash_steps)
        self.dir_approach = +1 if direction_of_approach >= 0 else -1
        self.min_snr      = float(min_snr)
        self.min_prom     = float(min_prominence)
        self.per_step_clamp = int(per_step_clamp)
        self._log = log_fn or (lambda s: None)

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC: run the sweep
    # ──────────────────────────────────────────────────────────────────
    def sweep(self) -> CoarseResult:
        """
        Execute the symmetric sweep.

        Sweep is centred on the CURRENT z, extending ±range/2 either side.
        After the sweep the stage is parked at the best-fit peak,
        approaching from `direction_of_approach`.
        """
        import time

        half = self.range_steps // 2
        # Sample positions, in cumulative-dz space, monotonic in approach dir.
        # Internally we always sample from -half to +half (low → high) and
        # then the final move handles direction.
        sample_dz = np.linspace(-half, +half, self.n_samples).astype(int).tolist()

        # Move to the start of the sweep with backlash compensation.
        # Current cumulative dz inside this routine is tracked locally.
        cum_dz = 0
        # Step 1: overshoot the start by `backlash` steps in the OPPOSITE
        # direction of approach, then approach the start.
        target_start = sample_dz[0]                       # most negative
        overshoot    = target_start - self.dir_approach * self.backlash
        cum_dz = self._move_to(cum_dz, overshoot)
        cum_dz = self._move_to(cum_dz, target_start)
        time.sleep(self.settle_s)

        # Step 2: walk through each sample point in ascending order.
        scores: List[float] = []
        positions: List[int] = []
        for tgt in sample_dz:
            cum_dz = self._move_to(cum_dz, tgt)
            time.sleep(self.settle_s)
            frame = self.grab_frame()
            if frame is None:
                self._log(f"[coarse] frame grab failed at z={tgt}")
                scores.append(np.nan)
                positions.append(tgt)
                continue
            s, _info = self.score_fn(frame)
            scores.append(float(s))
            positions.append(tgt)
            self._log(f"[coarse] z={tgt:+5d}  score={s:.4f}")

        # ── Curve analysis ───────────────────────────────────────────
        scores_arr = np.asarray(scores, dtype=np.float64)
        positions_arr = np.asarray(positions, dtype=np.int64)
        finite = np.isfinite(scores_arr)
        if finite.sum() < 5:
            return CoarseResult(
                success=False, best_z=0, best_score=0.0,
                z_samples=positions, score_samples=scores,
                parabolic_peak=None, sweep_range=(sample_dz[0], sample_dz[-1]),
                snr=0.0, prominence=0.0,
                diagnostic="too few valid samples",
            )

        valid_scores = scores_arr[finite]
        valid_pos    = positions_arr[finite]

        idx_max = int(np.argmax(valid_scores))
        peak_score = float(valid_scores[idx_max])
        peak_pos   = int(valid_pos[idx_max])

        median = float(np.median(valid_scores))
        snr = peak_score / max(median, 1e-9)

        # 2nd-best for prominence (anywhere outside ±1 sample of peak)
        mask = np.ones_like(valid_scores, dtype=bool)
        if idx_max - 1 >= 0:           mask[idx_max - 1] = False
        mask[idx_max] = False
        if idx_max + 1 < len(mask):    mask[idx_max + 1] = False
        if mask.any():
            second = float(valid_scores[mask].max())
            prominence = (peak_score - second) / max(peak_score, 1e-9)
        else:
            prominence = 1.0

        # Parabolic refinement using neighbours of the discrete max
        parabolic_peak: Optional[float] = None
        if 0 < idx_max < len(valid_scores) - 1:
            parabolic_peak = self._parabola_apex(
                valid_pos[idx_max - 1], valid_scores[idx_max - 1],
                valid_pos[idx_max],     valid_scores[idx_max],
                valid_pos[idx_max + 1], valid_scores[idx_max + 1],
            )

        # Move the stage to the discrete best position with backlash comp,
        # always approaching from the chosen direction.
        cum_dz = self._move_to_with_backlash(cum_dz, peak_pos)
        time.sleep(self.settle_s)

        # ── Quality gates ────────────────────────────────────────────
        ok = (snr >= self.min_snr) and (prominence >= self.min_prom)
        diag = "ok" if ok else (
            f"poor curve quality  snr={snr:.2f} prom={prominence:.2f}"
        )

        return CoarseResult(
            success=ok,
            best_z=peak_pos,
            best_score=peak_score,
            z_samples=positions,
            score_samples=scores,
            parabolic_peak=parabolic_peak,
            sweep_range=(sample_dz[0], sample_dz[-1]),
            snr=float(snr),
            prominence=float(prominence),
            diagnostic=diag,
        )

    # ──────────────────────────────────────────────────────────────────
    # MOVEMENT HELPERS
    # ──────────────────────────────────────────────────────────────────
    def _move_to(self, cur: int, target: int) -> int:
        """Issue clamped relative moves until we reach `target`."""
        remaining = target - cur
        while remaining != 0:
            step = max(-self.per_step_clamp,
                       min(self.per_step_clamp, remaining))
            self.stage_move(step)
            cur += step
            remaining = target - cur
        return cur

    def _move_to_with_backlash(self, cur: int, target: int) -> int:
        """Approach `target` from `direction_of_approach` with overshoot."""
        overshoot_pos = target - self.dir_approach * self.backlash
        cur = self._move_to(cur, overshoot_pos)
        cur = self._move_to(cur, target)
        return cur

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _parabola_apex(x0: float, y0: float,
                       x1: float, y1: float,
                       x2: float, y2: float) -> Optional[float]:
        """
        Fit y = a x² + b x + c through three points and return the x of
        the apex. Returns None if points are degenerate.
        """
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) < 1e-9:
            return None
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
        if abs(a) < 1e-12:
            return None
        return float(-b / (2 * a))
