"""
autofocus/fine.py
==================
Fine, drift-tracking autofocus controller for use AFTER coarse sweep.

DESIGN PHILOSOPHY
─────────────────
Coarse sweep finds the peak once. The fine controller's job is to:
    1. Refine the discrete peak to sub-step accuracy.
    2. Track slow focal drift (sample heating, mechanical creep,
       liquid evaporation, cell sedimentation in colloidal samples).

It must do this WITHOUT:
    - oscillating around the peak when noise dominates.
    - reacting to single-frame outliers (transient particle motion).
    - moving when the score is unreliable (saturated/dim frames).
    - "winding up" mechanical backlash from rapid direction changes.

CONTROL FEATURES (each addresses a specific failure mode)
─────────────────────────────────────────────────────────
1. DEADBAND on score-delta
   |Δscore| < deadband → motion suppressed.
   Stops single-pixel sensor noise from triggering motor steps.

2. HYSTERESIS on direction reversal
   Direction flips only after `hysteresis_frames` consecutive
   regression frames. One bad frame is forgiven.

3. ADAPTIVE STEP SIZE
   step shrinks geometrically as we approach the peak: the controller
   uses `coarse_step` while score is changing rapidly, halves it on
   each plateau, down to `min_step`. Mirrors a binary search.

4. CONFIDENCE GATING
   If the metrics framework reports low confidence (saturated, dim,
   empty frame), the controller returns 0 — the motor never moves
   when we don't trust the score.

5. STABILITY DETECTOR / LOCK
   `stable_required` consecutive in-deadband frames declares LOCK.
   Once LOCKED the controller holds position. Re-engages only on
   clear regression (score < best × (1 - drift_threshold)).

6. SMOOTHED SCORE (rolling median)
   The decision input is a rolling median of recent scores, not the
   raw score. Median is robust to a single outlier frame; mean would
   blend the outlier in.

7. ANTI-WINDUP COOLDOWN
   After every direction reversal, the next move is delayed by
   `cooldown_frames` to allow mechanical backlash to be consumed
   without further direction changes confusing the score.

TUNING DZ
─────────
The coarse step should be ≈ 1/8 of your stage's depth-of-field in
motor steps. Too big → the controller skips over the peak. Too small
→ slow tracking. The fine step should be at the limit of your
mechanical resolution (1–4 steps for a 28BYJ-48 with 1:64 gearing).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, List, Tuple


# ──────────────────────────────────────────────────────────────────────
class FineState(str, Enum):
    INIT     = "init"
    CLIMB    = "climb"      # actively walking toward peak
    REFINE   = "refine"     # near peak, small steps
    LOCKED   = "locked"     # in-focus, holding
    HOLD     = "hold"       # bad frame / low confidence


@dataclass
class FineDecision:
    """One controller cycle output."""
    dz:        int
    state:     FineState
    smoothed:  float
    delta:     float
    reason:    str = ""


# ══════════════════════════════════════════════════════════════════════
class FineFocusController:
    """
    Drift-tracking autofocus controller.

    Usage:
        ctl = FineFocusController(coarse_step=8, fine_step=2, min_step=1)
        for each frame:
            decision = ctl.update(score=raw_score, confidence=conf)
            if decision.dz != 0:
                stage.move_z(decision.dz)
    """

    def __init__(
        self,
        # ── step sizes (motor steps) ─────────────────────────────────
        coarse_step:        int   = 8,
        fine_step:          int   = 2,
        min_step:           int   = 1,
        # ── deadband / hysteresis ────────────────────────────────────
        deadband:           float = 0.005,   # |Δscore_n| below this → no move
        hysteresis_frames:  int   = 3,
        cooldown_frames:    int   = 2,
        # ── stability ────────────────────────────────────────────────
        stable_required:    int   = 5,
        drift_threshold:    float = 0.10,    # 10 % below best → re-engage
        # ── confidence ───────────────────────────────────────────────
        confidence_floor:   float = 0.35,
        # ── smoothing ────────────────────────────────────────────────
        smooth_window:      int   = 5,
        # ── safety ───────────────────────────────────────────────────
        per_command_clamp:  int   = 32,
    ):
        self.coarse_step      = int(coarse_step)
        self.fine_step        = int(fine_step)
        self.min_step         = int(min_step)
        self.deadband         = float(deadband)
        self.hysteresis_n     = int(hysteresis_frames)
        self.cooldown_n       = int(cooldown_frames)
        self.stable_required  = int(stable_required)
        self.drift_thresh     = float(drift_threshold)
        self.conf_floor       = float(confidence_floor)
        self.smooth_window    = int(smooth_window)
        self.clamp            = int(per_command_clamp)

        self.reset()

    # ──────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset state machine without touching tunables."""
        self._state         = FineState.INIT
        self._direction     = +1
        self._step          = self.coarse_step
        self._last_smoothed = None
        self._best_smoothed = -1e30
        self._best_at_dz    = 0
        self._cum_dz        = 0
        self._regress_n     = 0
        self._stable_n      = 0
        self._cooldown_left = 0
        self._score_buf: Deque[float] = deque(maxlen=self.smooth_window)
        self._frame_count   = 0
        self._history: List[Tuple[int, float, FineState]] = []

    # ──────────────────────────────────────────────────────────────────
    def update(self,
               score: float,
               confidence: float = 1.0) -> FineDecision:
        """
        Submit one frame's focus score and confidence; receive a
        recommended dz.

        The caller is responsible for issuing the motor command
        (we don't reach the hardware here — keeps testing pure).
        """
        self._frame_count += 1

        # ── 1. Confidence gate ───────────────────────────────────────
        if confidence < self.conf_floor:
            self._state = FineState.HOLD
            return FineDecision(dz=0, state=self._state,
                                smoothed=self._last_smoothed or 0.0,
                                delta=0.0,
                                reason=f"low confidence {confidence:.2f}")

        # ── 2. Smoothing (rolling median) ────────────────────────────
        self._score_buf.append(float(score))
        smoothed = self._median(self._score_buf)

        # Track best-ever
        if smoothed > self._best_smoothed:
            self._best_smoothed = smoothed
            self._best_at_dz    = self._cum_dz

        # ── 3. Cooldown check ────────────────────────────────────────
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._last_smoothed = smoothed
            return FineDecision(dz=0, state=self._state,
                                smoothed=smoothed, delta=0.0,
                                reason="cooldown after reversal")

        # ── 4. INIT → CLIMB ──────────────────────────────────────────
        if self._state == FineState.INIT:
            self._state = FineState.CLIMB
            self._last_smoothed = smoothed
            return self._issue(self._direction * self._step,
                               smoothed, 0.0, "first probe")

        # ── 5. Compute delta vs previous smoothed score ──────────────
        prev = self._last_smoothed
        delta = smoothed - prev

        # ── 6. Deadband test ─────────────────────────────────────────
        if abs(delta) < self.deadband:
            self._stable_n += 1
            self._regress_n = 0
            if (self._stable_n >= self.stable_required
                and self._state in (FineState.CLIMB, FineState.REFINE)):
                self._state = FineState.LOCKED
                self._last_smoothed = smoothed
                return FineDecision(dz=0, state=self._state,
                                    smoothed=smoothed, delta=delta,
                                    reason="LOCK acquired")
            self._last_smoothed = smoothed
            return FineDecision(dz=0, state=self._state,
                                smoothed=smoothed, delta=delta,
                                reason="deadband")
        else:
            self._stable_n = 0

        # ── 7. State-specific decisions ──────────────────────────────
        if self._state == FineState.LOCKED:
            # drifted away from best?
            if (self._best_smoothed > 0 and
                smoothed < self._best_smoothed * (1 - self.drift_thresh)):
                self._state = FineState.REFINE
                self._step = self.fine_step
                self._last_smoothed = smoothed
                return self._issue(self._direction * self._step,
                                   smoothed, delta, "drift detected")
            self._last_smoothed = smoothed
            return FineDecision(dz=0, state=self._state,
                                smoothed=smoothed, delta=delta,
                                reason="locked")

        if delta > 0:
            # improving — keep going
            self._regress_n = 0
            self._last_smoothed = smoothed
            return self._issue(self._direction * self._step,
                               smoothed, delta, "improving")

        # delta < 0 → regression
        self._regress_n += 1
        if self._regress_n < self.hysteresis_n:
            # one or two bad frames are not enough — keep going
            self._last_smoothed = smoothed
            return self._issue(self._direction * self._step,
                               smoothed, delta,
                               f"regression {self._regress_n}/{self.hysteresis_n}")

        # confirmed regression: flip direction, shrink step, set cooldown
        self._direction *= -1
        if self._step > self.fine_step:
            self._step = self.fine_step
            self._state = FineState.REFINE
        elif self._step > self.min_step:
            self._step = max(self.min_step, self._step // 2)
        self._regress_n = 0
        self._cooldown_left = self.cooldown_n
        self._last_smoothed = smoothed
        return self._issue(self._direction * self._step,
                           smoothed, delta, "REVERSE + cooldown")

    # ──────────────────────────────────────────────────────────────────
    def _issue(self, raw_dz: int, smoothed: float, delta: float,
               reason: str) -> FineDecision:
        """Apply per-command clamp and update cumulative dz."""
        dz = max(-self.clamp, min(self.clamp, int(raw_dz)))
        self._cum_dz += dz
        self._history.append((self._cum_dz, smoothed, self._state))
        return FineDecision(dz=dz, state=self._state,
                            smoothed=smoothed, delta=delta, reason=reason)

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _median(buf: Deque[float]) -> float:
        if not buf:
            return 0.0
        s = sorted(buf)
        n = len(s)
        m = n // 2
        if n % 2:
            return s[m]
        return 0.5 * (s[m - 1] + s[m])

    # ──────────────────────────────────────────────────────────────────
    @property
    def state(self) -> FineState:           return self._state
    @property
    def cum_dz(self) -> int:                return self._cum_dz
    @property
    def best_score(self) -> float:          return self._best_smoothed
    @property
    def history(self):                      return list(self._history)
