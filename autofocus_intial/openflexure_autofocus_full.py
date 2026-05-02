"""
openflexure_autofocus_full.py
==============================
Production wrapper around your UnifiedFocusMetricEngine. Adds:

    1. Coarse z-sweep autofocus  (improved version of yours)
    2. CONTINUOUS fine tracking  (handles drift, doesn't hunt)
    3. CSV telemetry logging     (every frame)
    4. Lifecycle management       (Ctrl-C releases motors, closes serial)
    5. Backlash-aware moves       (delegated to BacklashAwareStage)

This file IMPORTS your existing UnifiedFocusMetricEngine from
`openflexure_autofocus_fixed.py` unmodified — your metric / preprocess
logic is preserved exactly.

USAGE
─────
    from hardware import RealCamera, SerialStage, BacklashAwareStage
    from openflexure_autofocus_full import OpenFlexureAutofocusV2

    cam   = RealCamera(index=0)
    raw   = SerialStage(port="/dev/ttyACM0", baud=115200)
    stage = BacklashAwareStage(raw, backlash_steps=80, settle_s=0.20)

    af = OpenFlexureAutofocusV2(stage=stage, camera=cam,
                                csv_path="run.csv", debug=True)
    af.autofocus()                  # one-shot coarse sweep + park
    af.track(duration_s=300)        # 5 min of drift tracking
    af.close()                      # releases motors, flushes CSV
"""
from __future__ import annotations

import csv
import logging
import os
import signal
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Import unchanged from the user's existing file
from openflexure_autofocus_fixed import (
    UnifiedFocusMetricEngine,
    FocusSample,
    FocusMetrics,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# FINE-TRACKING CONTROLLER
# ══════════════════════════════════════════════════════════════════════
@dataclass
class _FineState:
    """Internal state of the fine-tracking controller."""
    direction:       int   = +1
    step:            int   = 4
    last_smoothed:   Optional[float] = None
    best_smoothed:   float = -1e30
    regress_count:   int   = 0
    stable_count:    int   = 0
    cooldown_left:   int   = 0
    state:           str   = "init"   # init | climb | refine | locked | hold


class FineTracker:
    """
    Drift-tracking controller that runs AFTER coarse sweep.

    DESIGN
    ──────
    Replaces your single-shot UnifiedFocusController for the tracking
    phase. Differences:

    1. ROLLING-MEDIAN SMOOTHING
       Decisions are based on the median of the last N scores, not the
       raw score. One bad frame doesn't trigger a reversal.

    2. DEADBAND
       |Δsmoothed| below `deadband` → no motion. Stops noise from
       chattering the motor.

    3. HYSTERESIS
       Direction flips only after `hysteresis_frames` *consecutive*
       regressions. Single-frame drops are ignored.

    4. COOLDOWN AFTER REVERSAL
       After flipping direction, suppress motion for `cooldown_frames`
       so backlash compensation has time to settle before the next
       decision is made.

    5. CONFIDENCE GATING
       If your metric engine reports low confidence, motion is held.
       This stops the controller from chasing focus during a momentary
       overexposure or when a particle drifts out of frame.

    6. LOCK + DRIFT DETECTION
       After `stable_required` frames inside the deadband, declare LOCK
       (zero motion). Re-engage CLIMB if the score later falls
       `drift_threshold` below the all-time best.

    Returns ONLY a `dz` per frame; it doesn't issue motor commands —
    the orchestrator does that. This keeps the controller pure-Python,
    side-effect-free, and easy to test.
    """

    def __init__(self,
                 coarse_step:        int   = 8,
                 fine_step:          int   = 2,
                 min_step:           int   = 1,
                 deadband:           float = 0.01,
                 hysteresis_frames:  int   = 3,
                 cooldown_frames:    int   = 2,
                 stable_required:    int   = 5,
                 drift_threshold:    float = 0.10,
                 confidence_floor:   float = 0.35,
                 smooth_window:      int   = 5):
        self.coarse_step = int(coarse_step)
        self.fine_step   = int(fine_step)
        self.min_step    = int(min_step)
        self.deadband    = float(deadband)
        self.hysteresis  = int(hysteresis_frames)
        self.cooldown_n  = int(cooldown_frames)
        self.stable_req  = int(stable_required)
        self.drift_thr   = float(drift_threshold)
        self.conf_floor  = float(confidence_floor)
        self._buf: deque[float] = deque(maxlen=int(smooth_window))
        self._s = _FineState(step=self.coarse_step)

    # ── helpers ────────────────────────────────────────────────────
    def reset(self) -> None:
        self._buf.clear()
        self._s = _FineState(step=self.coarse_step)

    @property
    def state(self) -> str: return self._s.state
    @property
    def best(self)  -> float: return self._s.best_smoothed

    @staticmethod
    def _median(buf: "deque[float]") -> float:
        if not buf: return 0.0
        s = sorted(buf); n = len(s); m = n // 2
        return s[m] if n % 2 else 0.5 * (s[m-1] + s[m])

    # ── decide ─────────────────────────────────────────────────────
    def update(self,
               combined_score: float,
               confidence: float = 1.0) -> int:
        """
        Submit one frame's combined focus score. Return signed dz.
        Caller is responsible for actually moving the stage.
        """
        s = self._s

        # 1. Confidence gate
        if confidence < self.conf_floor:
            s.state = "hold"
            return 0

        # 2. Smoothing
        self._buf.append(float(combined_score))
        smoothed = self._median(self._buf)
        if smoothed > s.best_smoothed:
            s.best_smoothed = smoothed

        # 3. Cooldown
        if s.cooldown_left > 0:
            s.cooldown_left -= 1
            s.last_smoothed = smoothed
            return 0

        # 4. INIT — first probe
        if s.state == "init":
            if s.last_smoothed is None:
                s.last_smoothed = smoothed
                s.state = "climb"
                return s.direction * s.step
            s.state = "climb"

        prev = s.last_smoothed
        delta = (smoothed - prev) if prev is not None else 0.0
        s.last_smoothed = smoothed

        # 5. Deadband + lock
        if abs(delta) < self.deadband:
            s.regress_count = 0
            s.stable_count += 1
            if (s.stable_count >= self.stable_req
                and s.state in ("climb", "refine")):
                s.state = "locked"
                return 0
            return 0
        else:
            s.stable_count = 0

        # 6. LOCKED — drift detection
        if s.state == "locked":
            if (s.best_smoothed > 0 and
                smoothed < s.best_smoothed * (1 - self.drift_thr)):
                s.state = "refine"
                s.step  = self.fine_step
                return s.direction * s.step
            return 0

        # 7. CLIMB / REFINE
        if delta > 0:
            s.regress_count = 0
            return s.direction * s.step

        # delta < 0 → regression
        s.regress_count += 1
        if s.regress_count < self.hysteresis:
            return s.direction * s.step

        # confirmed regression → flip direction, shrink step, cooldown
        s.direction *= -1
        if s.step > self.fine_step:
            s.step = self.fine_step
            s.state = "refine"
        elif s.step > self.min_step:
            s.step = max(self.min_step, s.step // 2)
        s.regress_count = 0
        s.cooldown_left = self.cooldown_n
        return s.direction * s.step


# ══════════════════════════════════════════════════════════════════════
# OPENFLEXURE AUTOFOCUS V2
# ══════════════════════════════════════════════════════════════════════
class OpenFlexureAutofocusV2:
    """
    Full autofocus pipeline:

        autofocus()  → coarse z-sweep, parabolic peak fit, park stage
        track(...)   → continuous drift-tracking until stop condition

    Backwards-compatible: an existing call site that just does
    `af.autofocus()` (your code) keeps working.

    Expects `stage` to either be a BacklashAwareStage (preferred) or
    anything with `position`, `move_relative(z=)`, `release()`, `close()`.
    """

    def __init__(
        self,
        stage,
        camera,
        # ── coarse sweep tunables ────────────────────────────────────
        sweep_range:      int   = 3000,
        sweep_step:       int   = 150,
        # ── fine tracker tunables ────────────────────────────────────
        fine_coarse_step: int   = 8,
        fine_step:        int   = 2,
        fine_deadband:    float = 0.01,
        fine_conf_floor:  float = 0.35,
        # ── timing ───────────────────────────────────────────────────
        track_period_s:   float = 0.10,    # ~10 Hz fine loop
        # ── output ───────────────────────────────────────────────────
        csv_path:         Optional[str] = None,
        debug:            bool  = False,
    ):
        self.stage  = stage
        self.camera = camera
        self.debug  = debug

        self.metric_engine = UnifiedFocusMetricEngine()

        self.sweep_range = int(sweep_range)
        self.sweep_step  = int(sweep_step)

        self.fine = FineTracker(
            coarse_step      = fine_coarse_step,
            fine_step        = fine_step,
            deadband         = fine_deadband,
            confidence_floor = fine_conf_floor,
        )
        self.track_period = float(track_period_s)

        self._csv_writer = None
        self._csv_file   = None
        if csv_path:
            self._open_csv(csv_path)

        self._stop_requested = False
        self._coarse_samples: List[FocusSample] = []
        self._coarse_curve:   Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────────────
    # CSV
    # ──────────────────────────────────────────────────────────────────
    def _open_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._csv_file = open(path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "phase", "t", "stage_z", "score", "confidence", "state",
            "tenengrad", "laplacian", "brenner", "counting", "jpeg",
        ])
        if self.debug:
            print(f"[autofocus] writing CSV → {path}")

    def _log_row(self,
                 phase: str, t: float, z: int,
                 score: float, conf: float, state: str,
                 metrics: FocusMetrics) -> None:
        if self._csv_writer is None: return
        self._csv_writer.writerow([
            phase, f"{t:.4f}", z,
            f"{score:.6f}", f"{conf:.4f}", state,
            f"{metrics.tenengrad:.4f}",
            f"{metrics.laplacian:.4f}",
            f"{metrics.brenner:.4f}",
            f"{metrics.counting_metric:.4f}",
            f"{metrics.jpeg_sharpness:.4f}",
        ])
        # Flush so the file is readable while the run is in progress
        if self._csv_file is not None:
            self._csv_file.flush()

    # ──────────────────────────────────────────────────────────────────
    # COARSE SWEEP  (uses BacklashAwareStage.move_absolute_z if available)
    # ──────────────────────────────────────────────────────────────────
    def _move_to_z(self, z: int) -> None:
        """
        Move to absolute z. Uses backlash-aware path if the stage
        provides `move_absolute_z`, else falls back to a naked
        relative move.
        """
        if hasattr(self.stage, "move_absolute_z"):
            self.stage.move_absolute_z(int(z))
        else:
            cur = self.stage.position["z"]
            self.stage.move_relative(z=int(z - cur))

    def autofocus(self) -> int:
        """
        Run a coarse z-sweep and park at the peak.
        Returns the absolute z chosen as best focus.
        """
        t0 = time.monotonic()
        cur = self.stage.position["z"]
        start = cur - self.sweep_range // 2

        samples: List[FocusSample] = []
        for z in np.arange(start, start + self.sweep_range, self.sweep_step):
            z = int(z)
            self._move_to_z(z)
            frame = self.camera.grab_frame()
            pf = self.metric_engine.evaluate(frame)
            samples.append(FocusSample(
                z=z,
                confidence=pf.metrics.confidence,
                metrics=pf.metrics,
            ))
            self._log_row("coarse", time.monotonic() - t0,
                          self.stage.position["z"],
                          score=self._combined_from_metrics(pf.metrics),
                          conf=pf.metrics.confidence,
                          state="sweep",
                          metrics=pf.metrics)
            if self.debug:
                print(f"  coarse z={z:+6d}  ten={pf.metrics.tenengrad:7.2f}  "
                      f"count={pf.metrics.counting_metric:7.0f}  "
                      f"conf={pf.metrics.confidence:.2f}")

        # Build combined curve and pick the peak
        ten   = self._normalise([s.metrics.tenengrad        for s in samples])
        count = self._normalise([s.metrics.counting_metric  for s in samples])
        lap   = self._normalise([s.metrics.laplacian        for s in samples])
        combined = 0.55 * ten + 0.35 * count + 0.10 * lap
        smooth = gaussian_filter1d(combined, sigma=1.2)
        # Ignore the first/last 3 (edge artifacts of the smoother)
        valid = smooth[3:-3]
        peak_idx = int(np.argmax(valid)) + 3 if len(valid) else int(np.argmax(smooth))
        peak_z = samples[peak_idx].z

        # Sub-step parabolic refinement on the three samples around peak
        if 0 < peak_idx < len(samples) - 1:
            x0, x1, x2 = samples[peak_idx-1].z, samples[peak_idx].z, samples[peak_idx+1].z
            y0, y1, y2 = smooth[peak_idx-1],   smooth[peak_idx],   smooth[peak_idx+1]
            denom = (x0 - x1)*(x0 - x2)*(x1 - x2)
            if abs(denom) > 1e-9:
                a = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
                b = (x2*x2*(y0-y1) + x1*x1*(y2-y0) + x0*x0*(y1-y2)) / denom
                if abs(a) > 1e-12:
                    peak_z = int(round(-b / (2*a)))

        if self.debug:
            print(f"  → peak z = {peak_z}")

        # Park there
        self._move_to_z(peak_z)

        # Reset fine tracker so it starts fresh from this position
        self.fine.reset()
        self._coarse_samples = samples
        self._coarse_curve   = smooth
        return peak_z

    # ──────────────────────────────────────────────────────────────────
    # FINE TRACKING
    # ──────────────────────────────────────────────────────────────────
    def track(self,
              duration_s: Optional[float] = None,
              max_frames: Optional[int]   = None,
              on_frame:   Optional[Callable] = None) -> int:
        """
        Run the continuous drift-tracking loop. Returns number of frames
        processed.

        Stops when:
          * duration_s elapsed
          * max_frames reached
          * Ctrl-C received
          * self._stop_requested set true (e.g. via shutdown handler)

        on_frame(z, score, conf, state, metrics) is called every cycle
        if provided — useful for live overlays.
        """
        self._install_signal_handlers()
        t0 = time.monotonic()
        frames = 0

        while not self._stop_requested:
            if duration_s is not None and (time.monotonic() - t0) >= duration_s:
                if self.debug: print("[autofocus] track: duration reached")
                break
            if max_frames is not None and frames >= max_frames:
                if self.debug: print("[autofocus] track: frame limit")
                break

            # Acquire
            try:
                frame = self.camera.grab_frame()
            except Exception as e:
                logger.error(f"[autofocus] camera read failed: {e}")
                break

            # Score
            pf = self.metric_engine.evaluate(frame)
            score = self._combined_from_metrics(pf.metrics)
            conf  = pf.metrics.confidence

            # Decide
            dz = self.fine.update(score, conf)

            # Act
            if dz != 0:
                self.stage.move_relative(z=int(dz))

            # Log
            self._log_row("fine", time.monotonic() - t0,
                          self.stage.position["z"],
                          score=score, conf=conf, state=self.fine.state,
                          metrics=pf.metrics)

            if on_frame is not None:
                try:
                    on_frame(self.stage.position["z"], score, conf,
                             self.fine.state, pf.metrics)
                except Exception as e:
                    logger.warning(f"on_frame callback raised: {e}")

            frames += 1
            time.sleep(self.track_period)

        if self.debug:
            print(f"[autofocus] track: {frames} frames, "
                  f"final state={self.fine.state}, "
                  f"final z={self.stage.position['z']}")
        return frames

    # ──────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────────────────
    def _install_signal_handlers(self) -> None:
        def handler(sig, _frame):
            logger.warning(f"[autofocus] signal {sig} → stopping")
            self._stop_requested = True
        try:
            signal.signal(signal.SIGINT,  handler)
            signal.signal(signal.SIGTERM, handler)
        except (ValueError, AttributeError):
            pass

    def close(self) -> None:
        """Release motors, close serial, flush CSV. Always safe to call."""
        try: self.stage.release()
        except Exception: pass
        try: self.stage.close()
        except Exception: pass
        try: self.camera.close()
        except Exception: pass
        if self._csv_file:
            try: self._csv_file.close()
            except Exception: pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    # ──────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _normalise(x) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32)
        if a.size == 0: return a
        return (a - a.min()) / (a.max() - a.min() + 1e-6)

    @staticmethod
    def _combined_from_metrics(m: FocusMetrics) -> float:
        """
        Same recipe as the coarse-curve combiner but for ONE frame.
        Used by the fine tracker, which has only the current frame's
        metrics (not a curve). Heuristic normalisation by typical
        bright-field values keeps the score in [0, ~1].
        """
        ten   = min(1.0, m.tenengrad        / 100.0)
        count = min(1.0, m.counting_metric  / 30000.0)
        lap   = min(1.0, m.laplacian        / 500.0)
        return 0.55 * ten + 0.35 * count + 0.10 * lap
