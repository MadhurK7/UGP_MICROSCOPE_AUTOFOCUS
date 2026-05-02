"""
autofocus/system.py
===================
Full closed-loop autofocus orchestrator.

ARCHITECTURE
────────────

    ┌──────────┐   frame    ┌──────────────┐   metrics  ┌──────────────┐
    │  Camera  ├──────────► │ Preprocessor ├──────────► │  MetricBank  │
    └──────────┘            └──────────────┘            └──────┬───────┘
                                                               │
                                                               ▼
                                                        ┌──────────────┐
                                                        │  Combiner +  │
                                                        │ Confidence   │
                                                        └──────┬───────┘
                                                               │ (score, conf)
                            ┌──────────────────────────────────┤
                            │                                  │
                            ▼                                  ▼
                  ┌──────────────────┐               ┌──────────────────┐
                  │ CoarseSweep (1×) │               │ FineController   │
                  │ z-sweep + park   │               │  (every frame)   │
                  └────────┬─────────┘               └────────┬─────────┘
                           │ stage.move_z(dz)                 │ stage.move_z(dz)
                           ▼                                  ▼
                                ┌─────────────────────┐
                                │   BaseStage (USB)   │
                                └─────────────────────┘

WORKFLOW
────────
1. `coarse_focus()`  — runs once. Sweeps over a wide z range, finds the
                       global peak, parks the stage there from below.
2. `track()`         — main loop. Every frame: preprocess → metrics →
                       combine → confidence → fine controller → stage.
                       Exits cleanly on Ctrl-C, returning a session log.

THREADING NOTES (Raspberry Pi)
──────────────────────────────
On Pi 4 the inner loop completes in roughly 8–14 ms (preprocess ≈ 5 ms,
metrics ≈ 3 ms, controller < 1 ms, motor command 5–30 ms when moves are
issued). At 30 FPS the budget is 33.3 ms, so the loop runs single-
threaded comfortably.

If a heavier camera or larger frames push the budget, the system can be
split:
    Thread A (camera)     : producer, drops oldest frame on full queue
    Thread B (autofocus)  : consumer, runs preprocess+metrics+controller
    Thread C (display)    : optional, reads latest result snapshot

Currently we run single-threaded for simplicity — the architecture
permits the split with no API change.

SERIAL & MOTOR
──────────────
The OpenFlexure firmware is reply-driven: each `mr` command blocks on
"done." before the next can be issued. This naturally rate-limits the
control loop and prevents queue overflow.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .preprocessor import Preprocessor
from .metrics      import MetricBank, AdaptiveCombiner, ConfidenceEstimator
from .coarse       import CoarseSweepAutofocus, CoarseResult
from .fine         import FineFocusController, FineState, FineDecision
from .stage_iface  import BaseStage, NullStage

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
@dataclass
class FrameRecord:
    """One frame of session telemetry — captured for validation."""
    t:           float
    cum_z:       int
    score:       float
    confidence:  float
    state:       str
    condition:   str
    metrics:     Dict[str, float] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
class AutofocusSystem:
    """
    Top-level orchestrator. Owns one of every component.

    Usage:
        af = AutofocusSystem(
            grab_frame = camera.read,
            stage      = SerialStage(port="/dev/ttyACM0"),
        )

        coarse_result = af.coarse_focus()
        af.track(stop_after_seconds=120)
        records = af.session_log
    """

    def __init__(
        self,
        grab_frame:  Callable[[], np.ndarray],
        stage:       Optional[BaseStage] = None,
        # ── component overrides (else default-constructed) ───────────
        preprocessor: Optional[Preprocessor]            = None,
        metric_bank:  Optional[MetricBank]              = None,
        combiner:     Optional[AdaptiveCombiner]        = None,
        confidence:   Optional[ConfidenceEstimator]     = None,
        fine:         Optional[FineFocusController]     = None,
        # ── coarse sweep tunables (used by coarse_focus()) ───────────
        coarse_range_steps: int   = 400,
        coarse_n_samples:   int   = 21,
        coarse_settle_s:    float = 0.20,
        coarse_backlash:    int   = 120,
        # ── fine loop tunables ───────────────────────────────────────
        fine_settle_s:      float = 0.05,
        # ── safety ───────────────────────────────────────────────────
        max_session_dz:     int   = 4000,    # absolute |cum_z| safety stop
    ):
        self.grab_frame    = grab_frame
        self.stage         = stage or NullStage()
        self.pp            = preprocessor or Preprocessor()
        self.bank          = metric_bank  or MetricBank()
        self.combiner      = combiner     or AdaptiveCombiner()
        self.conf_est      = confidence   or ConfidenceEstimator()
        self.fine          = fine         or FineFocusController()
        self.coarse_range  = coarse_range_steps
        self.coarse_n      = coarse_n_samples
        self.coarse_settle = coarse_settle_s
        self.coarse_back   = coarse_backlash
        self.fine_settle   = fine_settle_s
        self.max_session_dz = max_session_dz

        self.session_log: List[FrameRecord] = []
        self._cum_dz_total = 0       # session-wide; used by track()

    # ──────────────────────────────────────────────────────────────────
    def _score_one(self, frame: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Pipeline a frame from raw → (score, info) for the controllers."""
        r = self.pp.process(frame)
        m = self.bank.compute(r.enhanced, r.valid_mask)
        s = self.combiner.combine(m, r.condition["type"])
        c = self.conf_est.estimate(m, r.condition)
        info = {
            "metrics":    m,
            "condition":  r.condition,
            "confidence": c,
            "preproc":    r,
        }
        return s, info

    # ──────────────────────────────────────────────────────────────────
    # COARSE FOCUS — run once at session start
    # ──────────────────────────────────────────────────────────────────
    def coarse_focus(self) -> CoarseResult:
        """
        Execute one OpenFlexure-style symmetric sweep, then park the
        stage at the discovered peak (with backlash compensation).
        """
        logger.info("[autofocus] running coarse sweep…")
        cs = CoarseSweepAutofocus(
            stage_move    = self.stage.move_z,
            grab_frame    = self.grab_frame,
            score_fn      = self._score_one,
            range_steps   = self.coarse_range,
            n_samples     = self.coarse_n,
            settle_s      = self.coarse_settle,
            backlash_steps= self.coarse_back,
            log_fn        = lambda s: logger.debug(s),
        )
        result = cs.sweep()
        logger.info(
            f"[autofocus] coarse done — success={result.success} "
            f"best_z={result.best_z} snr={result.snr:.2f} "
            f"prom={result.prominence:.2f}"
        )
        # Reset the fine controller's cumulative origin to wherever
        # the coarse sweep parked us.
        self.fine.reset()
        self._cum_dz_total = result.best_z if result.success else 0
        return result

    # ──────────────────────────────────────────────────────────────────
    # FINE TRACKING — main loop
    # ──────────────────────────────────────────────────────────────────
    def track(self,
              stop_after_seconds: Optional[float] = None,
              stop_after_frames:  Optional[int]   = None,
              on_frame: Optional[Callable[[FrameRecord, Dict[str, Any]], None]] = None,
             ) -> List[FrameRecord]:
        """
        Run the closed-loop fine controller until a stop condition trips
        or KeyboardInterrupt is raised. Returns the session log.

        on_frame: optional callback invoked with (record, info) for live
                  display, plotting, or external logging.
        """
        logger.info("[autofocus] entering tracking loop")
        t0 = time.monotonic()
        try:
            while True:
                t_now = time.monotonic()
                if stop_after_seconds is not None and (t_now - t0) >= stop_after_seconds:
                    logger.info("[autofocus] stop: time limit")
                    break
                if stop_after_frames is not None and len(self.session_log) >= stop_after_frames:
                    logger.info("[autofocus] stop: frame limit")
                    break

                frame = self.grab_frame()
                if frame is None:
                    time.sleep(self.fine_settle)
                    continue

                score, info = self._score_one(frame)
                conf  = info["confidence"]
                cond  = info["condition"]["type"]
                decision = self.fine.update(score=score, confidence=conf)

                # SAFETY: absolute session stop
                proposed_total = self._cum_dz_total + decision.dz
                if abs(proposed_total) > self.max_session_dz:
                    logger.warning("[autofocus] cum_dz safety bound hit; halting")
                    break

                if decision.dz != 0:
                    ok = self.stage.move_z(decision.dz)
                    if ok:
                        self._cum_dz_total += decision.dz
                    if self.fine_settle > 0:
                        time.sleep(self.fine_settle)

                rec = FrameRecord(
                    t=t_now - t0,
                    cum_z=self._cum_dz_total,
                    score=score,
                    confidence=conf,
                    state=decision.state.value,
                    condition=cond,
                    metrics=dict(info["metrics"]),
                )
                self.session_log.append(rec)
                if on_frame:
                    try:
                        on_frame(rec, info)
                    except Exception as e:
                        logger.warning(f"[autofocus] on_frame callback error: {e}")
        except KeyboardInterrupt:
            logger.info("[autofocus] interrupted by user")
        finally:
            try:
                self.stage.release()
            except Exception:
                pass
        return list(self.session_log)
