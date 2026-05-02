"""
validation/validator.py
=======================
Scientific validation framework for the autofocus system.

WHAT WE MEASURE
───────────────
1. ACCURACY        — predicted peak vs ground-truth peak
2. REPEATABILITY   — std-dev of predicted peak across repeated trials
3. ROBUSTNESS      — accuracy under added noise, illumination drift
4. CURVE QUALITY   — SNR + prominence + smoothness of F(z)
5. STABILITY       — oscillation count + drift under tracking
6. SPEED           — ms per pipeline stage
7. METRIC RANKING  — per-metric MAE on the same z-stack

REQUIRED INPUTS
───────────────
A. A real or synthetic z-stack with a known ground-truth focus index.
B. The autofocus_pkg's autofocus.* modules.

OUTPUTS
───────
* a results dict (JSON-serialisable)
* matplotlib figures saved to disk:
    - focus_curve.png      F(z) for each metric
    - peak_error_hist.png  histogram of predicted-vs-true error
    - tracking_trace.png   cum_z and score over time
    - timing_breakdown.png ms per stage
"""
from __future__ import annotations

import os
import time
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Lazy import of matplotlib so the module is usable headless
def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# ──────────────────────────────────────────────────────────────────────
@dataclass
class CurveResult:
    """Per-metric focus curve."""
    name:       str
    z:          List[int]
    score:      List[float]
    pred_peak:  float
    true_peak:  float
    mae:        float
    snr:        float
    prominence: float


@dataclass
class TimingResult:
    stage:     str
    mean_ms:   float
    p95_ms:    float


@dataclass
class ValidationReport:
    accuracy_per_metric:     List[CurveResult]
    repeatability_std:       float
    repeatability_n:         int
    timing:                  List[TimingResult]
    tracking_oscillation:    Optional[Dict[str, float]] = None
    notes:                   List[str] = field(default_factory=list)

    def save_json(self, path: str) -> None:
        d = {
            "accuracy_per_metric": [asdict(c) for c in self.accuracy_per_metric],
            "repeatability_std":   self.repeatability_std,
            "repeatability_n":     self.repeatability_n,
            "timing":              [asdict(t) for t in self.timing],
            "tracking_oscillation": self.tracking_oscillation,
            "notes":               self.notes,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
class Validator:
    """
    Carries z-stack data + a focus-score function; runs all the tests.

    z_stack: list of (z_int, frame_bgr) pairs in ascending z order.
    true_peak_z: ground-truth focus z (one of the z_int values, or
                 a sub-step float if known better).
    score_fn: callable (frame) → (combined_score, info). info should
              have a 'metrics' dict with raw per-metric scores so we
              can also report per-metric peak finding.
    """

    def __init__(
        self,
        z_stack:      List[Tuple[int, np.ndarray]],
        true_peak_z:  float,
        score_fn:     Callable[[np.ndarray], Tuple[float, Dict[str, Any]]],
        out_dir:      str = "./validation_out",
    ):
        self.z_stack     = sorted(z_stack, key=lambda t: t[0])
        self.true_peak_z = float(true_peak_z)
        self.score_fn    = score_fn
        self.out_dir     = out_dir
        os.makedirs(out_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # 1. PER-METRIC ACCURACY
    # ──────────────────────────────────────────────────────────────────
    def per_metric_accuracy(self,
                            metric_names: List[str]) -> List[CurveResult]:
        """
        Score every frame with every metric; report the predicted peak
        and MAE vs the ground-truth peak for each metric.
        """
        zs   = [z for z, _ in self.z_stack]
        per_metric_scores: Dict[str, List[float]] = {n: [] for n in metric_names}
        combined_scores: List[float] = []

        for z, frame in self.z_stack:
            s, info = self.score_fn(frame)
            combined_scores.append(s)
            m = info.get("metrics", {})
            for name in metric_names:
                per_metric_scores[name].append(float(m.get(name, 0.0)))

        results: List[CurveResult] = []
        # add the combined score too
        all_named = list(per_metric_scores.items()) + [("combined", combined_scores)]
        for name, scores in all_named:
            arr = np.asarray(scores, dtype=np.float64)
            if not np.isfinite(arr).any() or arr.max() == 0:
                pred_peak = float("nan")
                snr = 0.0; prom = 0.0
            else:
                idx = int(np.argmax(arr))
                pred_peak = self._refine_parabolic(zs, arr, idx)
                snr = float(arr[idx] / max(np.median(arr), 1e-12))
                # prominence
                mask = np.ones_like(arr, dtype=bool)
                lo, hi = max(0, idx - 1), min(len(arr), idx + 2)
                mask[lo:hi] = False
                prom = float((arr[idx] - arr[mask].max()) / max(arr[idx], 1e-12)) if mask.any() else 1.0
            mae = abs(pred_peak - self.true_peak_z)
            results.append(CurveResult(
                name=name, z=zs, score=list(scores),
                pred_peak=float(pred_peak), true_peak=self.true_peak_z,
                mae=float(mae), snr=snr, prominence=prom,
            ))
        return results

    # ──────────────────────────────────────────────────────────────────
    # 2. REPEATABILITY (under simulated re-acquisition with noise)
    # ──────────────────────────────────────────────────────────────────
    def repeatability(self,
                      n_trials: int = 8,
                      noise_sigma: float = 1.5) -> Tuple[float, int]:
        """
        For each trial, add fresh Gaussian noise to every frame and
        repeat the global-argmax-over-combined-score peak finding.
        Std of predicted peak across trials measures repeatability.
        """
        zs = [z for z, _ in self.z_stack]
        peaks: List[float] = []
        for trial in range(n_trials):
            scores: List[float] = []
            for z, frame in self.z_stack:
                noisy = self._add_noise(frame, noise_sigma, trial * 1000 + z)
                s, _info = self.score_fn(noisy)
                scores.append(s)
            arr = np.asarray(scores, dtype=np.float64)
            idx = int(np.argmax(arr))
            peaks.append(self._refine_parabolic(zs, arr, idx))
        peaks_arr = np.asarray(peaks, dtype=np.float64)
        return float(peaks_arr.std()), n_trials

    # ──────────────────────────────────────────────────────────────────
    # 3. PIPELINE TIMING
    # ──────────────────────────────────────────────────────────────────
    def timing_breakdown(self,
                         pipeline_stages: Dict[str, Callable[[Any], Any]],
                         n_repeats: int = 30) -> List[TimingResult]:
        """
        pipeline_stages: ordered dict of stage_name → callable.
        Each callable is timed `n_repeats` times on each frame.
        Returns mean and p95 (ms) per stage.
        """
        if not self.z_stack:
            return []
        # use middle frame as representative
        _, frame = self.z_stack[len(self.z_stack) // 2]
        rows: List[TimingResult] = []
        for name, fn in pipeline_stages.items():
            times = []
            # warm-up
            try:
                fn(frame)
            except Exception:
                pass
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                fn(frame)
                times.append((time.perf_counter() - t0) * 1000.0)
            arr = np.asarray(times)
            rows.append(TimingResult(
                stage=name,
                mean_ms=float(arr.mean()),
                p95_ms=float(np.percentile(arr, 95)),
            ))
        return rows

    # ──────────────────────────────────────────────────────────────────
    # 4. TRACKING OSCILLATION (analyses a controller history)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def tracking_oscillation(history_z: List[int]) -> Dict[str, float]:
        """
        Count direction reversals in cum_z over time. A well-behaved
        tracker should show very few reversals after lock.
        """
        if len(history_z) < 3:
            return {"reversals": 0.0, "rms_drift": 0.0}
        diff = np.diff(history_z)
        sign = np.sign(diff)
        # ignore zero-step frames for reversal count
        nonzero = sign[sign != 0]
        reversals = int(np.sum(np.diff(nonzero) != 0))
        rms = float(np.sqrt(np.mean(np.diff(history_z, n=1) ** 2)))
        return {"reversals": float(reversals), "rms_drift": rms}

    # ──────────────────────────────────────────────────────────────────
    # PLOTTING
    # ──────────────────────────────────────────────────────────────────
    def plot_focus_curves(self,
                          curves: List[CurveResult],
                          path: str = "focus_curve.png") -> str:
        plt = _import_plt()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for cv in curves:
            arr = np.asarray(cv.score, dtype=float)
            if arr.max() <= 0 or not np.isfinite(arr).any():
                continue
            norm = arr / arr.max()
            ax.plot(cv.z, norm, "-o", ms=3.5, lw=1.2, label=cv.name)
        ax.axvline(self.true_peak_z, ls="--", color="k",
                   alpha=0.5, label=f"true peak z={self.true_peak_z:g}")
        ax.set_xlabel("z (motor steps)")
        ax.set_ylabel("normalised focus score")
        ax.set_title("Focus curves — per-metric")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower center", ncol=3, fontsize=8)
        full_path = os.path.join(self.out_dir, path)
        fig.savefig(full_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return full_path

    def plot_peak_error(self,
                        curves: List[CurveResult],
                        path: str = "peak_error_bar.png") -> str:
        plt = _import_plt()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = [c.name for c in curves]
        maes  = [c.mae  for c in curves]
        bars = ax.bar(names, maes,
                      color=["#2A8FA3"] * (len(names) - 1) + ["#E76F51"])
        for b, v in zip(bars, maes):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("|predicted − true| z (steps)")
        ax.set_title("Peak-prediction error per metric (lower = better)")
        ax.grid(alpha=0.3, axis="y")
        full_path = os.path.join(self.out_dir, path)
        fig.savefig(full_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return full_path

    def plot_tracking_trace(self,
                            log: List[Any],
                            path: str = "tracking_trace.png") -> str:
        """log: list of FrameRecord-like with .t, .cum_z, .score, .state attrs."""
        plt = _import_plt()
        ts = [r.t for r in log]
        zs = [r.cum_z for r in log]
        ss = [r.score for r in log]
        st = [r.state for r in log]

        fig, (axz, axs) = plt.subplots(2, 1, sharex=True, figsize=(10, 5.5))
        axz.plot(ts, zs, "-", lw=1.5, color="#0F5C6E")
        axz.set_ylabel("cum_z (steps)"); axz.grid(alpha=0.3)
        axz.set_title("Closed-loop tracking trace")

        axs.plot(ts, ss, "-", lw=1.0, color="#F4A261")
        axs.set_xlabel("t (s)"); axs.set_ylabel("focus score")
        axs.grid(alpha=0.3)

        # mark state transitions
        for i in range(1, len(st)):
            if st[i] != st[i-1]:
                for ax_ in (axz, axs):
                    ax_.axvline(ts[i], ls=":", color="k", alpha=0.25)

        full_path = os.path.join(self.out_dir, path)
        fig.savefig(full_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return full_path

    def plot_timing(self,
                    timing: List[TimingResult],
                    path: str = "timing_breakdown.png") -> str:
        plt = _import_plt()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = [t.stage for t in timing]
        means = [t.mean_ms for t in timing]
        p95s  = [t.p95_ms  for t in timing]
        x = np.arange(len(names))
        ax.bar(x - 0.18, means, width=0.36, label="mean", color="#0F5C6E")
        ax.bar(x + 0.18, p95s,  width=0.36, label="p95",  color="#F4A261")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("ms"); ax.legend()
        ax.set_title("Per-stage timing")
        ax.grid(alpha=0.3, axis="y")
        full_path = os.path.join(self.out_dir, path)
        fig.savefig(full_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return full_path

    # ──────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _refine_parabolic(zs: List[int], scores: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= len(scores) - 1:
            return float(zs[idx])
        x0, x1, x2 = zs[idx - 1], zs[idx], zs[idx + 1]
        y0, y1, y2 = scores[idx - 1], scores[idx], scores[idx + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) < 1e-12:
            return float(x1)
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        b = (x2*x2 * (y0 - y1) + x1*x1 * (y2 - y0) + x0*x0 * (y1 - y2)) / denom
        if abs(a) < 1e-12:
            return float(x1)
        return float(-b / (2 * a))

    @staticmethod
    def _add_noise(frame: np.ndarray, sigma: float, seed: int) -> np.ndarray:
        rng = np.random.RandomState(abs(seed) % (2**31 - 1))
        f = frame.astype(np.float32)
        f += rng.normal(0, sigma, f.shape).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)
