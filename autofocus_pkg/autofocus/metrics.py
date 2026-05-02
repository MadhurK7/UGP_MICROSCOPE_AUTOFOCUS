"""
autofocus/metrics.py
====================
Modular focus-metric framework with adaptive, illumination-aware weighting.

WHY MULTIPLE METRICS?
─────────────────────
Every focus metric has a failure mode:

    Tenengrad (Sobel-energy)
        + robust on textured fields
        - biased toward globally bright regions
        - peaks on Airy halos for translucent particles

    Laplacian variance (Pech-Pacheco)
        + classical, fast
        - amplifies sensor noise in dim frames
        - non-monotonic for low-contrast colloids

    Brenner gradient
        + sharp, narrow peak right at focus
        - very noisy off-focus → can mislead hill-climber
        - direction-biased (horizontal differences only)

    JPEG sharpness (OpenFlexure)
        + decoupled from absolute brightness — uses compressed file size
        + robust to global illumination changes
        - requires JPEG-encode round trip per frame (~1–3 ms)

    Counting-metric (slope-sign-change)
        + invariant to overall intensity scale
        + holds up under low-light / overexposed conditions
        - lower SNR than gradient metrics on well-lit frames

The COMBINED score adapts weights to the scene condition, so the
metric currently best-suited to the conditions dominates the score.

METRIC NORMALISATION
────────────────────
Each metric returns a raw value with its own scale. Before combining
we divide by an empirical scale constant to bring all metrics into
roughly [0, 1] for typical bright-field microscopy frames at 480p.
The absolute score has NO physical meaning — only relative changes
between successive frames matter for autofocus.

CONFIDENCE
──────────
We export a confidence value alongside the score. The fine controller
SUPPRESSES motion when confidence is low (dim/empty/saturated frames).
This is the single biggest stability improvement over a naive system.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import io

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Empirical normalisation scales (calibrated on Pi camera 640×480
# bright-field colloidal images). These keep the scores in a similar
# numeric range; tune per camera if needed.
SCALE = {
    "tenengrad":   2_000_000.0,
    "laplacian":     1_000.0,
    "brenner":      10_000.0,
    "jpeg":          60_000.0,
    "counting":     100_000.0,
}


# ══════════════════════════════════════════════════════════════════════
# RAW METRIC PRIMITIVES (each takes uint8 grayscale, returns float)
# ══════════════════════════════════════════════════════════════════════
def tenengrad(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Sum-of-squared Sobel gradients. Robust under most conditions."""
    f  = gray.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    if mask is not None:
        energy = energy * mask
        denom = max(1, int(mask.sum()))
        return float(energy.sum() / denom)
    return float(energy.mean())


def laplacian_variance(gray: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> float:
    """Variance of Laplacian. Cheap; sensitive to noise."""
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    if mask is not None:
        valid = lap[mask > 0]
        if valid.size < 16:
            return 0.0
        return float(valid.var())
    return float(lap.var())


def brenner(gray: np.ndarray, k: int = 2) -> float:
    """Squared horizontal-shift differences. Peaks sharply at focus."""
    f = gray.astype(np.float32)
    diff = f[:, k:] - f[:, :-k]
    return float((diff * diff).mean())


def jpeg_sharpness(gray: np.ndarray, quality: int = 90) -> float:
    """
    OpenFlexure-style metric: encode the image as JPEG and use the
    compressed size as a proxy for high-frequency content. Works
    because JPEG quantises high-frequency DCT coefficients more
    aggressively, so blurry images compress smaller.

    Cost: a single JPEG encode (~1–3 ms on Pi 4 for 640×480).
    """
    # cv2.imencode returns bytes — len() of the buffer is the file size
    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return 0.0
    return float(len(buf))


def counting_metric(gray: np.ndarray, presmooth_sigma: float = 1.2) -> float:
    """
    Slope-sign-change counting metric.

    Idea: a sharp image has many local extrema along scanlines.
    Counting how often the gradient changes sign is a contrast-
    invariant proxy for sharpness.

    Particularly robust under:
        - non-uniform illumination
        - overexposure (saturated regions have zero gradient → no
          spurious extrema)
        - low intensity (extrema still exist; their amplitude is
          irrelevant, only the count matters)

    IMPORTANT — pre-smoothing
    ─────────────────────────
    Without a small pre-smooth, this metric responds primarily to
    sensor noise: every noisy pixel contributes to sign changes, and
    noise is uncorrelated with focus. A 0.6-px Gaussian pre-blur
    suppresses pixel-level noise while leaving real edges (which span
    several pixels at any reasonable focus) intact. After this fix the
    count tracks SIGNAL extrema, not NOISE extrema.
    """
    f = gray.astype(np.float32)
    if presmooth_sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), presmooth_sigma)
    dx = f[:, 1:] - f[:, :-1]
    dy = f[1:, :] - f[:-1, :]
    sx = np.sign(dx)
    sy = np.sign(dy)
    cx = (np.abs(np.diff(sx, axis=1)) > 1).sum()
    cy = (np.abs(np.diff(sy, axis=0)) > 1).sum()
    return float(cx + cy)


# ══════════════════════════════════════════════════════════════════════
# METRIC BANK — computes all metrics on one frame
# ══════════════════════════════════════════════════════════════════════
class MetricBank:
    """
    Computes a fixed set of focus metrics on a frame and exposes the
    raw and normalised values.

    Usage:
        bank = MetricBank()
        m = bank.compute(gray, valid_mask=valid)
        # m is a dict with raw values + normalised values
    """

    METRICS = ["tenengrad", "laplacian", "brenner", "jpeg", "counting"]

    def __init__(self,
                 enabled: Optional[List[str]] = None,
                 use_jpeg: bool = True):
        """
        Args:
            enabled:  whitelist subset of METRICS, or None for all.
            use_jpeg: disable to save 1–3 ms/frame on Pi if needed.
        """
        if enabled is None:
            enabled = list(self.METRICS)
        if not use_jpeg and "jpeg" in enabled:
            enabled = [m for m in enabled if m != "jpeg"]
        self.enabled = enabled

    def compute(self,
                gray: np.ndarray,
                valid_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all enabled metrics. Returns dict containing both
        raw (e.g. 'tenengrad') and normalised (e.g. 'tenengrad_n')
        values. Normalised values are in roughly [0, 1] but may exceed
        1 on extremely sharp inputs.
        """
        out: Dict[str, float] = {}
        if "tenengrad" in self.enabled:
            out["tenengrad"] = tenengrad(gray, valid_mask)
        if "laplacian" in self.enabled:
            out["laplacian"] = laplacian_variance(gray, valid_mask)
        if "brenner" in self.enabled:
            out["brenner"]   = brenner(gray)
        if "jpeg" in self.enabled:
            out["jpeg"]      = jpeg_sharpness(gray)
        if "counting" in self.enabled:
            out["counting"]  = counting_metric(gray)
        # normalised companions
        for k in list(out.keys()):
            out[k + "_n"] = out[k] / SCALE.get(k, 1.0)
        return out


# ══════════════════════════════════════════════════════════════════════
# ADAPTIVE COMBINER
# ══════════════════════════════════════════════════════════════════════
class AdaptiveCombiner:
    """
    Combines per-metric scores into a single weighted focus score.

    Weighting is chosen from a lookup keyed by the preprocessor's
    scene-condition string. The lookup encodes which metrics are
    known to be most reliable under that condition.

    A user can override the lookup by passing `weight_override`.
    """

    # Default weight tables. Each row sums to 1.0 (verified in __init__).
    DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
        "ok": {
            "tenengrad": 0.40, "laplacian": 0.20,
            "brenner":   0.20, "jpeg":      0.10, "counting": 0.10,
        },
        "low_intensity": {
            # counting + jpeg are scale-invariant → preferred
            "tenengrad": 0.20, "laplacian": 0.10,
            "brenner":   0.10, "jpeg":      0.25, "counting": 0.35,
        },
        "low_contrast": {
            "tenengrad": 0.25, "laplacian": 0.10,
            "brenner":   0.15, "jpeg":      0.20, "counting": 0.30,
        },
        "overexposed": {
            # gradient metrics are zero on saturated patches
            "tenengrad": 0.10, "laplacian": 0.05,
            "brenner":   0.10, "jpeg":      0.30, "counting": 0.45,
        },
        "dark": {
            # all metrics noisy — spread weights so no single metric dominates
            "tenengrad": 0.20, "laplacian": 0.20,
            "brenner":   0.20, "jpeg":      0.20, "counting": 0.20,
        },
        "empty": {
            # weights don't matter — confidence will force motion off
            "tenengrad": 0.20, "laplacian": 0.20,
            "brenner":   0.20, "jpeg":      0.20, "counting": 0.20,
        },
    }

    def __init__(self, weight_override: Optional[Dict[str, Dict[str, float]]] = None):
        self._w = dict(weight_override) if weight_override else dict(self.DEFAULT_WEIGHTS)
        for cond, ws in self._w.items():
            tot = sum(ws.values())
            if tot > 0:
                self._w[cond] = {k: v / tot for k, v in ws.items()}

    def combine(self,
                metrics: Dict[str, float],
                condition: str) -> float:
        """
        Args:
            metrics:   output of MetricBank.compute (uses '<name>_n').
            condition: scene-type string from the preprocessor.

        Returns:
            Single scalar focus score (higher = sharper).
        """
        weights = self._w.get(condition, self._w["ok"])
        s = 0.0
        for k, w in weights.items():
            s += w * float(metrics.get(k + "_n", 0.0))
        return s


# ══════════════════════════════════════════════════════════════════════
# CONFIDENCE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════
class ConfidenceEstimator:
    """
    Estimate how much we should trust the current focus score.

    Inputs:
        - raw metrics dict
        - scene condition dict
        - recent history of scores (for stationarity)

    Output: confidence in [0, 1].
        1.0 → strong signal, controller can act
        0.0 → suppress motion
    """

    def __init__(self,
                 min_valid_fraction: float = 0.50,
                 ten_floor:  float = 0.005,   # tenengrad_n below this = weak
                 std_floor:  float = 4.0):
        self.min_valid = min_valid_fraction
        self.ten_floor = ten_floor
        self.std_floor = std_floor

    def estimate(self,
                 metrics: Dict[str, float],
                 condition: Dict[str, any],
                 history: Optional[List[float]] = None) -> float:
        # 1. signal strength
        ten_n = metrics.get("tenengrad_n", 0.0)
        sig = min(1.0, ten_n / max(self.ten_floor, 1e-9))

        # 2. valid pixel fraction (penalise saturated/clipped frames)
        valid_f = float(condition.get("valid_fraction", 1.0))
        valid_score = max(0.0, (valid_f - self.min_valid) / (1.0 - self.min_valid))

        # 3. global std (penalise empty/low-contrast frames)
        std_score = min(1.0, condition.get("std", 0.0) / (3 * self.std_floor))

        # 4. stationarity bonus — if recent history is stable, we're more sure
        stationarity = 1.0
        if history and len(history) >= 4:
            arr = np.asarray(history[-4:], dtype=np.float32)
            mean = arr.mean()
            if mean > 1e-9:
                cv = arr.std() / mean
                stationarity = float(np.clip(1.0 - cv, 0.0, 1.0))

        # weighted aggregate
        c = 0.40 * sig + 0.25 * valid_score + 0.20 * std_score + 0.15 * stationarity
        return float(np.clip(c, 0.0, 1.0))
