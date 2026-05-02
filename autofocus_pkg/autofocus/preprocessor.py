"""
autofocus/preprocessor.py
=========================
Illumination-aware preprocessor for colloidal-microscopy autofocus.

PIPELINE (per frame):
    BGR  →  grayscale
         →  illumination flat-field normalisation
         →  saturation/clipping mask
         →  optional CLAHE (only if global contrast is poor)
         →  mild Gaussian denoise (preserves edges for AF metrics)
         →  ROI selection (centre crop AND/OR particle clusters)

WHY EACH STAGE EXISTS
─────────────────────
1. ILLUMINATION NORMALISATION (large-σ Gaussian flat-field)
   Bright-field colloidal microscopy produces a strong vignette and
   condenser-misalignment gradient. A focus metric like Tenengrad will
   respond to these gradients themselves, so the score becomes biased
   by where in the frame contrast happens to be highest. Dividing by
   a smoothed copy of the frame removes this DC trend without touching
   particle-scale features.

2. SATURATION / CLIPPING MASK
   Saturated pixels carry NO focus information — their derivatives are
   zero. Worse, a saturated region in one z-plane often becomes
   un-saturated in another, biasing the focus metric. We compute a
   binary mask and downstream metrics ignore these pixels.

3. CONDITIONAL CLAHE
   Always applying CLAHE harms metrics on already-good frames (it
   amplifies sensor noise into spurious "edges"). We measure the
   pre-CLAHE std-dev: only when it falls below a threshold does
   CLAHE engage. This handles dim/translucent colloids without
   poisoning bright-well-lit frames.

4. MILD DENOISE
   Heavy denoising (NLM, bilateral) destroys the edge information
   that focus metrics depend on. A small-sigma Gaussian (~0.8 px)
   removes pure-pixel sensor noise without touching the diffraction-
   limited features we care about.

5. ROI EXTRACTION
   Two complementary modes:
     a) CENTRE ROI — for whole-field focus on the optical axis where
        aberrations are minimal.
     b) PARTICLE ROI — extract small windows around bright/dark blobs
        and average the metric over them. Critical for sparse fields:
        a single in-focus particle in an empty field gives a higher
        signal-to-bias ratio than a whole-frame metric does.

PHASE HALOS / TRANSLUCENT COLLOIDS
──────────────────────────────────
Translucent colloids show a bright Airy ring at ±focus and a darker
core at exact focus (Becke line). Edge metrics like Sobel/Laplacian
respond to the BRIGHT RING and so peak slightly off-focus. The
counting metric in metrics.py and the symmetric particle-ROI logic
here mitigate this — see metrics.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PreprocResult:
    """Everything downstream needs from one frame."""
    gray:           np.ndarray            # raw grayscale (uint8)
    normalised:     np.ndarray            # flat-field-corrected (uint8)
    enhanced:       np.ndarray            # post-CLAHE & denoise (uint8)
    valid_mask:     np.ndarray            # uint8, 1 = trustworthy pixel
    centre_roi:     np.ndarray            # cropped enhanced image
    particle_rois:  List[np.ndarray]      # small windows around blobs
    condition:      Dict[str, Any]        # scene diagnostic
    illum_field:    np.ndarray            # smoothed background (debug)


# ──────────────────────────────────────────────────────────────────────
class Preprocessor:
    """
    Configurable, real-time preprocessor.

    Tunables documented inline. Defaults are calibrated for a Pi camera
    at 640×480, bright-field, 1–10 µm latex/silica spheres on glass.
    """

    def __init__(
        self,
        # ── illumination normalisation ───────────────────────────────
        illum_sigma:        float = 35.0,   # large-σ for flat-field estimate
        # ── saturation / clipping ────────────────────────────────────
        sat_high:           int   = 250,    # treat ≥ this as saturated
        sat_low:            int   = 5,      # treat ≤ this as clipped
        # ── conditional CLAHE ────────────────────────────────────────
        clahe_clip:         float = 2.5,
        clahe_tiles:        int   = 8,
        clahe_engage_std:   float = 25.0,   # CLAHE only if std < this
        # ── denoise ──────────────────────────────────────────────────
        denoise_sigma:      float = 0.8,    # mild — preserve edges
        # ── ROI: centre crop ─────────────────────────────────────────
        centre_roi_frac:    float = 0.65,
        # ── ROI: particles (DoG blob detector) ───────────────────────
        particle_dog_small: float = 1.5,    # σ for inner blob detector
        particle_dog_large: float = 4.0,    # σ for outer blob detector
        particle_min_area:  int   = 6,
        particle_max_area:  int   = 600,
        particle_pad:       int   = 12,     # patch padding around blobs
        max_particles:      int   = 12,     # limit ROIs per frame
        # ── condition classifier ─────────────────────────────────────
        dark_thresh:        int   = 40,
        low_intensity:      int   = 90,
        max_sat_frac:       float = 0.05,
    ):
        self.illum_sigma     = illum_sigma
        self.sat_high        = sat_high
        self.sat_low         = sat_low
        self.clahe_engage_std = clahe_engage_std
        self.denoise_sigma   = denoise_sigma
        self.centre_roi_frac = centre_roi_frac
        self.dog_small       = particle_dog_small
        self.dog_large       = particle_dog_large
        self.p_min_area      = particle_min_area
        self.p_max_area      = particle_max_area
        self.p_pad           = particle_pad
        self.max_particles   = max_particles
        self.dark_thresh     = dark_thresh
        self.low_intensity   = low_intensity
        self.max_sat_frac    = max_sat_frac

        # CLAHE object — pre-built once
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles)
        )

    # ──────────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> PreprocResult:
        """
        Run the full pipeline on one BGR/grayscale frame.

        Args:
            frame: uint8 BGR (H,W,3) or grayscale (H,W).

        Returns:
            PreprocResult with all intermediate stages exposed for
            downstream metrics, debug, and validation.
        """
        # ── 1. grayscale ────────────────────────────────────────────
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # ── 2. illumination flat-field normalisation ────────────────
        # Estimate the large-scale brightness field with a strong blur,
        # then divide. We multiply by 128 to keep the result in mid-gray
        # so downstream uint8 conversions don't clip.
        illum = cv2.GaussianBlur(gray, (0, 0), self.illum_sigma)
        illum_safe = np.maximum(illum.astype(np.float32), 1.0)
        norm = (gray.astype(np.float32) / illum_safe) * 128.0
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        # ── 3. saturation / clipping mask ───────────────────────────
        # 1 = trustworthy, 0 = saturated/clipped on the RAW image.
        # Use raw (not normalised) because saturation is a sensor
        # property — normalisation can hide it.
        valid = ((gray > self.sat_low) & (gray < self.sat_high)).astype(np.uint8)

        # ── 4. conditional CLAHE ────────────────────────────────────
        global_std = float(norm.std())
        if global_std < self.clahe_engage_std:
            enhanced = self._clahe.apply(norm)
        else:
            enhanced = norm

        # ── 5. mild denoise ──────────────────────────────────────────
        enhanced = cv2.GaussianBlur(enhanced, (0, 0), self.denoise_sigma)

        # ── 6a. centre ROI ───────────────────────────────────────────
        centre_roi = self._crop_centre(enhanced, self.centre_roi_frac)

        # ── 6b. particle ROIs ────────────────────────────────────────
        particle_rois = self._extract_particle_patches(enhanced)

        # ── 7. scene condition classifier ────────────────────────────
        condition = self._classify(gray, valid)

        return PreprocResult(
            gray=gray,
            normalised=norm,
            enhanced=enhanced,
            valid_mask=valid,
            centre_roi=centre_roi,
            particle_rois=particle_rois,
            condition=condition,
            illum_field=illum,
        )

    # ──────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _crop_centre(img: np.ndarray, frac: float) -> np.ndarray:
        h, w = img.shape[:2]
        rh, rw = int(h * frac), int(w * frac)
        y0 = (h - rh) // 2
        x0 = (w - rw) // 2
        return img[y0:y0 + rh, x0:x0 + rw]

    def _extract_particle_patches(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Detect bright/dark blobs via Difference-of-Gaussians and
        return small image patches around each one.

        Why DoG: a band-pass filter that responds to features whose
        scale is between the two σ values. This naturally selects
        particle-sized blobs and rejects vignettes (too large) and
        sensor noise (too small).

        Why bright AND dark blobs: at ±focus colloids appear bright
        (Airy ring) or dark (Becke line). Tracking only one polarity
        misses half the focus surface.
        """
        f = gray.astype(np.float32)
        g_small = cv2.GaussianBlur(f, (0, 0), self.dog_small)
        g_large = cv2.GaussianBlur(f, (0, 0), self.dog_large)
        dog = g_small - g_large

        # Find both bright and dark blobs
        bright = (dog >  3.0).astype(np.uint8) * 255
        dark   = (dog < -3.0).astype(np.uint8) * 255
        binary = cv2.bitwise_or(bright, dark)

        # connectedComponents is faster than findContours for this
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        patches: List[np.ndarray] = []
        # Sort components by area so larger (likely-real) blobs come first
        order = sorted(
            range(1, n),
            key=lambda i: stats[i, cv2.CC_STAT_AREA],
            reverse=True,
        )
        for i in order:
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.p_min_area or area > self.p_max_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            x0 = max(0, x - self.p_pad)
            y0 = max(0, y - self.p_pad)
            x1 = min(gray.shape[1], x + w + self.p_pad)
            y1 = min(gray.shape[0], y + h + self.p_pad)
            patch = gray[y0:y1, x0:x1]
            if patch.size > 0:
                patches.append(patch)
            if len(patches) >= self.max_particles:
                break

        return patches

    def _classify(self, gray: np.ndarray, valid: np.ndarray) -> Dict[str, Any]:
        """
        Diagnose the scene to inform metric weighting downstream.

        Possible types:
            ok               — trust everything
            dark             — mean intensity below threshold; metrics noisy
            low_intensity    — usable but counting-metric preferred
            overexposed      — too many saturated pixels; trust counting only
            low_contrast     — std too small; widen weighting bias
            empty            — no usable signal at all; controller should hold
        """
        mean   = float(gray.mean())
        std    = float(gray.std())
        sat_f  = float((gray >= self.sat_high).mean())
        dark_f = float((gray <= self.sat_low).mean())
        valid_f = float(valid.mean())

        if valid_f < 0.10:
            scene = "empty"
        elif sat_f > self.max_sat_frac:
            scene = "overexposed"
        elif mean < self.dark_thresh:
            scene = "dark"
        elif mean < self.low_intensity:
            scene = "low_intensity"
        elif std < 8.0:
            scene = "low_contrast"
        else:
            scene = "ok"

        return {
            "type": scene,
            "mean": mean,
            "std":  std,
            "saturated_fraction": sat_f,
            "dark_fraction":      dark_f,
            "valid_fraction":     valid_f,
        }
