# openflexure_autofocus_fixed.py


from __future__ import annotations

import time
from dataclasses import dataclass
from collections import deque
from typing import List, Dict, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ImageCondition:
    type: str
    mean: float
    std: float
    saturated_fraction: float
    dark_fraction: float


@dataclass
class FocusMetrics:
    tenengrad: float
    laplacian: float
    brenner: float
    contrast: float
    counting_metric: float
    jpeg_sharpness: float
    confidence: float


@dataclass
class ProcessedFrame:
    gray: np.ndarray
    normalized: np.ndarray
    enhanced: np.ndarray
    roi: np.ndarray
    condition: ImageCondition
    metrics: FocusMetrics


@dataclass
class FocusSample:
    z: int
    confidence: float
    metrics: FocusMetrics


# ============================================================
# METRIC ENGINE
# ============================================================

class UnifiedFocusMetricEngine:

    def __init__(
        self,
        illumination_sigma=35.0,
        gaussian_sigma=0.8,
        clahe_clip=2.5,
        clahe_grid=8,
        roi_fraction=0.65,
    ):

        self.illumination_sigma = illumination_sigma
        self.gaussian_sigma = gaussian_sigma
        self.roi_fraction = roi_fraction

        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid)
        )

        self._metric_history = {
            k: deque(maxlen=100)
            for k in [
                "tenengrad",
                "laplacian",
                "brenner",
                "contrast",
                "counting_metric",
                "jpeg_sharpness",
            ]
        }

    # ========================================================
    # MAIN EVALUATION
    # ========================================================

    def evaluate(self, frame):

        gray, normalized, enhanced = self._preprocess(frame)

        roi = self._extract_roi(enhanced)

        condition = self._analyze_condition(roi)

        raw_metrics = self._compute_metrics_smart(roi)

        confidence = self._confidence(raw_metrics)

        metrics = FocusMetrics(
            tenengrad=raw_metrics["tenengrad"],
            laplacian=raw_metrics["laplacian"],
            brenner=raw_metrics["brenner"],
            contrast=raw_metrics["contrast"],
            counting_metric=raw_metrics["counting_metric"],
            jpeg_sharpness=raw_metrics["jpeg_sharpness"],
            confidence=float(confidence),
        )

        return ProcessedFrame(
            gray=gray,
            normalized=normalized,
            enhanced=enhanced,
            roi=roi,
            condition=condition,
            metrics=metrics,
        )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    def _preprocess(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bg = cv2.GaussianBlur(gray, (0, 0), self.illumination_sigma)
        bg = np.maximum(bg, 1)

        normalized = (
            gray.astype(np.float32) /
            bg.astype(np.float32)
        ) * 128.0

        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        denoised = cv2.GaussianBlur(
            normalized,
            (0, 0),
            self.gaussian_sigma
        )

        enhanced = self.clahe.apply(denoised)

        return gray, normalized, enhanced

    # ========================================================
    # ROI
    # ========================================================

    def _extract_roi(self, image):

        h, w = image.shape

        rw = int(w * self.roi_fraction)
        rh = int(h * self.roi_fraction)

        x1 = (w - rw) // 2
        y1 = (h - rh) // 2

        return image[y1:y1+rh, x1:x1+rw]

    # ========================================================
    # CONDITION ANALYSIS
    # ========================================================

    def _analyze_condition(self, gray):

        mean = float(np.mean(gray))
        std = float(np.std(gray))

        sat = float(np.mean(gray > 250))
        dark = float(np.mean(gray < 20))

        lap = cv2.Laplacian(gray, cv2.CV_64F).var()

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        ten = np.mean(np.sqrt(gx**2 + gy**2))

        ctype = "normal"

        if mean < 45:
            ctype = "dark"

        elif sat > 0.05:
            ctype = "overexposed"

        elif mean < 90:
            ctype = "low_intensity"

        if std < 12:
            ctype += "_low_contrast"

        if lap < 18 and ten < 12:
            ctype += "_blurred"

        return ImageCondition(
            type=ctype,
            mean=mean,
            std=std,
            saturated_fraction=sat,
            dark_fraction=dark,
        )

    # ========================================================
    # PARTICLE DETECTION
    # ========================================================

    def _particle_mask(self, gray):

        blur1 = cv2.GaussianBlur(gray, (0, 0), 1)
        blur2 = cv2.GaussianBlur(gray, (0, 0), 4)

        dog = cv2.subtract(blur1, blur2)

        dog = cv2.normalize(
            dog,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        _, binary = cv2.threshold(
            dog,
            140,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((3, 3), np.uint8)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )

        return binary

    # ========================================================
    # SMART METRICS
    # ========================================================

    def _compute_metrics_smart(self, gray):

        mask = self._particle_mask(gray)

        roi = cv2.bitwise_and(gray, gray, mask=mask)

        if np.count_nonzero(mask) < 50:
            roi = gray

        return self._raw_metrics(roi)

    # ========================================================
    # RAW METRICS
    # ========================================================

    def _raw_metrics(self, gray):

        metrics = {}

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        metrics["tenengrad"] = float(
            np.mean(np.sqrt(gx**2 + gy**2))
        )

        metrics["laplacian"] = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

        shifted = np.roll(gray, -2, axis=1)

        metrics["brenner"] = float(
            np.mean(
                (
                    shifted.astype(np.float32) -
                    gray.astype(np.float32)
                ) ** 2
            )
        )

        metrics["contrast"] = float(gray.std())

        metrics["counting_metric"] = self._counting_metric(gray)

        _, enc = cv2.imencode(
            '.jpg',
            gray,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        metrics["jpeg_sharpness"] = float(len(enc))

        return metrics

    # ========================================================
    # COUNTING METRIC
    # ========================================================

    def _counting_metric(self, gray):

        g = gray.astype(np.float32)

        dx = g[:, 1:] - g[:, :-1]
        dy = g[1:, :] - g[:-1, :]

        sx = np.sign(dx)
        sy = np.sign(dy)

        cx = np.abs(sx[:, 1:] - sx[:, :-1]) > 1
        cy = np.abs(sy[1:, :] - sy[:-1, :]) > 1

        return float(np.sum(cx) + np.sum(cy))

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def _confidence(self, metrics):

        values = np.array([
            metrics["tenengrad"],
            metrics["laplacian"],
            metrics["counting_metric"],
        ], dtype=np.float32)

        values = values / (np.max(values) + 1e-6)

        agreement = 1.0 / (np.std(values) + 1e-6)

        strength = np.mean(values)

        confidence = (
            0.5 * np.tanh(agreement / 2) +
            0.5 * np.tanh(strength * 2)
        )

        return float(np.clip(confidence, 0, 1))


# ============================================================
# AUTOFOCUS CONTROLLER
# ============================================================

class UnifiedFocusController:

    def __init__(
        self,
        coarse_step=30,
        fine_step=5,
        deadband=0.03,
        damping=0.7,
    ):

        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.deadband = deadband
        self.damping = damping

        self.last_score = None
        self.direction = 1

    def update(self, score):

        if self.last_score is None:
            self.last_score = score
            return self.coarse_step

        delta = score - self.last_score

        self.last_score = score

        if abs(delta) < self.deadband:
            return 0

        if delta < 0:
            self.direction *= -1

        step = (
            self.coarse_step
            if abs(delta) > 0.10
            else self.fine_step
        )

        step = int(step * self.damping)

        return self.direction * step


# ============================================================
# OPENFLEXURE AUTOFOCUS
# ============================================================

class OpenFlexureAutofocus:

    def __init__(
        self,
        stage,
        camera,
        debug=False,
    ):

        self.stage = stage
        self.camera = camera
        self.debug = debug

        self.metric_engine = UnifiedFocusMetricEngine()

        self.controller = UnifiedFocusController()

        self.sweep_range = 3000
        self.sweep_step = 150

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_curve(self, x):

        x = np.array(x, dtype=np.float32)

        return (
            x - x.min()
        ) / (
            x.max() - x.min() + 1e-6
        )

    # ========================================================
    # BUILD FOCUS CURVE
    # ========================================================

    def _build_focus_curve(self, samples):

        ten = self._normalize_curve([
            s.metrics.tenengrad for s in samples
        ])

        count = self._normalize_curve([
            s.metrics.counting_metric for s in samples
        ])

        lap = self._normalize_curve([
            s.metrics.laplacian for s in samples
        ])

        combined = (
            0.55 * ten +
            0.35 * count +
            0.10 * lap
        )

        smooth = gaussian_filter1d(combined, sigma=1.2)

        return smooth

    # ========================================================
    # COARSE SWEEP
    # ========================================================

    def _coarse_sweep(self):

        current_z = self.stage.position["z"]

        start_z = current_z - self.sweep_range // 2

        samples = []

        for z in np.arange(
            start_z,
            start_z + self.sweep_range,
            self.sweep_step
        ):

            self._move_absolute(int(z))

            pf = self.metric_engine.evaluate(
                self.camera.grab_frame()
            )

            samples.append(
                FocusSample(
                    z=int(z),
                    confidence=pf.metrics.confidence,
                    metrics=pf.metrics,
                )
            )

            if self.debug:
                print(
                    f"Z={z}  "
                    f"TEN={pf.metrics.tenengrad:.2f}  "
                    f"COUNT={pf.metrics.counting_metric:.2f}"
                )

        return samples

    # ========================================================
    # AUTOFOCUS
    # ========================================================

    def autofocus(self):

        samples = self._coarse_sweep()

        focus_curve = self._build_focus_curve(samples)

        valid_region = focus_curve[3:-3]

        peak_idx = np.argmax(valid_region) + 3

        peak_z = samples[peak_idx].z

        if self.debug:
            print(f"Peak focus Z = {peak_z}")

        self._move_absolute(peak_z)

        return peak_z

    # ========================================================
    # STAGE MOVEMENT
    # ========================================================

    def _move_absolute(self, z):

        dz = z - self.stage.position["z"]

        self.stage.move_relative(z=dz)

        time.sleep(0.15)


# ============================================================
# MOCK TEST
# ============================================================

if __name__ == "__main__":

    class MockStage:

        def __init__(self):
            self._z = 0

        @property
        def position(self):
            return {"z": self._z}

        def move_relative(self, z=0):
            self._z += z

    class MockCamera:

        def __init__(self, stage):

            self.stage = stage

            base = np.zeros((480, 640), dtype=np.uint8)

            rng = np.random.RandomState(0)

            for _ in range(300):

                x = rng.randint(20, 620)
                y = rng.randint(20, 460)

                cv2.circle(base, (x, y), 2, 255, -1)

            self.base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        def grab_frame(self):

            z = self.stage.position["z"]

            sigma = abs(z) / 120.0

            if sigma < 0.3:
                sigma = 0.3

            blurred = cv2.GaussianBlur(
                self.base,
                (0, 0),
                sigma
            )

            noise = np.random.normal(
                0,
                4,
                blurred.shape
            ).astype(np.float32)

            img = blurred.astype(np.float32) + noise

            return np.clip(img, 0, 255).astype(np.uint8)
    stage = MockStage()

    camera = MockCamera(stage)

    af = OpenFlexureAutofocus(
        stage=stage,
        camera=camera,
        debug=True,
    )

    samples = af._coarse_sweep()

    curve = af._build_focus_curve(samples)

    z_vals = [s.z for s in samples]

    ten = [s.metrics.tenengrad for s in samples]
    lap = [s.metrics.laplacian for s in samples]
    bren = [s.metrics.brenner for s in samples]
    count = [s.metrics.counting_metric for s in samples]

    plt.figure(figsize=(10,6))

    plt.plot(z_vals, ten, label="Tenengrad")
    plt.plot(z_vals, lap, label="Laplacian")
    plt.plot(z_vals, bren, label="Brenner")
    plt.plot(z_vals, count, label="Counting")
    plt.plot(z_vals, curve, linewidth=4, label="Combined")

    plt.axvline(
        z_vals[np.argmax(curve)],
        linestyle="--",
        label="Detected Focus"
    )

    plt.xlabel("Z position")
    plt.ylabel("Focus score")

    plt.title("Autofocus Validation")

    plt.legend()

    plt.grid(True)

    plt.savefig("validation_plot.png")

    plt.show()