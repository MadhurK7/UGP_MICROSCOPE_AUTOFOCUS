from __future__ import annotations

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

from openflexure_autofocus import (
    UnifiedFocusMetricEngine
)

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = r"C:\Users\Raman-PC\Downloads\Autofocus_UGP\archive\train\blood-18-5"

TRUE_FOCUS_Z = 0

# ============================================================
# LOAD DATASET
# ============================================================


def load_dataset(folder):

    images = []

    supported = (".png", ".jpg", ".jpeg", ".tif")

    files = sorted(os.listdir(folder))

    if len(files) == 0:
        print("ERROR: Folder empty")
        return images, None

    true_focus_z = None

    z_counter = 0

    for fname in files:

        if not fname.lower().endswith(supported):
            continue

        path = os.path.join(folder, fname)

        print("Loading:", path)

        img = cv2.imread(path)

        if img is None:
            print("FAILED:", fname)
            continue

        z = z_counter

        # --------------------------------------------
        # Detect ground truth from filename
        # --------------------------------------------

        if "best" in fname.lower():

            true_focus_z = z

            print(f">>> TRUE FOCUS FOUND: {fname} at z={z}")

        images.append((z, img))

        z_counter += 1

    print(f"\nLoaded {len(images)} images")

    return images, true_focus_z


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):

    x = np.array(x, dtype=np.float32)

    return (
        x - x.min()
    ) / (
        x.max() - x.min() + 1e-6
    )

# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(images, TRUE_FOCUS_Z):

    engine = UnifiedFocusMetricEngine()

    z_vals = []

    ten_scores = []
    lap_scores = []
    bren_scores = []
    count_scores = []
    conf_scores = []

    # --------------------------------------------------------

    for z, img in images:

        pf = engine.evaluate(img)

        m = pf.metrics

        z_vals.append(z)

        ten_scores.append(m.tenengrad)
        lap_scores.append(m.laplacian)
        bren_scores.append(m.brenner)
        count_scores.append(m.counting_metric)
        conf_scores.append(m.confidence)

        print(
            f"Z={z:5d} | "
            f"TEN={m.tenengrad:8.2f} | "
            f"LAP={m.laplacian:8.2f} | "
            f"BRE={m.brenner:8.2f} | "
            f"COUNT={m.counting_metric:8.2f}"
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    ten_n = normalize(ten_scores)
    lap_n = normalize(lap_scores)
    bren_n = normalize(bren_scores)
    count_n = normalize(count_scores)

    # ========================================================
    # COMBINED CURVE
    # ========================================================

    combined = (
        0.45 * ten_n +
        0.35 * bren_n +
        0.15 * lap_n +
        0.05 * count_n
    )

    combined = gaussian_filter1d(combined, sigma=1)

    # ========================================================
    # DETECT PEAK
    # ========================================================

    peak_idx = np.argmax(combined)

    detected_focus_z = z_vals[peak_idx]

    error = abs(detected_focus_z - TRUE_FOCUS_Z)

    print("\n==============================")
    print(f"Detected focus z : {detected_focus_z}")
    print(f"Ground truth z   : {TRUE_FOCUS_Z}")
    print(f"Absolute error   : {error}")
    print("==============================")

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(12, 7))

    plt.plot(z_vals, ten_n, "-o", label="Tenengrad")
    plt.plot(z_vals, lap_n, "-o", label="Laplacian")
    plt.plot(z_vals, bren_n, "-o", label="Brenner")
    plt.plot(z_vals, count_n, "-o", label="Counting")

    plt.plot(
        z_vals,
        combined,
        linewidth=4,
        color="black",
        label="Combined"
    )

    plt.axvline(
        TRUE_FOCUS_Z,
        linestyle="--",
        color="green",
        label="True Focus"
    )

    plt.axvline(
        detected_focus_z,
        linestyle="--",
        color="red",
        label="Detected Focus"
    )

    plt.xlabel("Z position")

    plt.ylabel("Normalized focus score")

    plt.title("Autofocus Validation on Real Dataset")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "dataset_validation.png",
        dpi=300
    )

    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    
    images, true_focus_z = load_dataset(DATASET_DIR)

    print(f"\nLoaded {len(images)} images")

    validate_dataset(images, true_focus_z)
