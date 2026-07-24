# --------------------------------------------------------------------
# Computes per-object color metrics (mean and coefficient of variation)
# across HSV and LAB color channels for each labeled fruit mask.
#
# Inputs:
#   - raw_img         : BGR uint8 image (required)
#   - labeled_objects : labeled mask (0=background, positive int = object)
#   - corrected_img   : color-corrected BGR uint8 image (optional)
#
# For each object and each input image provided, computes per-channel:
#   - Mean
#   - Coefficient of variation (CV)
#
# CV conventions:
#   - Hue (H):      circular mean and circular CV (scipy.stats.circmean/circstd)
#                   using degrees [0, 360). Circular CV = (circ_std / 180) * 100
#                   normalized to the [0, 180] half-circle range of OpenCV hue.
#   - All other channels (S, V, L, a, b):
#                   range-normalized CV = (std / empirical_range) * 100
#                   where empirical_range is computed from the full image
#                   (not per-mask) to provide a stable, consistent denominator
#                   across all objects in the image.
#
# Output key naming convention:
#   obj_{channel}_{stat}_{source}
#   e.g. obj_hue_mean_raw, obj_hue_cv_corrected, obj_L_mean_raw
#
# Output format mirrors calculate_size_shape: dict keyed by label (int)
# with a 'traits' sub-dict and a 'qc' sub-dict.
# --------------------------------------------------------------------

import cv2
import numpy as np
from scipy.stats import circmean, circstd


def calculate_color_metrics(raw_img, labeled_objects, corrected_img=None):
    """
    Compute per-object color metrics across HSV and LAB channels for each
    labeled fruit mask.

    Args:
        raw_img         : BGR uint8 image (required). Used as the 'raw' source.
        labeled_objects : labeled mask (uint8 or int), same spatial dimensions
                          as raw_img. 0 = background; positive integers label
                          individual objects.
        corrected_img   : color-corrected BGR uint8 image (optional). Must be
                          the same dimensions as raw_img. When provided, metrics
                          are computed for both sources and stored with '_raw'
                          and '_corrected' suffixes. When None, only '_raw'
                          metrics are computed.

    Returns:
        metrics_by_label : dict keyed by label id (int).
            Each value is a dict with:
                'traits' : dict of metric key -> float (or None on failure)
                'qc'     : dict with 'mask_pixel_count' (int) and
                           'color_metrics_ok' (bool)

    Notes:
        - Hue uses circular statistics (degrees). OpenCV hue is [0, 180) for
          uint8; values are scaled to [0, 360) before circular computation,
          then circular CV is normalized back to the [0, 180] range.
        - Range-normalized CV uses the empirical range computed from the full
          image (all pixels, including background) to ensure a stable
          denominator consistent across all objects in the image.
        - LAB conversion uses cv2.COLOR_BGR2LAB. For uint8 input: L in [0,255],
          a and b in [0, 255] (128 = neutral).
    """

    sources = {"raw": raw_img}
    if corrected_img is not None:
        sources["corrected"] = corrected_img

    # Pre-compute color space conversions and empirical ranges per source.
    # Arrays are flattened to (N_pixels, 3) so per-label extraction uses
    # integer fancy indexing rather than repeated 2D boolean mask scans.
    converted = {}
    emp_ranges = {}

    for src_name, src_img in sources.items():
        hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        converted[src_name] = {
            "hsv": hsv.reshape(-1, 3),
            "lab": lab.reshape(-1, 3),
        }

        # Empirical ranges from all image pixels (background included).
        # Using the full image rather than masked pixels ensures the denominator
        # is independent of object identity and stable across label iterations.
        emp_ranges[src_name] = {
            # H is circular — range not used for CV; placeholder for consistency
            "H": None,
            "S": _empirical_range(hsv[:, :, 1]),
            "V": _empirical_range(hsv[:, :, 2]),
            "L": _empirical_range(lab[:, :, 0]),
            "a": _empirical_range(lab[:, :, 1]),
            "b": _empirical_range(lab[:, :, 2]),
        }

    # Build label → flat pixel index mapping in one sort pass instead of
    # scanning the full array once per label with a boolean comparison.
    flat_labels = labeled_objects.ravel()
    sort_order  = np.argsort(flat_labels, kind="stable")
    sorted_flat = flat_labels[sort_order]
    unique_vals, start_pos = np.unique(sorted_flat, return_index=True)
    end_pos = np.append(start_pos[1:], len(flat_labels))
    label_indices = {
        int(val): sort_order[s:e]
        for val, s, e in zip(unique_vals, start_pos, end_pos)
        if val != 0
    }

    labels = sorted(label_indices.keys())
    metrics_by_label = {}

    for label in labels:
        idx         = label_indices[label]
        pixel_count = len(idx)

        if pixel_count == 0:
            metrics_by_label[label] = {
                "traits": {},
                "qc": {"mask_pixel_count": 0, "color_metrics_ok": False},
            }
            continue

        traits = {}

        for src_name in sources:
            hsv_flat = converted[src_name]["hsv"]  # (N_pixels, 3)
            lab_flat = converted[src_name]["lab"]  # (N_pixels, 3)
            er       = emp_ranges[src_name]

            # ------ HSV channels ------

            # Hue: circular statistics
            # OpenCV uint8 hue is [0, 180); scale to [0, 360) for circular math
            hue_pixels = hsv_flat[idx, 0] * 2.0   # [0, 360)
            h_circ_mean = float(circmean(hue_pixels, high=360.0, low=0.0))
            h_circ_std  = float(circstd(hue_pixels,  high=360.0, low=0.0))
            # Normalize circular CV to [0, 180] OpenCV range so it is
            # comparable to the range-normalized CVs of other channels
            h_cv = (h_circ_std / 180.0) * 100.0

            traits[f"obj_hue_mean_{src_name}"] = h_circ_mean / 2.0   # back to [0,180)
            traits[f"obj_hue_cv_{src_name}"]   = h_cv

            # Saturation
            s_pixels = hsv_flat[idx, 1]
            traits[f"obj_sat_mean_{src_name}"] = float(np.mean(s_pixels))
            traits[f"obj_sat_cv_{src_name}"]   = _range_norm_cv(s_pixels, er["S"])

            # Value
            v_pixels = hsv_flat[idx, 2]
            traits[f"obj_val_mean_{src_name}"] = float(np.mean(v_pixels))
            traits[f"obj_val_cv_{src_name}"]   = _range_norm_cv(v_pixels, er["V"])

            # ------ LAB channels ------

            # L (lightness)
            l_pixels = lab_flat[idx, 0]
            traits[f"obj_L_mean_{src_name}"] = float(np.mean(l_pixels))
            traits[f"obj_L_cv_{src_name}"]   = _range_norm_cv(l_pixels, er["L"])

            # a (green–red opponent axis; 128 = neutral in uint8 LAB)
            a_pixels = lab_flat[idx, 1]
            traits[f"obj_a_mean_{src_name}"] = float(np.mean(a_pixels))
            traits[f"obj_a_cv_{src_name}"]   = _range_norm_cv(a_pixels, er["a"])

            # b (blue–yellow opponent axis; 128 = neutral in uint8 LAB)
            b_pixels = lab_flat[idx, 2]
            traits[f"obj_b_mean_{src_name}"] = float(np.mean(b_pixels))
            traits[f"obj_b_cv_{src_name}"]   = _range_norm_cv(b_pixels, er["b"])

        metrics_by_label[label] = {
            "traits": traits,
            "qc": {
                "mask_pixel_count": pixel_count,
                "color_metrics_ok": True,
            },
        }

    return metrics_by_label


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _empirical_range(channel_array):
    """
    Return the empirical range (max - min) of a full-image channel array.
    Returns None if the range is zero to guard against zero-division downstream.
    """
    r = float(channel_array.max()) - float(channel_array.min())
    return r if r > 0 else None


def _range_norm_cv(pixels, empirical_range):
    """
    Range-normalized coefficient of variation (%).

    CV = (std(pixels) / empirical_range) * 100

    Uses the empirical range of the full image channel (pre-computed) as the
    denominator rather than the per-mask mean. This avoids instability for
    channels whose per-mask mean is near zero (LAB a, b) and ensures the CV
    is on a consistent scale across all objects in the image.

    Returns None if empirical_range is None or zero.
    """
    if empirical_range is None or empirical_range == 0:
        return None
    return float((np.std(pixels) / empirical_range) * 100.0)