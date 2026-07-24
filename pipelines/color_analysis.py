# --------------------------------------------------------------------
# Computes per-object colorimetric statistics across HSV, LAB, and RGB
# color channels for each labeled seed mask, on a single (already
# color-calibrated) image.
#
# Input:
#   - img             : color-calibrated BGR uint8 image
#   - labeled_objects : labeled mask (0=background, positive int = object),
#                       same labeling scheme consumed by shape_analysis.py
#
# For each object and each channel, computes six descriptive statistics:
#   mean, median, mode, min, max, std
#
# Scale conventions (fixed, not configurable):
#   - Hue (H):        circular quantity. All six statistics are reported
#                      in true color-wheel degrees [0, 360). OpenCV stores
#                      uint8 hue as [0, 180); values are doubled to degrees
#                      before any statistic is computed, so mode/min/max are
#                      on the same scale as the circular mean/std/median.
#                        - mean / std  -> scipy.stats.circmean / circstd
#                        - median      -> "circular median": pixels are
#                          recentered around the circular mean (wrapped to
#                          [0, 360)), a linear median is taken on the
#                          recentered values, then the shift is undone.
#                          This avoids the failure mode of a naive linear
#                          median when hue values straddle the 0/360
#                          wraparound point (common for red/tan seed color).
#                        - mode / min / max -> plain statistics on the
#                          degree-scaled values (doubling is a monotonic,
#                          order-preserving transform, so this is equivalent
#                          to computing on raw OpenCV hue and then doubling
#                          the result).
#   - Saturation, Value (S, V): OpenCV native uint8 scale [0, 255].
#   - Red, Green, Blue (R, G, B): native uint8 scale [0, 255].
#   - Lightness, a*, b* (L, a, b): OpenCV's uint8 L*a*b* encoding —
#                      L in [0, 255]; a, b in [0, 255] with 128 = neutral.
#                      (Not the standard CIE L in [0,100]/a,b in
#                      [-128,127] scale.)
#
# Output format mirrors calculate_size_shape(): dict keyed by label (int)
# with a 'traits' sub-dict (internal metric keys -> float) and a 'qc'
# sub-dict.
# --------------------------------------------------------------------

import numpy as np
import cv2
from scipy.stats import circmean, circstd


def calculate_color_metrics(img, labeled_objects):
    """
    Compute per-object colorimetric statistics across HSV, LAB, and RGB
    channels from a single color-calibrated image and a labeled mask.

    Args:
        img             : color-calibrated BGR uint8 image, shape (H, W, 3).
        labeled_objects : labeled mask (uint8/uint16/int), same spatial
                          dimensions as img. 0 = background; positive
                          integers label individual objects.

    Returns:
        metrics_by_label : dict keyed by label id (int).
            Each value is a dict with:
                'traits' : dict of internal metric key -> float (or {} if
                           the label has no pixels)
                'qc'     : dict with 'mask_pixel_count' (int) and
                           'color_metrics_ok' (bool)
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64).reshape(-1, 3)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64).reshape(-1, 3)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64).reshape(-1, 3)

    # Build label -> flat pixel index mapping in one sort pass instead of
    # scanning the full array once per label with a boolean comparison.
    flat_labels = labeled_objects.ravel()
    sort_order = np.argsort(flat_labels, kind="stable")
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
        idx = label_indices[label]
        pixel_count = len(idx)

        if pixel_count == 0:
            metrics_by_label[label] = {
                "traits": {},
                "qc": {"mask_pixel_count": 0, "color_metrics_ok": False},
            }
            continue

        traits = {}

        # ------ HSV: Hue (circular) ------
        hue_stats = _hue_stats(hsv[idx, 0])
        traits["obj_hue_circmean"] = hue_stats["circmean"]
        traits["obj_hue_circstd"] = hue_stats["circstd"]
        traits["obj_hue_median"] = hue_stats["median"]
        traits["obj_hue_mode"] = hue_stats["mode"]
        traits["obj_hue_min"] = hue_stats["min"]
        traits["obj_hue_max"] = hue_stats["max"]

        # ------ HSV: Saturation, Value ------
        _add_channel_stats(traits, "obj_sat", hsv[idx, 1])
        _add_channel_stats(traits, "obj_val", hsv[idx, 2])

        # ------ LAB: Lightness, a*, b* (OpenCV uint8 encoding) ------
        _add_channel_stats(traits, "obj_L", lab[idx, 0])
        _add_channel_stats(traits, "obj_a", lab[idx, 1])
        _add_channel_stats(traits, "obj_b", lab[idx, 2])

        # ------ RGB ------
        _add_channel_stats(traits, "obj_R", rgb[idx, 0])
        _add_channel_stats(traits, "obj_G", rgb[idx, 1])
        _add_channel_stats(traits, "obj_B", rgb[idx, 2])

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

def _channel_stats(pixels):
    """
    Generic per-channel descriptive statistics: mean, median, mode, min,
    max, std.

    Mode is computed via np.unique rather than scipy.stats.mode to avoid
    API differences across scipy versions (keepdims argument, return
    shape).
    """
    vals, counts = np.unique(pixels, return_counts=True)
    mode_val = float(vals[np.argmax(counts)])
    return {
        "mean": float(np.mean(pixels)),
        "median": float(np.median(pixels)),
        "mode": mode_val,
        "min": float(np.min(pixels)),
        "max": float(np.max(pixels)),
        "std": float(np.std(pixels)),
    }


def _add_channel_stats(traits, key_prefix, pixels):
    """Compute _channel_stats() for pixels and write obj_{prefix}_{stat} keys."""
    stats = _channel_stats(pixels)
    traits[f"{key_prefix}_mean"] = stats["mean"]
    traits[f"{key_prefix}_median"] = stats["median"]
    traits[f"{key_prefix}_mode"] = stats["mode"]
    traits[f"{key_prefix}_min"] = stats["min"]
    traits[f"{key_prefix}_max"] = stats["max"]
    traits[f"{key_prefix}_std"] = stats["std"]


def _hue_stats(hue_pixels_native):
    """
    Circular statistics for Hue.

    Args:
        hue_pixels_native: OpenCV native uint8 hue values, range [0, 180).

    Returns:
        dict with 'circmean', 'circstd', 'median', 'mode', 'min', 'max',
        all in true color-wheel degrees [0, 360).
    """
    hue_deg = hue_pixels_native.astype(np.float64) * 2.0  # [0, 180) -> [0, 360)

    circ_mean = float(circmean(hue_deg, high=360.0, low=0.0))
    circ_std = float(circstd(hue_deg, high=360.0, low=0.0))

    # Circular median: recenter pixels around the circular mean so the
    # wraparound point (0/360) sits opposite the data mass, take a plain
    # linear median, then undo the shift. Valid for unimodal hue
    # distributions (the expected case for a single seed's surface color).
    shifted = (hue_deg - circ_mean + 180.0) % 360.0
    median_deg = (float(np.median(shifted)) - 180.0 + circ_mean) % 360.0

    # Mode/min/max: doubling is monotonic and order-preserving, so these
    # are equivalent to computing on the raw OpenCV hue and doubling after.
    vals, counts = np.unique(hue_deg, return_counts=True)
    mode_deg = float(vals[np.argmax(counts)])

    return {
        "circmean": circ_mean,
        "circstd": circ_std,
        "median": median_deg,
        "mode": mode_deg,
        "min": float(np.min(hue_deg)),
        "max": float(np.max(hue_deg)),
    }
