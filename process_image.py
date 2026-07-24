import sys
import argparse
import os
import uuid
import json
import logging
from datetime import datetime, timezone
import cv2
import math

from pipelines.utils import enforce_https, readimage
from pipelines.object_labeling import label_objects_rowwise
# from pipelines.ref_mask import create_chip_mask, create_masks
from pipelines.ref_mask import create_masks
from pipelines.seed_mask import create_seed_mask
# from pipelines.color_correction import apply_color_correction
from pipelines.color_correction_v2 import apply_color_correction
from pipelines.size_marker_metadata import size_marker
from pipelines.shape_analysis import calculate_size_shape
from pipelines.color_analysis import calculate_color_metrics
from pipelines.create_chip_mask_v3 import create_chip_mask, REFERENCE_RGB

logger = logging.getLogger(__name__)

PIPELINE_NAME = os.getenv("PIPELINE_NAME", "seed_size_shape")
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "0.1.0")

# Envelope schema version — bump when the canonical envelope shape changes.
SCHEMA_VERSION = "1.0"

# Physical diameter of the size marker, in inches. Pipeline-internal default;
# not part of the framework HTTP contract (see API_Standardization_Tracker.md Subtask 4).
DEFAULT_MARKER_DIAMETER_IN = float(os.getenv("MARKER_DIAMETER_IN", "0.75"))

# --------------------------------------------------------------------
# Internal metric keys from analysis/shape_analysis.py 
# -> (Public trait key, unit, rounding digits)
# --------------------------------------------------------------------

TRAITS_MAP = {
        # internal_key: (public_trait_key, unit, ndigits)
        "obj_area_mask": ("Object Area From Segmentation Mask|IMGSTAT:0000006", "mm^2", 2),
        "obj_area_hull": ("Object Convex Hull Area|IMGSTAT:0000007", "mm^2", 2),
        "obj_perimeter_mask": ("Object Perimeter From Segmentation Mask|IMGSTAT:0000010", "mm", 2),
        "obj_solidity": ("Object Solidity|IMGSTAT:0000011", None, 4),
        "obj_diam_max_ellipse": ("Object Maximum Diameter From Fitted Ellipse|IMGSTAT:0000008", "mm", 2),
        "obj_diam_min_ellipse": ("Object Minimum Diameter From Fitted Ellipse|IMGSTAT:0000009", "mm", 2),
    }

# --------------------------------------------------------------------
# Internal metric keys from pipelines/color_analysis.py
# -> (Public trait key, unit, rounding digits)
#
# Scale conventions (see pipelines/color_analysis.py docstring):
#   - Hue: degrees [0, 360), unit "deg"
#   - Saturation, Value, Red, Green, Blue: OpenCV native uint8 [0, 255], unit None
#   - Lightness, a*, b*: OpenCV uint8 LAB encoding [0, 255] (128 = neutral), unit None
# --------------------------------------------------------------------

COLOR_TRAITS_MAP = {
        # internal_key: (public_trait_key, unit, ndigits)

        # --- HSV: Hue (circular) ---
        "obj_hue_circmean": ("Object Hue Circular mean (deg)|IMGSTAT:0000087", "deg", 2),
        "obj_hue_circstd":  ("Object Hue Circular Standard Deviation (deg)|IMGSTAT:0000089", "deg", 2),
        "obj_hue_median":   ("Object Hue Median (deg)|IMGSTAT:0000091", "deg", 2),
        "obj_hue_mode":     ("Object Hue Mode|IMGSTAT:0000144", "deg", 2),
        "obj_hue_min":      ("Object Hue Minimum|IMGSTAT:0000147", "deg", 2),
        "obj_hue_max":      ("Object Hue Maximum|IMGSTAT:0000150", "deg", 2),

        # --- HSV: Saturation ---
        "obj_sat_mean":   ("Object Saturation Arithmetic Mean|IMGSTAT:0000093", None, 2),
        "obj_sat_std":    ("Object Saturation Standard Deviation|IMGSTAT:0000095", None, 2),
        "obj_sat_median": ("Object Saturation Median|IMGSTAT:0000097", None, 2),
        "obj_sat_mode":   ("Object Saturation Mode|IMGSTAT:0000145", None, 2),
        "obj_sat_min":    ("Object Saturation Minimum|IMGSTAT:0000148", None, 2),
        "obj_sat_max":    ("Object Saturation Maximum|IMGSTAT:0000151", None, 2),

        # --- HSV: Value ---
        "obj_val_mean":   ("Object Value Arithmetic Mean|IMGSTAT:0000099", None, 2),
        "obj_val_std":    ("Object Value Standard Deviation|IMGSTAT:0000101", None, 2),
        "obj_val_median": ("Object Value Median|IMGSTAT:0000103", None, 2),
        "obj_val_mode":   ("Object Value Mode|IMGSTAT:0000146", None, 2),
        "obj_val_min":    ("Object Value Minimum|IMGSTAT:0000149", None, 2),
        "obj_val_max":    ("Object Value Maximum|IMGSTAT:0000152", None, 2),

        # --- RGB: Red ---
        "obj_R_mean":   ("Object Red Arithmetic Mean|IMGSTAT:0000108", None, 2),
        "obj_R_median": ("Object Red Median|IMGSTAT:0000114", None, 2),
        "obj_R_mode":   ("Object Red Mode|IMGSTAT:0000120", None, 2),
        "obj_R_min":    ("Object Red Minimum|IMGSTAT:0000126", None, 2),
        "obj_R_max":    ("Object Red Maximum|IMGSTAT:0000132", None, 2),
        "obj_R_std":    ("Object Red Standard Deviation|IMGSTAT:0000138", None, 2),

        # --- RGB: Green ---
        "obj_G_mean":   ("Object Green Arithmetic Mean|IMGSTAT:0000109", None, 2),
        "obj_G_median": ("Object Green Median|IMGSTAT:0000115", None, 2),
        "obj_G_mode":   ("Object Green Mode|IMGSTAT:0000121", None, 2),
        "obj_G_min":    ("Object Green Minimum|IMGSTAT:0000127", None, 2),
        "obj_G_max":    ("Object Green Maximum|IMGSTAT:0000133", None, 2),
        "obj_G_std":    ("Object Green Standard Deviation|IMGSTAT:0000139", None, 2),

        # --- RGB: Blue ---
        "obj_B_mean":   ("Object Blue Arithmetic Mean|IMGSTAT:0000110", None, 2),
        "obj_B_median": ("Object Blue Median|IMGSTAT:0000116", None, 2),
        "obj_B_mode":   ("Object Blue Mode|IMGSTAT:0000122", None, 2),
        "obj_B_min":    ("Object Blue Minimum|IMGSTAT:0000128", None, 2),
        "obj_B_max":    ("Object Blue Maximum|IMGSTAT:0000134", None, 2),
        "obj_B_std":    ("Object Blue Standard Deviation|IMGSTAT:0000140", None, 2),

        # --- LAB: Lightness ---
        "obj_L_mean":   ("Object Lightness Arithmetic Mean|IMGSTAT:0000111", None, 2),
        "obj_L_median": ("Object Lightness Median|IMGSTAT:0000117", None, 2),
        "obj_L_mode":   ("Object Lightness Mode|IMGSTAT:0000123", None, 2),
        "obj_L_min":    ("Object Lightness Minimum|IMGSTAT:0000129", None, 2),
        "obj_L_max":    ("Object Lightness Maximum|IMGSTAT:0000135", None, 2),
        "obj_L_std":    ("Object Lightness Standard Deviation|IMGSTAT:0000141", None, 2),

        # --- LAB: a* (red-green) ---
        "obj_a_mean":   ("Object Red-Green (a*) Arithmetic Mean|IMGSTAT:0000112", None, 2),
        "obj_a_median": ("Object Red-Green (a*) Median|IMGSTAT:0000118", None, 2),
        "obj_a_mode":   ("Object Red-Green (a*) Mode|IMGSTAT:0000124", None, 2),
        "obj_a_min":    ("Object Red-Green (a*) Minimum|IMGSTAT:0000130", None, 2),
        "obj_a_max":    ("Object Red-Green (a*) Maximum|IMGSTAT:0000136", None, 2),
        "obj_a_std":    ("Object Red-Green (a*) Standard Deviation|IMGSTAT:0000142", None, 2),

        # --- LAB: b* (blue-yellow) ---
        "obj_b_mean":   ("Object Blue-Yellow (b*) Arithmetic Mean|IMGSTAT:0000113", None, 2),
        "obj_b_median": ("Object Blue-Yellow (b*) Median|IMGSTAT:0000119", None, 2),
        "obj_b_mode":   ("Object Blue-Yellow (b*) Mode|IMGSTAT:0000125", None, 2),
        "obj_b_min":    ("Object Blue-Yellow (b*) Minimum|IMGSTAT:0000131", None, 2),
        "obj_b_max":    ("Object Blue-Yellow (b*) Maximum|IMGSTAT:0000137", None, 2),
        "obj_b_std":    ("Object Blue-Yellow (b*) Standard Deviation|IMGSTAT:0000143", None, 2),
    }

def _to_float(x):
    """Best-effort conversion to float; returns None if not convertible."""
    if x is None:
        return None
    try:
        # Handles numpy scalars cleanly
        return float(x)
    except (TypeError, ValueError):
        return None

def safe_round(x, ndigits=2):
    """
    Convert to float and round safely.
    Returns None for None, non-numeric, NaN, or Inf.
    """
    fx = _to_float(x)
    if fx is None or math.isnan(fx) or math.isinf(fx):
        return None
    return round(fx, ndigits)

def _meta_value(sm_metadata, trait, default=None):
    return next((item["value"] for item in sm_metadata if item.get("trait") == trait), default)

def analyze_image(image_path, marker_diameter_in=DEFAULT_MARKER_DIAMETER_IN):
    """
    Run the image analysis pipeline on a single image.

    Pure analysis function — no file I/O, no job IDs, no URLs.
    Returns a dict with keys:
      - qc:            image-level QC flags
      - objects:       list of per-object trait dicts
      - traits_emitted: list of public trait keys (always all of TRAITS_MAP)
      - overlay_img:   numpy array of the result overlay image
    Raises exceptions on failure.
    """
    selected_keys = list(TRAITS_MAP.keys())
    color_keys = list(COLOR_TRAITS_MAP.keys())
    traits_emitted = [TRAITS_MAP[k][0] for k in selected_keys] + \
                      [COLOR_TRAITS_MAP[k][0] for k in color_keys]

    # --------------------------------------------------------------------
    # Read image
    # --------------------------------------------------------------------
    img, img_filename = readimage(filename=image_path)

    # --------------------------------------------------------------------
    # Reference masks (color card + size marker)
    # --------------------------------------------------------------------
    cc_mask, sm_mask = create_masks(img, raise_errors=False)

    # Chip mask for color correction
    # chip_mask = create_chip_mask(img, cc_mask)
    chip_mask = create_chip_mask(
        img = img,
        cc_mask = cc_mask,
        reference_rgb = REFERENCE_RGB,
        n_cols = 4,
        n_rows = 6,
        valley_threshold = None,
        valley_threshold_offset = 20, #only used when valley_threshold=None
        min_valley_distance = 40,
        edge_margin_frac = 0.08,
        min_band_width_frac = 0.60,
        max_band_width_frac = 1.3,
        min_valid_chip_fraction = 0.75,
        max_acceptable_delta_e = 60.0,
        return_overlay = False, 
        return_chip_data = False)

    # Color correction
    corrected_img, _ = apply_color_correction(img, chip_mask, method = "affine")
    

    # --------------------------------------------------------------------
    # Seed/object masks
    # --------------------------------------------------------------------
    seed_mask = create_seed_mask(corrected_img, cc_mask, sm_mask)

    # --------------------------------------------------------------------
    # Size marker metadata (calibration)
    # --------------------------------------------------------------------
    sm_metadata = size_marker(sm_mask, marker_diameter_in)
    size_marker_detected = bool(_meta_value(sm_metadata, "size_marker_detected", False))

    # --------------------------------------------------------------------
    # Label objects
    # --------------------------------------------------------------------
    labeled_mask, labeled_img = label_objects_rowwise(
        seed_mask, corrected_img, output_mask_path=None, display_result=False
    )

    # --------------------------------------------------------------------
    # Shape analysis (only if calibration present)
    # --------------------------------------------------------------------
    size_data = {}
    color_data = {}
    overlay_img = labeled_img.copy()

    if size_marker_detected:
        size_data, overlay_img = calculate_size_shape(labeled_img, labeled_mask, sm_metadata)

        # ----------------------------------------------------------------
        # Color analysis — performed on the color-calibrated image
        # (corrected_img), keyed by the same object labels as size_data.
        # ----------------------------------------------------------------
        color_data = calculate_color_metrics(corrected_img, labeled_mask)

    # --------------------------------------------------------------------
    # QC flags (image-level)
    # --------------------------------------------------------------------
    color_card_present = bool(cc_mask is not None and int(cc_mask.max()) > 0)
    object_count = len(size_data)

    analysis_pass = True
    if not size_marker_detected:
        analysis_pass = False
    if object_count == 0:
        analysis_pass = False

    # --------------------------------------------------------------------
    # Build objects list with trait dicts
    # --------------------------------------------------------------------
    objects = []
    if analysis_pass:
        for idx, (label_id, obj_data) in enumerate(size_data.items(), start=1):
            obj_id = f"obj_{idx:03d}"
            traits_in = (obj_data or {}).get("traits", {})
            color_obj = color_data.get(label_id, {})
            color_traits_in = (color_obj or {}).get("traits", {})
            traits_out = {}

            for internal_key in selected_keys:
                public_key, unit, ndigits = TRAITS_MAP[internal_key]
                raw = traits_in.get(internal_key, None)
                val = safe_round(raw, ndigits=ndigits) if ndigits is not None else _to_float(raw)
                traits_out[public_key] = {"value": val, "unit": unit}

            for internal_key in color_keys:
                public_key, unit, ndigits = COLOR_TRAITS_MAP[internal_key]
                raw = color_traits_in.get(internal_key, None)
                val = safe_round(raw, ndigits=ndigits) if ndigits is not None else _to_float(raw)
                traits_out[public_key] = {"value": val, "unit": unit}

            qc_out = dict((obj_data or {}).get("qc") or {})
            qc_out.update((color_obj or {}).get("qc") or {})

            objects.append({
                "object_id": obj_id,
                "source_label": str(label_id),
                "bbox": (obj_data or {}).get("bbox"),
                "qc": qc_out,
                "traits": traits_out,
            })

    return {
        "qc": {
            "analysis_pass": analysis_pass,
            "color_card_present": color_card_present,
            "size_marker_detected": size_marker_detected,
            "object_count": object_count,
        },
        "objects": objects,
        "traits_emitted": traits_emitted,
        "overlay_img": overlay_img,
    }


def process_image(image_path, results_dir, host_url=None, marker_diameter_in=DEFAULT_MARKER_DIAMETER_IN):
    """
    Run the reference image analysis pipeline on a single image.

    Thin wrapper around analyze_image() that handles all file I/O:
    writes the overlay image and JSON sidecar to results_dir.

    Returns a Python dict (result envelope). Caller is responsible for printing JSON.
    Raises exceptions on failure.
    """
    os.makedirs(results_dir, exist_ok=True)
    job_id = str(uuid.uuid4())

    result = analyze_image(image_path, marker_diameter_in=marker_diameter_in)

    # --------------------------------------------------------------------
    # Output filenames
    # --------------------------------------------------------------------
    filename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(filename)[0]

    composite_image_name = f"{name_no_ext}_ResultImage_{job_id}.png"
    composite_image_path = os.path.join(results_dir, composite_image_name)
    cv2.imwrite(composite_image_path, result["overlay_img"])

    # Host URL handling
    host_url = os.environ.get('HOSTURL') if not host_url else host_url
    host_url = enforce_https(host_url) if host_url else host_url
    composite_url = f"{host_url}download/{composite_image_name}" if host_url else composite_image_path

    # --------------------------------------------------------------------
    # Canonical envelope
    # --------------------------------------------------------------------
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": {"name": PIPELINE_NAME, "version": PIPELINE_VERSION},
        "input": {"image_filename": filename},
        "qc": result["qc"],
        "traits_emitted": result["traits_emitted"],
        "derived_images": [
            {"role": "overlay", "filename": composite_image_name, "url": composite_url}
        ],
        "objects": result["objects"],
    }

    # Save JSON sidecar
    result_json_name = f"{name_no_ext}_metadata_{job_id}.json"
    result_json_path = os.path.join(results_dir, result_json_name)
    with open(result_json_path, 'w') as f:
        json.dump(envelope, f, indent=2)

    return envelope


def main():
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', 'server.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [process_image] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr)
        ]
    )

    parser = argparse.ArgumentParser(description="Run the reference image analysis pipeline.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("results_dir", help="Directory to save outputs")
    parser.add_argument("--host_url", help="Base URL for download links")
    parser.add_argument("--marker_diameter_in", type=float, default=DEFAULT_MARKER_DIAMETER_IN,
                    help=f"Physical diameter of the size marker in inches (default: {DEFAULT_MARKER_DIAMETER_IN})")
    args = parser.parse_args()

    try:
        payload = process_image(
            args.image_path,
            args.results_dir,
            host_url=args.host_url,
            marker_diameter_in=args.marker_diameter_in,
        )
        print(json.dumps(payload))
        sys.exit(0)

    except Exception as e:
        logger.exception("Pipeline failed")
        error_payload = {
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(json.dumps(error_payload))
        sys.exit(1)


if __name__ == "__main__":
    main()
