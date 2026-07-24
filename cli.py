"""
cli.py — standalone entry point for the BreedBase image analysis pipeline.

Usage:
    bb-analyze image.jpg --output-dir ./results
    bb-analyze image.jpg --output-dir ./results --format csv
    bb-analyze path/to/images_dir --output-dir ./results          # batch
    bb-analyze path/to/images_dir --output-dir ./results --format csv
"""

import sys
import os
import csv
import io
import json
import logging
import argparse
import uuid
from pathlib import Path
from datetime import datetime, timezone

import cv2

from process_image import (
    analyze_image,
    PIPELINE_NAME,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    DEFAULT_MARKER_DIAMETER_IN,
)

# Directory (batch) mode scans only the top level of the given directory
# (no recursion) for files with these extensions, matching api/app.py's
# ALLOWED_EXTENSIONS.
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _find_images(dir_path):
    """Sorted, non-recursive list of image paths directly inside dir_path."""
    return sorted(
        p for p in Path(dir_path).iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )


def _to_csv(envelopes):
    """Flatten one or more envelopes into a single CSV (one row per object)."""
    rows = []
    for envelope in envelopes:
        qc = envelope["qc"]
        pipeline = envelope["pipeline"]
        for obj in envelope["objects"]:
            row = {
                "job_id": envelope["job_id"],
                "timestamp": envelope["timestamp"],
                "pipeline_name": pipeline["name"],
                "pipeline_version": pipeline["version"],
                "image_filename": envelope["input"]["image_filename"],
                "qc_analysis_pass": qc.get("analysis_pass"),
                "qc_color_card_present": qc.get("color_card_present"),
                "qc_size_marker_detected": qc.get("size_marker_detected"),
                "object_count": qc.get("object_count"),
                "object_id": obj["object_id"],
                "source_label": obj["source_label"],
                "bbox_x": obj["bbox"]["x"],
                "bbox_y": obj["bbox"]["y"],
                "bbox_w": obj["bbox"]["w"],
                "bbox_h": obj["bbox"]["h"],
                "qc_contour_found": obj["qc"].get("contour_found"),
                "qc_ellipse_fit_ok": obj["qc"].get("ellipse_fit_ok"),
                "qc_mask_pixel_count": obj["qc"].get("mask_pixel_count"),
                "qc_color_metrics_ok": obj["qc"].get("color_metrics_ok"),
            }
            for trait_key, trait_val in obj["traits"].items():
                row[trait_key] = trait_val["value"]
            rows.append(row)

    if not rows:
        return ""

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _analyze_and_write_overlay(image_path, output_dir, marker_diameter_in, host_url):
    """
    Run analyze_image() on a single image, write its overlay PNG, and build
    its envelope dict. Shared by both single-image and batch modes.

    Raises on analysis failure (caller decides how to handle it).
    """
    result = analyze_image(image_path, marker_diameter_in=marker_diameter_in)

    job_id = str(uuid.uuid4())
    filename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(filename)[0]

    overlay_name = f"{name_no_ext}_ResultImage_{job_id}.png"
    overlay_path = os.path.join(output_dir, overlay_name)
    cv2.imwrite(overlay_path, result["overlay_img"])

    overlay_url = f"{host_url}download/{overlay_name}" if host_url else overlay_path
    derived_images = [{"role": "overlay", "filename": overlay_name, "url": overlay_url}]

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": {"name": PIPELINE_NAME, "version": PIPELINE_VERSION},
        "input": {"image_filename": filename},
        "qc": result["qc"],
        "traits_emitted": result["traits_emitted"],
        "derived_images": derived_images,
        "objects": result["objects"],
    }
    return envelope, overlay_path


def run_batch(args, host_url):
    image_paths = _find_images(args.image_path)
    if not image_paths:
        print(json.dumps({
            "error": f"No images (.jpg/.jpeg/.png) found in directory: {args.image_path}",
            "error_type": "NoImagesFound",
        }))
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    envelopes = []
    errors = []
    overlay_paths = []

    for image_path in image_paths:
        try:
            envelope, overlay_path = _analyze_and_write_overlay(
                str(image_path), args.output_dir, args.marker_diameter, host_url
            )
            envelopes.append(envelope)
            overlay_paths.append(overlay_path)
        except Exception as e:
            logging.warning("Skipping %s: %s", image_path, e)
            errors.append({
                "image_filename": image_path.name,
                "error": str(e),
                "error_type": type(e).__name__,
            })

    batch_id = str(uuid.uuid4())
    batch_result = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": {"name": PIPELINE_NAME, "version": PIPELINE_VERSION},
        "input_dir": str(args.image_path),
        "images_processed": len(envelopes),
        "images_failed": len(errors),
        "results": envelopes,
        "errors": errors,
    }

    sidecar_name = f"batch_metadata_{batch_id}.{args.format}"
    sidecar_path = os.path.join(args.output_dir, sidecar_name)

    if args.format == "json":
        with open(sidecar_path, "w") as f:
            json.dump(batch_result, f, indent=2)
    else:
        with open(sidecar_path, "w", newline="") as f:
            f.write(_to_csv(envelopes))

    for overlay_path in overlay_paths:
        print(f"Overlay:  {overlay_path}")
    print(f"Results:  {sidecar_path}")
    print(f"Processed {len(envelopes)} image(s), {len(errors)} failed.")

    sys.exit(0 if envelopes else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Run the BreedBase image analysis pipeline on a single image "
                    "or on all images in a directory."
    )
    parser.add_argument(
        "image_path",
        help="Path to an input image, or a directory of images to batch-process "
             "(top-level .jpg/.jpeg/.png files only, no recursion)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory to save overlay image(s) and the results sidecar",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format for the results sidecar (default: json)",
    )
    parser.add_argument(
        "--marker-diameter",
        type=float,
        default=DEFAULT_MARKER_DIAMETER_IN,
        help=f"Physical diameter of the size marker in inches (default: {DEFAULT_MARKER_DIAMETER_IN})",
    )
    parser.add_argument(
        "--host-url",
        default=None,
        help="Base URL used to build download links in derived_images (optional)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    host_url = args.host_url.rstrip("/") + "/" if args.host_url else None

    if os.path.isdir(args.image_path):
        run_batch(args, host_url)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        envelope, overlay_path = _analyze_and_write_overlay(
            args.image_path, args.output_dir, args.marker_diameter, host_url
        )
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        print(json.dumps({"error": str(e), "error_type": type(e).__name__}))
        sys.exit(1)

    filename = os.path.basename(args.image_path)
    name_no_ext = os.path.splitext(filename)[0]
    sidecar_name = f"{name_no_ext}_metadata_{envelope['job_id']}.{args.format}"
    sidecar_path = os.path.join(args.output_dir, sidecar_name)

    if args.format == "json":
        with open(sidecar_path, "w") as f:
            json.dump(envelope, f, indent=2)
    else:
        with open(sidecar_path, "w", newline="") as f:
            f.write(_to_csv([envelope]))

    print(f"Overlay:  {overlay_path}")
    print(f"Results:  {sidecar_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
