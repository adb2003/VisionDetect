"""CSV and plain-text exports built from a stored result.

Both are generated from the same cached payload the dashboard renders, so a
downloaded file can never disagree with what was on screen when the button was
pressed.
"""

from __future__ import annotations

import csv
import io
from typing import Any

CSV_COLUMNS = [
    "object_id",
    "class_name",
    "class_id",
    "confidence",
    "confidence_percent",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "width",
    "height",
    "center_x",
    "center_y",
    "area_px",
]


def detections_csv(payload: dict[str, Any]) -> str:
    """The detection table as CSV, one row per detected object."""
    buffer = io.StringIO()
    # QUOTE_MINIMAL with \r\n: Excel's expectation, and class names never
    # contain commas in any standard label set but may in a custom one.
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)

    for detection in payload.get("detections", []):
        bbox = detection["bbox"]
        writer.writerow(
            [
                detection["object_id"],
                detection["class_name"],
                detection["class_id"],
                f"{detection['confidence']:.4f}",
                f"{detection['confidence'] * 100:.1f}",
                bbox["x_min"],
                bbox["y_min"],
                bbox["x_max"],
                bbox["y_max"],
                detection["width"],
                detection["height"],
                detection["center"]["x"],
                detection["center"]["y"],
                detection["area"],
            ]
        )

    return buffer.getvalue()


def summary_report(payload: dict[str, Any]) -> str:
    """A human-readable report: image, model, settings, counts, statistics."""
    image = payload["image"]
    info = payload["model_info"]
    settings = payload["settings"]
    summary = payload["summary"]

    lines: list[str] = []
    add = lines.append

    add("VisionDetect - Object Detection Report")
    add("=" * 52)
    add("")
    add(f"Analysis ID      : {payload['result_id']}")
    add(f"Generated (UTC)  : {payload['created_at']}")
    add("")

    add("Image")
    add("-" * 52)
    add(f"Filename         : {image['filename']}")
    add(f"Dimensions       : {image['width']} x {image['height']} px")
    add(f"Format           : {image['format'].upper()}")
    add(f"File size        : {image['size_bytes'] / 1024:.1f} KB")
    add("")

    add("Model")
    add("-" * 52)
    add(f"Model            : {info['label']} ({info['name']})")
    add(f"Backend          : {info['backend']}")
    add(f"Task             : {info['task']}")
    add(f"Inference device : {info['device'].upper()}")
    add(f"Network input    : {info['input_size']} x {info['input_size']} px")
    add(f"Class vocabulary : {info['class_count']} classes")
    add("")

    add("Detection settings")
    add("-" * 52)
    add(f"Confidence       : {settings['confidence_threshold']:.2f}")
    add(f"IoU threshold    : {settings['iou_threshold']:.2f}")
    add(f"Max detections   : {settings['max_detections']}")
    add("")

    add("Results")
    add("-" * 52)
    add(f"Total objects    : {summary['total_objects']}")
    add(f"Unique classes   : {summary['unique_classes']}")

    if summary["total_objects"]:
        add(f"Highest conf.    : {summary['max_confidence'] * 100:.1f}%")
        add(f"Average conf.    : {summary['avg_confidence'] * 100:.1f}%")
        add(f"Lowest conf.     : {summary['min_confidence'] * 100:.1f}%")
    add(f"Inference time   : {payload['inference_time_ms']:.1f} ms")
    add(f"Total processing : {payload['processing_time_ms']:.1f} ms")
    add("")

    add("Object counts")
    add("-" * 52)
    if payload.get("classes"):
        width = max(len(entry["class_name"]) for entry in payload["classes"])
        for entry in payload["classes"]:
            add(
                f"{entry['class_name']:<{width}}  {entry['count']:>4}  "
                f"(avg {entry['avg_confidence'] * 100:.1f}%, "
                f"max {entry['max_confidence'] * 100:.1f}%)"
            )
    else:
        add("No objects were detected above the confidence threshold.")
    add("")

    add("Detections")
    add("-" * 52)
    if payload.get("detections"):
        add(
            f"{'ID':>4}  {'Class':<18} {'Conf':>7}  "
            f"{'Bounding box (x1,y1,x2,y2)':<30} {'W x H':>12}"
        )
        for detection in payload["detections"]:
            bbox = detection["bbox"]
            box = (
                f"{bbox['x_min']},{bbox['y_min']},"
                f"{bbox['x_max']},{bbox['y_max']}"
            )
            size = f"{detection['width']} x {detection['height']}"
            add(
                f"{detection['object_id']:>4}  {detection['class_name']:<18} "
                f"{detection['confidence'] * 100:>6.1f}%  {box:<30} {size:>12}"
            )
    else:
        add("None.")
    add("")

    add("-" * 52)
    add(
        "All figures above are computed from this model's predictions on this "
        "image. No values are sampled or defaulted."
    )

    return "\n".join(lines) + "\n"
