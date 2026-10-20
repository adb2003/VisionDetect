"""The detection pipeline: image in, structured result out.

One function owns the sequence -- run the model, order the detections,
summarise them, render the annotation -- so a fresh upload and a re-run with new
settings produce byte-identical structures. Every number in the response is
computed here from the model's actual output; nothing is defaulted, sampled or
carried over from a previous run.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.detector import Detector, RawDetection, class_colour_hex
from app.image_utils import annotate, encode_image, to_data_url
from app.schemas import (
    BoundingBox,
    ClassSummary,
    DetectionOut,
    DetectionResponse,
    DetectionSettings,
    DetectionSummary,
    ImageMeta,
    ModelMeta,
    Point,
)


def new_result_id() -> str:
    return f"det_{uuid.uuid4().hex[:12]}"


def run_detection(
    *,
    detector: Detector,
    image: np.ndarray,
    filename: str,
    image_format: str,
    size_bytes: int,
    confidence: float,
    iou: float,
    max_detections: int,
    result_id: str | None = None,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
    include_data_url: bool = True,
    annotation_format: str = "png",
) -> tuple[DetectionResponse, list[RawDetection], bytes]:
    """Detect, summarise and annotate.

    Returns the response, the raw detections in the same order as
    `response.detections`, and the encoded annotated image.
    """
    started = time.perf_counter()

    raw, inference_ms = detector.detect(
        image,
        confidence=confidence,
        iou=iou,
        max_detections=max_detections,
    )

    # Most confident first, so object_id 1 is the strongest detection and the
    # table's default order needs no sorting in the browser.
    raw = sorted(raw, key=lambda d: -d.confidence)

    detections = [
        DetectionOut(
            object_id=index,
            class_id=item.class_id,
            class_name=item.class_name,
            confidence=round(item.confidence, 4),
            bbox=BoundingBox(
                x_min=item.x_min,
                y_min=item.y_min,
                x_max=item.x_max,
                y_max=item.y_max,
            ),
            width=item.width,
            height=item.height,
            center=Point(x=round(item.centre[0], 1), y=round(item.centre[1], 1)),
            area=item.area,
            color=class_colour_hex(item.class_id),
        )
        for index, item in enumerate(raw, start=1)
    ]

    summary = _summarise(raw)
    classes = _class_summaries(raw)

    annotated = annotate(
        image,
        raw,
        show_boxes=show_boxes,
        show_labels=show_labels,
        show_confidence=show_confidence,
    )
    annotated_bytes = encode_image(annotated, annotation_format)

    identifier = result_id or new_result_id()
    info = detector.info

    response = DetectionResponse(
        result_id=identifier,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        image=ImageMeta(
            filename=filename,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            format=image_format,
            size_bytes=size_bytes,
        ),
        model=info.name,
        model_info=ModelMeta(
            name=info.name,
            label=info.label,
            backend=info.backend,
            task=info.task,
            device=info.device,
            input_size=info.input_size,
            class_count=len(info.class_names),
        ),
        settings=DetectionSettings(
            confidence_threshold=confidence,
            iou_threshold=iou,
            max_detections=max_detections,
        ),
        processing_time_ms=round((time.perf_counter() - started) * 1000.0, 1),
        inference_time_ms=round(inference_ms, 1),
        summary=summary,
        classes=classes,
        detections=detections,
        annotated_image=(
            to_data_url(annotated_bytes, annotation_format) if include_data_url else None
        ),
        annotated_image_url=f"/api/results/{identifier}/image",
    )

    return response, raw, annotated_bytes


def _summarise(detections: list[RawDetection]) -> DetectionSummary:
    """Roll up the run. An image with nothing in it is a valid result, not an
    error, so the zero case returns real zeros rather than nulls."""
    if not detections:
        return DetectionSummary(
            total_objects=0,
            unique_classes=0,
            max_confidence=0.0,
            min_confidence=0.0,
            avg_confidence=0.0,
            class_counts={},
        )

    confidences = [d.confidence for d in detections]
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.class_name] = counts.get(detection.class_name, 0) + 1

    return DetectionSummary(
        total_objects=len(detections),
        unique_classes=len(counts),
        max_confidence=round(max(confidences), 4),
        min_confidence=round(min(confidences), 4),
        avg_confidence=round(sum(confidences) / len(confidences), 4),
        class_counts=dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    )


def _class_summaries(detections: list[RawDetection]) -> list[ClassSummary]:
    grouped: dict[str, list[RawDetection]] = {}
    for detection in detections:
        grouped.setdefault(detection.class_name, []).append(detection)

    summaries = [
        ClassSummary(
            class_id=members[0].class_id,
            class_name=name,
            count=len(members),
            color=class_colour_hex(members[0].class_id),
            avg_confidence=round(
                sum(m.confidence for m in members) / len(members), 4
            ),
            max_confidence=round(max(m.confidence for m in members), 4),
        )
        for name, members in grouped.items()
    ]
    # Largest group first: that is the order both charts and the filter list
    # want, and doing it once here keeps them consistent.
    summaries.sort(key=lambda summary: (-summary.count, summary.class_name))
    return summaries


def payload_without_image(response: DetectionResponse) -> dict[str, Any]:
    """Response as a dict with the inline data URL stripped.

    What gets cached and re-served. The base64 copy of the annotated image is
    typically larger than everything else combined and is always re-derivable
    from the stored pixels, so keeping it would double the cache's footprint for
    no gain.
    """
    data = response.model_dump(by_alias=True)
    data["annotated_image"] = None
    return data
