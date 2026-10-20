"""Tests for detection-head decoding.

Two head layouts are in use. The anchor-style head that YOLOv8 and YOLO11 emit
needs decoding and non-maximum suppression; the end-to-end head that YOLOv10
and YOLO26 emit has already done both and hands back a finished list.

Confusing them is not a crash, it is wrong boxes, so the discrimination between
them is tested directly rather than left to the first real image.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.detector import (
    Letterbox,
    decode_end2end,
    decode_yolo,
    is_end2end,
    letterbox,
)


# --------------------------------------------------------------------------
# Telling the two heads apart
# --------------------------------------------------------------------------


def test_end2end_output_is_recognised() -> None:
    """YOLO26 nano emits exactly this shape: 300 detections of 6 values."""
    assert is_end2end(np.zeros((1, 300, 6), dtype=np.float32))


def test_anchor_head_is_not_mistaken_for_end2end() -> None:
    """(1, 84, 8400) is COCO's anchor-style head and must decode the other way."""
    assert not is_end2end(np.zeros((1, 84, 8400), dtype=np.float32))


def test_a_six_class_anchor_head_is_not_mistaken_for_end2end() -> None:
    """The awkward case: a 6-channel axis exists, but it is the channel axis.

    A model with two classes emits (1, 6, 8400). The trailing axis is the anchor
    count, so the end-to-end test must look at the right axis.
    """
    assert not is_end2end(np.zeros((1, 6, 8400), dtype=np.float32))


# --------------------------------------------------------------------------
# End-to-end decoding
# --------------------------------------------------------------------------


def test_end2end_splits_rows_into_boxes_scores_and_classes() -> None:
    output = np.array(
        [[[10.0, 20.0, 110.0, 220.0, 0.9, 3.0], [5.0, 6.0, 15.0, 26.0, 0.4, 7.0]]],
        dtype=np.float32,
    )
    boxes, scores, class_ids = decode_end2end(output)

    assert boxes.tolist() == [[10.0, 20.0, 110.0, 220.0], [5.0, 6.0, 15.0, 26.0]]
    assert scores == pytest.approx([0.9, 0.4])
    assert class_ids.tolist() == [3, 7]
    assert class_ids.dtype == np.int32


def test_end2end_boxes_are_corners_not_centres() -> None:
    """The two heads differ here, and getting it wrong halves every box.

    An anchor head emits centre-form and needs converting; an end-to-end head
    emits corners already. Feeding corner-form boxes through the centre-form
    decoder would place them at a quarter size around the wrong point.
    """
    output = np.array([[[100.0, 100.0, 300.0, 500.0, 0.8, 0.0]]], dtype=np.float32)
    boxes, _scores, _ids = decode_end2end(output)

    x_min, y_min, x_max, y_max = boxes[0]
    assert (x_max - x_min, y_max - y_min) == (200.0, 400.0)


def test_end2end_survives_an_empty_result() -> None:
    boxes, scores, class_ids = decode_end2end(np.zeros((1, 0, 6), dtype=np.float32))
    assert len(boxes) == len(scores) == len(class_ids) == 0


# --------------------------------------------------------------------------
# Anchor-style decoding, for contrast
# --------------------------------------------------------------------------


def test_anchor_head_decodes_centre_form_to_corners() -> None:
    # (1, 4 + 2 classes, 1 anchor): centre (100, 200), size 40x60.
    output = np.zeros((1, 6, 1), dtype=np.float32)
    output[0, :4, 0] = [100.0, 200.0, 40.0, 60.0]
    output[0, 4, 0] = 0.1
    output[0, 5, 0] = 0.7

    boxes, scores, class_ids = decode_yolo(output, num_classes=2)

    assert boxes[0].tolist() == [80.0, 170.0, 120.0, 230.0]
    assert scores[0] == pytest.approx(0.7)
    assert class_ids[0] == 1


# --------------------------------------------------------------------------
# The inverse both paths share
# --------------------------------------------------------------------------


def test_letterbox_inverse_returns_original_coordinates() -> None:
    """Whichever head produced them, boxes are unmapped the same way."""
    image = np.zeros((684, 1493, 3), dtype=np.uint8)
    padded, transform = letterbox(image, 640)

    assert padded.shape == (640, 640, 3)

    # A box covering the whole letterboxed content should map back to the
    # whole source image.
    content = np.array(
        [
            [
                transform.pad_x,
                transform.pad_y,
                640 - transform.pad_x,
                640 - transform.pad_y,
            ]
        ]
    )
    restored = transform.to_source(content)[0]

    assert restored[0] == pytest.approx(0.0, abs=1.0)
    assert restored[2] == pytest.approx(1493.0, abs=2.0)
    assert restored[3] == pytest.approx(684.0, abs=2.0)


def test_letterbox_inverse_clips_to_the_frame() -> None:
    transform = Letterbox(
        scale=0.5, pad_x=10.0, pad_y=20.0, source_width=100, source_height=50
    )
    restored = transform.to_source(np.array([[-500.0, -500.0, 5000.0, 5000.0]]))[0]
    assert restored.tolist() == [0.0, 0.0, 100.0, 50.0]
