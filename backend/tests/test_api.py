"""End-to-end tests against the real model.

These are not mocked. Every assertion below is checked against what YOLO11n
actually predicts, which is the only way to catch the failure this system is
most exposed to: boxes that are plausible but in the wrong place.

`bus.jpg` and `zidane.jpg` ship inside the installed `ultralytics` package and
are the canonical YOLO sample images, with published, stable results -- 1 bus
and 4 people, 2 people and a tie. Asserting against those numbers verifies head
decoding, non-maximum suppression and the letterbox inverse together.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLES = Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets"
BUS = SAMPLES / "bus.jpg"
ZIDANE = SAMPLES / "zidane.jpg"

needs_samples = pytest.mark.skipif(
    not BUS.exists(), reason=f"Ultralytics sample images not found at {SAMPLES}"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def bus_result(client: TestClient) -> dict:
    """One detection run on bus.jpg, shared by the tests that inspect it."""
    with BUS.open("rb") as handle:
        response = client.post(
            "/api/detect",
            files={"file": ("bus.jpg", handle, "image/jpeg")},
            data={"confidence_threshold": "0.35", "iou_threshold": "0.45"},
        )
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------
# Service information
# --------------------------------------------------------------------------


def test_health_reports_a_loadable_model(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok", body["model_status"]
    assert body["model_ready"] is True
    assert body["device"] in {"cpu", "cuda"}
    assert body["defaults"]["confidence_threshold"] == pytest.approx(0.35)
    assert body["defaults"]["iou_threshold"] == pytest.approx(0.45)


def test_models_endpoint_lists_the_registry(client: TestClient) -> None:
    body = client.get("/api/models").json()
    names = {entry["name"] for entry in body["models"]}
    assert "yolo11n" in names
    assert body["default_model"] == "yolo11n"

    default = next(e for e in body["models"] if e["name"] == "yolo11n")
    assert default["status"] in {"loaded", "available"}
    assert default["is_default"] is True


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@needs_samples
def test_bus_detections_match_published_results(bus_result: dict) -> None:
    """The canonical result for this image: one bus and four people."""
    assert bus_result["summary"]["class_counts"] == {"person": 4, "bus": 1}
    assert bus_result["summary"]["total_objects"] == 5
    assert bus_result["summary"]["unique_classes"] == 2


@needs_samples
def test_summary_is_consistent_with_the_detection_list(bus_result: dict) -> None:
    """Every headline figure must be derivable from the detections themselves."""
    detections = bus_result["detections"]
    summary = bus_result["summary"]
    confidences = [d["confidence"] for d in detections]

    assert summary["total_objects"] == len(detections)
    assert summary["unique_classes"] == len({d["class_name"] for d in detections})
    assert summary["max_confidence"] == pytest.approx(max(confidences), abs=1e-4)
    assert summary["min_confidence"] == pytest.approx(min(confidences), abs=1e-4)
    assert summary["avg_confidence"] == pytest.approx(
        sum(confidences) / len(confidences), abs=1e-4
    )

    counted = sum(summary["class_counts"].values())
    assert counted == len(detections)

    # The per-class rollup drives the charts and must agree with the same source.
    for entry in bus_result["classes"]:
        members = [d for d in detections if d["class_name"] == entry["class_name"]]
        assert entry["count"] == len(members)
        assert entry["max_confidence"] == pytest.approx(
            max(d["confidence"] for d in members), abs=1e-4
        )


@needs_samples
def test_boxes_lie_within_the_original_image(bus_result: dict) -> None:
    """Coordinates must be in uploaded-image space, not network space.

    A letterbox inverse that is wrong still yields boxes inside 640x640, so the
    test checks the reported dimensions and that at least one box extends beyond
    the network input -- which only holds if the unmapping really happened.
    """
    width = bus_result["image"]["width"]
    height = bus_result["image"]["height"]
    assert (width, height) == (810, 1080)

    for detection in bus_result["detections"]:
        bbox = detection["bbox"]
        assert 0 <= bbox["x_min"] < bbox["x_max"] <= width
        assert 0 <= bbox["y_min"] < bbox["y_max"] <= height
        assert detection["width"] == bbox["x_max"] - bbox["x_min"]
        assert detection["height"] == bbox["y_max"] - bbox["y_min"]
        assert detection["center"]["x"] == pytest.approx(
            (bbox["x_min"] + bbox["x_max"]) / 2
        )

    assert max(d["bbox"]["y_max"] for d in bus_result["detections"]) > 640


@needs_samples
def test_the_bus_is_found_where_the_bus_is(bus_result: dict) -> None:
    """A spot check on absolute position, not just internal consistency."""
    bus = next(d for d in bus_result["detections"] if d["class_name"] == "bus")
    bbox = bus["bbox"]

    # The bus fills most of this 810x1080 frame, from near the left edge to near
    # the right, and its top is in the upper third.
    assert bus["confidence"] > 0.85
    assert bbox["x_min"] < 60 and bbox["x_max"] > 750
    assert 180 < bbox["y_min"] < 280
    assert bus["width"] * bus["height"] > 0.3 * 810 * 1080


@needs_samples
def test_object_ids_are_ordered_by_confidence(bus_result: dict) -> None:
    ids = [d["object_id"] for d in bus_result["detections"]]
    assert ids == list(range(1, len(ids) + 1))

    confidences = [d["confidence"] for d in bus_result["detections"]]
    assert confidences == sorted(confidences, reverse=True)


@needs_samples
def test_response_carries_an_annotated_image(bus_result: dict) -> None:
    assert bus_result["annotated_image"].startswith("data:image/png;base64,")
    assert len(bus_result["annotated_image"]) > 5000


@needs_samples
def test_a_higher_threshold_finds_fewer_objects(client: TestClient) -> None:
    """Thresholds must actually reach the model rather than being decorative."""
    counts = {}
    for threshold in ("0.30", "0.90"):
        with BUS.open("rb") as handle:
            response = client.post(
                "/api/detect",
                files={"file": ("bus.jpg", handle, "image/jpeg")},
                data={"confidence_threshold": threshold},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        counts[threshold] = body["summary"]["total_objects"]
        assert body["settings"]["confidence_threshold"] == pytest.approx(
            float(threshold)
        )
        for detection in body["detections"]:
            assert detection["confidence"] >= float(threshold)

    assert counts["0.90"] < counts["0.30"]


@needs_samples
def test_rerun_applies_new_settings_to_the_held_image(client: TestClient) -> None:
    """Re-applying settings must not need the image uploaded again.

    Uses its own upload rather than the shared fixture: a re-run deliberately
    overwrites the stored result and its history row in place, so pointing it at
    the fixture would invalidate every later test that reads the same result.
    """
    with BUS.open("rb") as handle:
        first = client.post(
            "/api/detect",
            files={"file": ("bus.jpg", handle, "image/jpeg")},
            data={"confidence_threshold": "0.35"},
        ).json()

    response = client.post(
        f"/api/results/{first['result_id']}/rerun",
        json={"confidence_threshold": 0.9},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["result_id"] == first["result_id"]
    assert body["settings"]["confidence_threshold"] == pytest.approx(0.9)
    assert body["summary"]["total_objects"] < first["summary"]["total_objects"]
    assert body["image"]["filename"] == "bus.jpg"

    # The overwrite must be visible on re-fetch, not just in this response.
    refetched = client.get(f"/api/results/{first['result_id']}").json()
    assert refetched["summary"]["total_objects"] == body["summary"]["total_objects"]


@needs_samples
def test_zidane_image_detects_a_different_class_mix(client: TestClient) -> None:
    with ZIDANE.open("rb") as handle:
        response = client.post(
            "/api/detect",
            files={"file": ("zidane.jpg", handle, "image/jpeg")},
            data={"confidence_threshold": "0.35"},
        )
    body = response.json()
    assert body["summary"]["class_counts"] == {"person": 2, "tie": 1}
    assert body["image"]["width"] == 1280


def test_an_image_with_no_objects_is_a_result_not_an_error(
    client: TestClient, tmp_path: Path
) -> None:
    """Zero detections must round-trip cleanly through every statistic."""
    import cv2
    import numpy as np

    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), blank)

    with path.open("rb") as handle:
        response = client.post(
            "/api/detect", files={"file": ("blank.png", handle, "image/png")}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["summary"]["total_objects"] == 0
    assert body["summary"]["unique_classes"] == 0
    assert body["summary"]["avg_confidence"] == 0.0
    assert body["detections"] == []
    assert body["classes"] == []
    assert body["annotated_image"].startswith("data:image/png;base64,")


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------


@needs_samples
def test_json_export_contains_the_real_predictions(
    client: TestClient, bus_result: dict
) -> None:
    response = client.get(f"/api/results/{bus_result['result_id']}/json")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]

    exported = json.loads(response.text)
    assert exported["summary"] == bus_result["summary"]
    assert len(exported["detections"]) == len(bus_result["detections"])
    assert exported["detections"][0]["bbox"] == bus_result["detections"][0]["bbox"]
    assert exported["image"]["filename"] == "bus.jpg"
    assert exported["model"] == "yolo11n"


@needs_samples
def test_csv_export_matches_the_detection_table(
    client: TestClient, bus_result: dict
) -> None:
    response = client.get(f"/api/results/{bus_result['result_id']}/csv")
    assert response.status_code == 200

    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == bus_result["summary"]["total_objects"]

    for row, detection in zip(rows, bus_result["detections"]):
        assert row["class_name"] == detection["class_name"]
        assert int(row["object_id"]) == detection["object_id"]
        assert float(row["confidence"]) == pytest.approx(
            detection["confidence"], abs=1e-4
        )
        assert int(row["x_min"]) == detection["bbox"]["x_min"]
        assert int(row["width"]) == detection["width"]


@needs_samples
def test_report_export_states_the_counts(client: TestClient, bus_result: dict) -> None:
    response = client.get(f"/api/results/{bus_result['result_id']}/report")
    assert response.status_code == 200
    text = response.text

    assert "VisionDetect - Object Detection Report" in text
    assert "bus.jpg" in text
    assert f"Total objects    : {bus_result['summary']['total_objects']}" in text
    assert "person" in text


@needs_samples
def test_annotated_image_download(client: TestClient, bus_result: dict) -> None:
    response = client.get(
        f"/api/results/{bus_result['result_id']}/image",
        params={"format": "png", "download": "true"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
    assert "bus_annotated.png" in response.headers["content-disposition"]


@needs_samples
def test_class_filter_changes_the_rendered_image(
    client: TestClient, bus_result: dict
) -> None:
    """Filtering must reach the renderer, not just the table."""
    everything = client.get(f"/api/results/{bus_result['result_id']}/image")
    only_bus = client.get(
        f"/api/results/{bus_result['result_id']}/image", params={"classes": "bus"}
    )
    nothing = client.get(
        f"/api/results/{bus_result['result_id']}/image",
        params={"classes": "bus", "boxes": "false", "labels": "false"},
    )

    assert everything.content != only_bus.content
    assert only_bus.content != nothing.content


@needs_samples
def test_jpeg_output_is_also_available(client: TestClient, bus_result: dict) -> None:
    response = client.get(
        f"/api/results/{bus_result['result_id']}/image", params={"format": "jpeg"}
    )
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8\xff")


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


@needs_samples
def test_history_records_the_analysis(client: TestClient, bus_result: dict) -> None:
    body = client.get("/api/history").json()
    entry = next(
        (e for e in body["entries"] if e["analysis_id"] == bus_result["result_id"]),
        None,
    )
    assert entry is not None, "the run was not written to history"
    assert entry["filename"] == "bus.jpg"
    assert entry["total_objects"] == bus_result["summary"]["total_objects"]
    assert entry["model"] == "yolo11n"
    assert "person" in entry["top_classes"]


# --------------------------------------------------------------------------
# Validation and error handling
# --------------------------------------------------------------------------


def test_a_text_file_renamed_as_png_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/detect",
        files={"file": ("notanimage.png", b"this is plain text", "image/png")},
    )
    assert response.status_code == 422
    assert "not a readable" in response.json()["detail"]


def test_an_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post("/api/detect", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_a_truncated_image_is_rejected(client: TestClient) -> None:
    """Valid magic bytes, unusable content: the decoder must be the judge."""
    response = client.post(
        "/api/detect",
        files={"file": ("broken.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "image/png")},
    )
    assert response.status_code == 422
    assert "could not be decoded" in response.json()["detail"]


def test_an_oversized_upload_is_rejected(client: TestClient) -> None:
    from app.config import get_settings

    limit = get_settings().max_upload_bytes
    payload = b"\xff\xd8\xff" + b"\x00" * limit
    response = client.post(
        "/api/detect", files={"file": ("huge.jpg", payload, "image/jpeg")}
    )
    assert response.status_code == 422
    assert "exceeds" in response.json()["detail"]


def test_an_unknown_model_is_reported_clearly(client: TestClient, tmp_path: Path) -> None:
    import cv2
    import numpy as np

    path = tmp_path / "small.png"
    cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))

    with path.open("rb") as handle:
        response = client.post(
            "/api/detect",
            files={"file": ("small.png", handle, "image/png")},
            data={"model": "does-not-exist"},
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "does-not-exist" in detail
    assert "yolo11n" in detail


def test_a_missing_result_gives_404(client: TestClient) -> None:
    response = client.get("/api/results/det_nothinghere/json")
    assert response.status_code == 404
    assert "no longer held" in response.json()["detail"]


def test_an_out_of_range_threshold_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/detect",
        files={"file": ("x.jpg", b"\xff\xd8\xff\x00", "image/jpeg")},
        data={"confidence_threshold": "5.0"},
    )
    assert response.status_code == 422
