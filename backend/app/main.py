"""VisionDetect REST API.

Routes are thin: validate, delegate, serialise. The detection work lives in
`pipeline.py`, the models in `detector.py`, the image handling in
`image_utils.py`.

Two decisions shape the surface:

* **The server starts even when no model can load.** `/api/health` then reports
  precisely why instead of the process failing to boot, which is the difference
  between a dashboard that says "weights not found at ..." and a connection
  error that tells the user nothing.
* **Detection routes are `def`, not `async def`.** Inference is seconds of
  blocking CPU work; on the event loop it would stall every other request
  including the health check. FastAPI runs sync routes in a thread pool, which
  is what this workload wants.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_DIR, get_settings
from app.detector import DetectorUnavailable, ModelRegistry
from app.exports import detections_csv, summary_report
from app.history import HistoryStore
from app.image_utils import (
    EncodingError,
    ImageError,
    annotate,
    decode_image,
    encode_image,
    media_type,
)
from app.pipeline import new_result_id, payload_without_image, run_detection
from app.results import ResultStore, StoredResult
from app.schemas import (
    DetectionResponse,
    DetectionSettings,
    HealthResponse,
    HistoryEntry,
    HistoryResponse,
    ModelsResponse,
    ModelStatus,
    RerunRequest,
)

logger = logging.getLogger("visionai")

settings = get_settings()
registry = ModelRegistry(settings)
results = ResultStore(
    ttl_seconds=settings.result_ttl_seconds,
    max_entries=settings.max_cached_results,
)
history = HistoryStore(settings.history_db, enabled=settings.history_enabled)

app = FastAPI(
    title=f"{settings.app_name} Object Detection API",
    version=settings.version,
    description=(
        "Upload an image, run a YOLO detection model over it, and retrieve "
        "bounding boxes, per-class statistics and annotated renderings."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(ImageError)
def _handle_image_error(_request: Any, exc: ImageError) -> JSONResponse:
    # 422: the request was well formed but the image itself is unusable.
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(EncodingError)
def _handle_encoding_error(_request: Any, exc: EncodingError) -> JSONResponse:
    # 500: the request and the image were both fine; this end failed.
    logger.error("Annotated image encoding failed: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(DetectorUnavailable)
def _handle_detector_unavailable(_request: Any, exc: DetectorUnavailable) -> JSONResponse:
    # 503: the service is up but cannot serve inference right now, and the
    # message says what would fix it.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# --------------------------------------------------------------------------
# Service information
# --------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Backend health, model readiness and inference device.

    Reports the default model's real state by loading it if it has not been
    loaded yet, so a green indicator means inference has actually been proven
    possible rather than merely configured.
    """
    model_ready = False
    model_status = "unknown"
    device = "cpu"
    providers: list[str] = []

    try:
        detector = registry.get(settings.default_model)
        model_ready = True
        model_status = "ready"
        device = detector.info.device
    except DetectorUnavailable as exc:
        model_status = str(exc)
    except Exception as exc:  # noqa: BLE001 - health must never raise
        model_status = f"{type(exc).__name__}: {exc}"

    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
    except ImportError:
        providers = []

    return HealthResponse(
        status="ok" if model_ready else "degraded",
        app=settings.app_name,
        version=settings.version,
        model_ready=model_ready,
        model_status=model_status,
        default_model=settings.default_model,
        device=device,
        providers=providers,
        loaded_models=registry.loaded(),
        history_enabled=settings.history_enabled,
        max_upload_mb=round(settings.max_upload_bytes / 1024 / 1024, 1),
        allowed_formats=list(settings.allowed_formats),
        defaults=DetectionSettings(
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            max_detections=settings.max_detections,
        ),
    )


@app.get("/api/models", response_model=ModelsResponse, tags=["service"])
def list_models() -> ModelsResponse:
    """Configured detection models and whether each can be served."""
    return ModelsResponse(
        models=[ModelStatus.model_validate(entry) for entry in registry.describe()],
        default_model=settings.default_model,
        loaded=registry.loaded(),
        max_loaded=settings.max_loaded_models,
    )


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@app.post("/api/detect", response_model=DetectionResponse, tags=["detection"])
def detect(
    file: Annotated[UploadFile, File(description="JPG, PNG or WEBP image")],
    confidence_threshold: Annotated[float | None, Form(ge=0.01, le=0.99)] = None,
    iou_threshold: Annotated[float | None, Form(ge=0.01, le=0.99)] = None,
    max_detections: Annotated[int | None, Form(ge=1, le=1000)] = None,
    model: Annotated[str | None, Form()] = None,
    show_labels: Annotated[bool, Form()] = True,
    show_confidence: Annotated[bool, Form()] = True,
    show_boxes: Annotated[bool, Form()] = True,
) -> DetectionResponse:
    """Run detection on an uploaded image."""
    raw = file.file.read()
    decoded = decode_image(
        raw,
        filename=file.filename or "upload",
        max_bytes=settings.max_upload_bytes,
        max_pixels=settings.max_image_pixels,
        allowed_formats=settings.allowed_formats,
    )

    detector = registry.get(model)
    result_id = new_result_id()

    response, detections, _ = run_detection(
        detector=detector,
        image=decoded.pixels,
        filename=file.filename or "upload",
        image_format=decoded.detected_format,
        size_bytes=decoded.size_bytes,
        confidence=_or_default(confidence_threshold, settings.confidence_threshold),
        iou=_or_default(iou_threshold, settings.iou_threshold),
        max_detections=_or_default(max_detections, settings.max_detections),
        result_id=result_id,
        show_boxes=show_boxes,
        show_labels=show_labels,
        show_confidence=show_confidence,
    )

    _store(response, detections, decoded.pixels)
    _persist_upload(result_id, decoded.detected_format, raw)
    return response


@app.post(
    "/api/results/{result_id}/rerun",
    response_model=DetectionResponse,
    tags=["detection"],
)
def rerun(result_id: str, request: Annotated[RerunRequest, Body()]) -> DetectionResponse:
    """Re-run detection on a held image with new settings.

    Lowering the confidence threshold cannot be done by filtering what was
    already returned -- the detections below the previous threshold were never
    sent -- so applying settings re-runs the model against the retained pixels.
    No re-upload, and the result keeps its ID so history stays one row.
    """
    stored = results.get(result_id)
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Result {result_id!r} is no longer held. Results are kept in "
                f"memory for {settings.result_ttl_seconds // 60} minutes; "
                "please upload the image again."
            ),
        )

    previous = stored.payload["settings"]
    detector = registry.get(request.model or stored.payload["model"])

    response, detections, _ = run_detection(
        detector=detector,
        image=stored.image,
        filename=stored.payload["image"]["filename"],
        image_format=stored.payload["image"]["format"],
        size_bytes=stored.payload["image"]["size_bytes"],
        confidence=_or_default(
            request.confidence_threshold, previous["confidence_threshold"]
        ),
        iou=_or_default(request.iou_threshold, previous["iou_threshold"]),
        max_detections=_or_default(
            request.max_detections, previous["max_detections"]
        ),
        result_id=result_id,
        show_boxes=request.show_boxes,
        show_labels=request.show_labels,
        show_confidence=request.show_confidence,
    )

    _store(response, detections, stored.image)
    return response


@app.get("/api/results/{result_id}", response_model=DetectionResponse, tags=["results"])
def get_result(result_id: str) -> DetectionResponse:
    """Re-fetch a held result without its inline image."""
    stored = _require(result_id)
    return DetectionResponse.model_validate(stored.payload)


@app.get("/api/results/{result_id}/image", tags=["results"])
def result_image(
    result_id: str,
    format: Annotated[str, Query(pattern="^(png|jpeg|jpg)$")] = "png",
    classes: Annotated[str | None, Query(description="Comma-separated class names")] = None,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    labels: bool = True,
    scores: bool = True,
    boxes: bool = True,
    download: bool = False,
) -> Response:
    """The annotated image, re-rendered to match the view being exported.

    `classes` and `min_confidence` mirror the filters the dashboard applies on
    the client, so an image downloaded while the view shows only `person` above
    60% contains exactly those boxes. Without them an export would quietly
    disagree with what was on screen when the button was pressed.
    """
    stored = _require(result_id)

    wanted = (
        {name.strip() for name in classes.split(",") if name.strip()}
        if classes is not None
        else None
    )
    selected = [
        detection
        for detection in stored.detections
        if (wanted is None or detection.class_name in wanted)
        and detection.confidence >= min_confidence
    ]

    rendered = annotate(
        stored.image,
        selected,
        show_boxes=boxes,
        show_labels=labels,
        show_confidence=scores,
    )
    payload = encode_image(rendered, format)

    headers = {"Cache-Control": "no-store"}
    if download:
        stem = _stem(stored.filename)
        extension = "jpg" if format in {"jpg", "jpeg"} else "png"
        headers["Content-Disposition"] = (
            f'attachment; filename="{stem}_annotated.{extension}"'
        )

    return Response(content=payload, media_type=media_type(format), headers=headers)


@app.get("/api/results/{result_id}/json", tags=["results"])
def result_json(result_id: str, download: bool = True) -> Response:
    """Structured detection results as a JSON file."""
    stored = _require(result_id)
    import json

    body = json.dumps(stored.payload, indent=2)
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = (
            f'attachment; filename="{_stem(stored.filename)}_detections.json"'
        )
    return Response(content=body, media_type="application/json", headers=headers)


@app.get("/api/results/{result_id}/csv", tags=["results"])
def result_csv(result_id: str) -> Response:
    """The detection table as CSV."""
    stored = _require(result_id)
    return Response(
        content=detections_csv(stored.payload),
        media_type="text/csv",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": (
                f'attachment; filename="{_stem(stored.filename)}_detections.csv"'
            ),
        },
    )


@app.get("/api/results/{result_id}/report", response_class=PlainTextResponse, tags=["results"])
def result_report(result_id: str) -> Response:
    """A readable summary report: image, model, settings, counts, statistics."""
    stored = _require(result_id)
    return Response(
        content=summary_report(stored.payload),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": (
                f'attachment; filename="{_stem(stored.filename)}_report.txt"'
            ),
        },
    )


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


@app.get("/api/history", response_model=HistoryResponse, tags=["history"])
def list_history(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> HistoryResponse:
    """Past analyses -- metadata only; images are not retained."""
    rows, total = history.list(limit=limit, offset=offset)
    live = results.ids()
    return HistoryResponse(
        entries=[
            HistoryEntry(**row, result_available=row["analysis_id"] in live)
            for row in rows
        ],
        total=total,
    )


@app.delete("/api/history/{analysis_id}", tags=["history"])
def delete_history(analysis_id: str) -> dict[str, str]:
    if not history.delete(analysis_id):
        raise HTTPException(status_code=404, detail=f"No history entry {analysis_id!r}.")
    results.discard(analysis_id)
    return {"detail": f"Deleted {analysis_id}."}


@app.delete("/api/history", tags=["history"])
def clear_history() -> dict[str, str]:
    removed = history.clear()
    results.clear()
    return {"detail": f"Cleared {removed} history entries."}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _or_default(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _require(result_id: str) -> StoredResult:
    stored = results.get(result_id)
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Result {result_id!r} is no longer held. Results are kept in "
                f"memory for {settings.result_ttl_seconds // 60} minutes."
            ),
        )
    return stored


def _stem(filename: str) -> str:
    """A filesystem-safe basename for generated downloads."""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
    return safe[:60] or "image"


def _store(
    response: DetectionResponse, detections: list[Any], image: Any
) -> None:
    """Cache the result and log it to history."""
    payload = payload_without_image(response)
    results.put(
        StoredResult(
            result_id=response.result_id,
            payload=payload,
            detections=detections,
            image=image,
            filename=response.image.filename,
        )
    )

    top = ", ".join(
        f"{entry.class_name} x{entry.count}" for entry in response.classes[:5]
    )
    try:
        history.record(
            analysis_id=response.result_id,
            filename=response.image.filename,
            total_objects=response.summary.total_objects,
            unique_classes=response.summary.unique_classes,
            processing_time_ms=response.processing_time_ms,
            model=response.model,
            width=response.image.width,
            height=response.image.height,
            confidence_threshold=response.settings.confidence_threshold,
            iou_threshold=response.settings.iou_threshold,
            top_classes=top,
        )
    except Exception:  # noqa: BLE001
        # History is a convenience. A locked or unwritable database must not
        # cost the user the detection they just waited for.
        logger.exception("Could not write history for %s", response.result_id)


def _persist_upload(result_id: str, image_format: str, data: bytes) -> None:
    """Write the original upload to disk, only when explicitly configured."""
    if not settings.persist_uploads:
        return
    try:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        extension = "jpg" if image_format == "jpeg" else image_format
        (settings.upload_dir / f"{result_id}.{extension}").write_bytes(data)
    except OSError:
        logger.exception("Could not persist upload %s", result_id)


# --------------------------------------------------------------------------
# Built frontend
# --------------------------------------------------------------------------
#
# When `frontend/dist` exists, the API also serves the dashboard, so the whole
# application runs as one process on one port with no proxy and no CORS. The
# Vite dev server remains the better experience while editing the UI -- it has
# hot reload -- but it is a second Node process, and on a memory-constrained
# machine this single-process mode is the one that will actually start.
#
# Mounted last so that every /api route above takes precedence; the catch-all
# returns index.html so client-side navigation works on a refresh.

FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"


@app.get("/", include_in_schema=False)
def root() -> Any:
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "app": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/health",
        "note": (
            "The dashboard is not built. Run `npm run build` in "
            "frontend, or use the Vite dev server on port 5173."
        ),
    }


if FRONTEND_DIST.is_dir():
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{asset_path:path}", include_in_schema=False)
    def frontend_files(asset_path: str) -> Any:
        """Serve a built file, falling back to the SPA entry point."""
        if asset_path.startswith("api/"):
            raise HTTPException(status_code=404, detail=f"No such endpoint: /{asset_path}")

        candidate = (FRONTEND_DIST / asset_path).resolve()
        # Containment check: a crafted path must not escape the dist directory.
        if (
            asset_path
            and candidate.is_file()
            and candidate.is_relative_to(FRONTEND_DIST.resolve())
        ):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
