"""Request and response models.

Field names are snake_case throughout and the detection payload follows the
structure specified for the exported JSON, so what `/api/detect` returns, what
`/api/results/{id}/json` writes to a file, and what the dashboard renders are
all the same document rather than three dialects of it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned box in the original image's pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Point(BaseModel):
    x: float
    y: float


class DetectionOut(BaseModel):
    """One detected object."""

    object_id: int
    """1-based, assigned in descending confidence order so object 1 is always
    the most certain detection in the image."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    width: int
    height: int
    center: Point
    area: int
    colour: str = Field(alias="color")
    """Hex colour for this class, so the browser overlay and the server-drawn
    annotation agree without duplicating the palette."""

    model_config = {"populate_by_name": True}


class ClassSummary(BaseModel):
    """Per-class rollup, used by the distribution charts and the filter list."""

    class_id: int
    class_name: str
    count: int
    colour: str = Field(alias="color")
    avg_confidence: float
    max_confidence: float

    model_config = {"populate_by_name": True}


class ImageMeta(BaseModel):
    filename: str
    width: int
    height: int
    format: str
    size_bytes: int


class DetectionSummary(BaseModel):
    total_objects: int
    unique_classes: int
    max_confidence: float
    min_confidence: float
    avg_confidence: float
    class_counts: dict[str, int]


class DetectionSettings(BaseModel):
    confidence_threshold: float
    iou_threshold: float
    max_detections: int


class ModelMeta(BaseModel):
    """What actually ran, as opposed to what was asked for."""

    name: str
    label: str
    backend: str
    task: str
    device: str
    input_size: int
    class_count: int


class DetectionResponse(BaseModel):
    """The complete result of one detection run."""

    result_id: str
    created_at: str
    image: ImageMeta
    model: str
    """Model registry key. A plain string so the exported JSON matches the
    documented export shape; `model_info` carries the detail."""

    model_info: ModelMeta
    settings: DetectionSettings
    processing_time_ms: float
    """End to end on the server: decode, inference, annotation, serialisation."""

    inference_time_ms: float
    """The model's forward pass alone."""

    summary: DetectionSummary
    classes: list[ClassSummary]
    detections: list[DetectionOut]
    annotated_image: str | None = None
    """Data URL of the annotated image, so the first render needs no second
    round trip. Omitted from history and re-fetch responses."""

    annotated_image_url: str

    model_config = {"protected_namespaces": ()}


class RerunRequest(BaseModel):
    """Re-run detection on an image already held for this result."""

    confidence_threshold: float | None = Field(default=None, ge=0.01, le=0.99)
    iou_threshold: float | None = Field(default=None, ge=0.01, le=0.99)
    max_detections: int | None = Field(default=None, ge=1, le=1000)
    model: str | None = None
    show_labels: bool = True
    show_confidence: bool = True
    show_boxes: bool = True

    model_config = {"protected_namespaces": ()}


class ModelStatus(BaseModel):
    name: str
    label: str
    description: str
    backend: str
    task: str | None
    device: str | None
    input_size: int | None
    status: str
    """loaded | available | missing | error."""

    detail: str
    class_count: int | None
    classes: list[str]
    weights_path: str
    is_default: bool
    recommended_confidence: float | None = None
    """Threshold this model is worth running at.

    Open-vocabulary scores sit far below a closed-set model's, so the dashboard
    adopts this when the model is selected rather than leaving the user at a
    default that would hide almost everything."""


class ModelsResponse(BaseModel):
    models: list[ModelStatus]
    default_model: str
    loaded: list[str]
    max_loaded: int


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    model_ready: bool
    """Whether the default model has been loaded and can serve a request."""

    model_status: str
    default_model: str
    device: str
    providers: list[str]
    loaded_models: list[str]
    history_enabled: bool
    max_upload_mb: float
    allowed_formats: list[str]
    defaults: DetectionSettings

    model_config = {"protected_namespaces": ()}


class HistoryEntry(BaseModel):
    analysis_id: str
    filename: str
    timestamp: str
    total_objects: int
    unique_classes: int
    processing_time_ms: float
    model: str
    width: int
    height: int
    confidence_threshold: float
    iou_threshold: float
    top_classes: str
    """Comma-separated "class x count", enough to recognise a past run without
    retaining the image."""

    result_available: bool
    """Whether the in-memory result is still retrievable, as opposed to being
    history metadata alone."""

    model_config = {"protected_namespaces": ()}


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]
    total: int


class ErrorResponse(BaseModel):
    detail: str
