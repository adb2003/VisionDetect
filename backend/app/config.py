"""Runtime configuration for the VisionDetect detection service.

Everything a deployment might reasonably want to change -- which weights to
serve, where they live, how large an upload may be, how long a result stays
retrievable -- is settled here and nowhere else, so swapping a model is an
environment change rather than a code change.

Model definitions are deliberately data, not Python. A `ModelSpec` carries only
what cannot be read back from the artefact itself: the registry key, the backend
that can execute it, and the path. Class names, network input size and task are
read from the model's own metadata at load time (see `detector.py`), which is
what lets a custom-trained checkpoint drop in without anyone editing this file.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

# app/ -> backend/ -> repository root (backend/ and frontend/ are siblings at the root)
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
REPO_ROOT = PROJECT_DIR


class ModelSpec(BaseModel):
    """One selectable detection model."""

    name: str
    """Registry key, used as the `model` field in requests and responses."""

    backend: str
    """onnx or ultralytics."""

    weights: str
    """Path to the checkpoint. Relative paths resolve against the weights
    directory first, then the repository root. A bare Ultralytics name such as
    yolo11n.pt is left alone so the library can fetch it on first use."""

    label: str
    """Human-readable name for the model picker."""

    description: str = ""

    prompts: list[str] = Field(default_factory=list)
    """Text vocabulary for open-vocabulary backends. Empty means the model's own
    closed class set is used."""

    label_map: dict[str, str] = Field(default_factory=dict)
    """Prompt -> display class, for collapsing synonyms.

    An open-vocabulary model is asked in several phrasings for the same object,
    because which phrasing scores depends on the model rather than on English --
    "unfinished ceiling" beats "exposed concrete soffit" by a wide margin on the
    same photograph. Those are not separate classes and must not be counted as
    such, so every phrasing maps onto one display class. A prompt mapped to the
    empty string is scored but never reported, which is how a prompt earns its
    place purely as context for the others."""

    class_thresholds: dict[str, float] = Field(default_factory=dict)
    """Display class -> minimum confidence, applied on top of the request's.

    Raises the bar for classes measured to be noisy; it never lowers it, so the
    confidence control in the dashboard stays a floor the user can rely on."""

    recommended_confidence: float | None = None
    """Threshold this model is worth running at, surfaced in the UI.

    Open-vocabulary scores sit far lower than a closed-set model's -- a correct
    detection at 0.20 is normal -- so the default of 0.35 would hide almost
    everything this vocabulary can find."""


# Shipped registry. The ONNX entry is the default serving path: it is the
# fastest of the three on CPU and does not load PyTorch, so the process stays
# small. The Ultralytics entries exist for .pt checkpoints, for anything that
# needs auto-download, and for open-vocabulary prompting.
DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(
        name="yolo11n",
        backend="onnx",
        weights="yolo11n.onnx",
        label="YOLO11n (COCO, ONNX)",
        description=(
            "Nano detection model over the 80-class COCO vocabulary, served "
            "through onnxruntime. Fastest option on CPU."
        ),
    ),
    ModelSpec(
        name="yolo11n-seg",
        backend="onnx",
        weights="yolo11n-seg.onnx",
        label="YOLO11n-seg (COCO, ONNX)",
        description=(
            "Segmentation checkpoint served for its detection head; boxes and "
            "classes only, mask coefficients are ignored."
        ),
    ),
    ModelSpec(
        name="yolo11n-pt",
        backend="ultralytics",
        weights="yolo11n.pt",
        label="YOLO11n (COCO, PyTorch)",
        description=(
            "The same model run through the Ultralytics API. Downloads itself "
            "if the checkpoint is absent."
        ),
    ),
]


class Settings(BaseModel):
    """Service settings, overridable through VISIONAI_* environment variables."""

    app_name: str = "VisionDetect"
    version: str = "1.0.0"

    weights_dir: Path = REPO_ROOT / "models"
    """Where ModelSpec.weights is resolved from. Defaults to the repository's
    existing model directory so a checkout that already has weights needs no
    download."""

    default_model: str = "yolo11n"

    max_loaded_models: int = 2
    """How many models may stay resident at once.

    Not a tidiness limit. A PyTorch checkpoint and a couple of onnxruntime
    sessions together run to gigabytes, and on a 16 GB laptop with a browser
    open the third one fails inside the allocator rather than at load time --
    which surfaces as a baffling mid-inference error. Holding the two most
    recently used models keeps the common single-model case free of reloads
    while bounding the footprint."""

    device: str = "auto"
    """auto | cpu | cuda. auto uses the GPU when the installed runtime actually
    offers one, and reports which it chose through /api/health."""

    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 300

    max_upload_bytes: int = 15 * 1024 * 1024
    max_image_pixels: int = 50_000_000
    """Guards against decompression bombs: a 50 MP ceiling is far above any
    phone or site camera while keeping worst-case memory bounded."""

    allowed_formats: tuple[str, ...] = ("jpg", "jpeg", "png", "webp")

    result_ttl_seconds: int = 3600
    max_cached_results: int = 50
    """Results live in memory only. The original image is kept for the TTL so
    settings can be re-applied without a re-upload, and is never written to disk
    unless persist_uploads is set."""

    persist_uploads: bool = False
    upload_dir: Path = BACKEND_DIR / "data" / "uploads"

    history_enabled: bool = True
    history_db: Path = BACKEND_DIR / "data" / "history.db"

    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    )

    models: list[ModelSpec] = Field(default_factory=lambda: list(DEFAULT_MODELS))

    def model_spec(self, name: str) -> ModelSpec | None:
        for spec in self.models:
            if spec.name == name:
                return spec
        return None

    def resolve_weights(self, weights: str) -> Path:
        """Locate a checkpoint: weights directory, then repository root.

        Returns the most likely path even when nothing exists, so the caller can
        report a precise "not found at ..." rather than a bare failure.
        """
        candidate = Path(weights)
        if candidate.is_absolute():
            return candidate
        in_weights_dir = self.weights_dir / candidate
        if in_weights_dir.exists():
            return in_weights_dir
        from_root = REPO_ROOT / candidate
        if from_root.exists():
            return from_root
        return in_weights_dir


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings from the environment once per process."""
    settings = Settings()

    if value := os.environ.get("VISIONAI_WEIGHTS_DIR"):
        settings.weights_dir = Path(value).expanduser()
    if value := os.environ.get("VISIONAI_DEFAULT_MODEL"):
        settings.default_model = value
    if value := os.environ.get("VISIONAI_DEVICE"):
        settings.device = value.strip().lower()
    if value := os.environ.get("VISIONAI_CONFIDENCE_THRESHOLD"):
        settings.confidence_threshold = float(value)
    if value := os.environ.get("VISIONAI_IOU_THRESHOLD"):
        settings.iou_threshold = float(value)
    if value := os.environ.get("VISIONAI_MAX_UPLOAD_MB"):
        settings.max_upload_bytes = int(float(value) * 1024 * 1024)
    if value := os.environ.get("VISIONAI_RESULT_TTL_SECONDS"):
        settings.result_ttl_seconds = int(value)
    if value := os.environ.get("VISIONAI_HISTORY_DB"):
        settings.history_db = Path(value).expanduser()
    if value := os.environ.get("VISIONAI_CORS_ORIGINS"):
        settings.cors_origins = tuple(
            origin.strip() for origin in value.split(",") if origin.strip()
        )

    settings.persist_uploads = _env_bool(
        "VISIONAI_PERSIST_UPLOADS", settings.persist_uploads
    )
    settings.history_enabled = _env_bool(
        "VISIONAI_HISTORY_ENABLED", settings.history_enabled
    )

    # The registry is data. `models.construction.json` ships beside this package
    # and carries the construction vocabulary, which is far too long to hold in
    # Python; VISIONAI_MODELS_FILE replaces it wholesale, which is how a
    # custom-trained model is added without touching this code.
    #
    # Listing a model costs nothing - `ModelRegistry.describe` reports
    # availability from the weights path and never loads anything - so the
    # open-vocabulary entries can be visible in the picker even on a machine
    # that cannot run them. Selecting one there reports why.
    registry_file = os.environ.get("VISIONAI_MODELS_FILE") or (
        str(shipped) if (shipped := BACKEND_DIR / "models.construction.json").exists() else ""
    )
    if registry_file:
        path = Path(registry_file).expanduser()
        data = json.loads(path.read_text(encoding="utf-8"))
        settings.models = [ModelSpec.model_validate(entry) for entry in data]

    if settings.model_spec(settings.default_model) is None and settings.models:
        settings.default_model = settings.models[0].name

    return settings
