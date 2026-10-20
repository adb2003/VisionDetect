"""Object detection inference.

Two backends sit behind one `Detector` protocol: `onnxruntime` for `.onnx`
artefacts and Ultralytics for `.pt` checkpoints. Everything above this module --
the API, the exports, the dashboard -- depends only on the protocol and on
`RawDetection`, so a construction-trained model, an open-vocabulary model or an
entirely different architecture can be substituted by adding a `ModelSpec`.

Two things here are worth stating plainly because getting either wrong produces
confident nonsense rather than an error:

* **Coordinates.** The network sees a 640x640 letterboxed square. Every box it
  emits is in that space and must be unmapped through the exact scale and
  padding that produced it. `Letterbox.to_source` owns that inverse so no
  backend re-derives it.
* **Class vocabulary.** Names are read from the artefact's own metadata, never
  hardcoded. A checkpoint trained on helmets and cable trays therefore reports
  helmets and cable trays with no code change.

Models are loaded once and cached per process. Loading is lazy and never
happens at import, so the service starts instantly and can report *why* a model
is unusable instead of failing to boot.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from app.config import ModelSpec, Settings


class DetectorUnavailable(RuntimeError):
    """A model cannot be served, with a reason fit to show a user."""


@dataclass(frozen=True)
class RawDetection:
    """One detected instance, in original-image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass
class ModelInfo:
    """What a loaded model turned out to be, read from its own metadata."""

    name: str
    label: str
    backend: str
    task: str
    device: str
    input_size: int
    class_names: list[str]
    weights_path: str
    description: str = ""


# --------------------------------------------------------------------------
# Class colours
# --------------------------------------------------------------------------
#
# The server-rendered annotation and the interactive overlay in the browser must
# agree on colour, or the same object appears in two colours depending on which
# view you are looking at. So the colour is derived from the class *index* here
# and sent to the client with every response. Deriving it from the index rather
# than from position in the results also means a colour belongs to a class
# permanently: filtering one class out never repaints the others.
#
# Colours are generated in OKLCH at a fixed lightness, not in HSV. In HSV at
# constant saturation and value, perceived lightness swings wildly with hue --
# yellow comes out near-white and blue near-black -- which produced a palette
# whose yellows and greens were invisible as chart fills on a white panel. OKLCH
# is perceptually uniform, so one lightness gives every hue the same weight and
# the same contrast against the background.
#
# A caveat worth stating plainly, because it is a property of the problem and
# not of this implementation: eighty classes cannot all be mutually
# distinguishable. The palette below was tuned against a colour validator and
# passes on lightness, chroma and normal-vision separation, but a categorical
# scale this large cannot also be safe for red-green colour blindness -- some
# green and pink pair will always be confusable. Colour is therefore never the
# only carrier of identity anywhere in this product: every box, bar, legend
# entry and table row is labelled with its class name in text.

_HUE_STEP = 160.0
"""Degrees of hue between consecutive class indices.

Not the golden angle, which is the usual choice. The golden angle spreads a
*prefix* well but puts classes five apart only 32 degrees apart, and in COCO
that pair is `person` and `bus` -- so the commonest street scene came out with
pink people inside a pink bus. A 160 degree step guarantees at least 40 degrees
between any two classes within eight indices of each other, which is the range
that matters because COCO numbers related classes together: vehicles 0-8,
furniture 56-61, electronics 62-68.

Classes exactly nine apart land on the same hue, which the lightness band below
separates. Classes eighteen apart repeat exactly; with eighty classes and one
hue circle some pair must, and these are the least likely to share a photograph.
"""

_PALETTE_BANDS = (0.55, 0.67)
_PALETTE_CHROMA = 0.20
_LIGHTNESS_RANGE = (0.48, 0.70)
_MIN_USEFUL_CHROMA = 0.115


def _oklch_to_srgb(lightness: float, chroma: float, hue_degrees: float) -> tuple[float, float, float]:
    """Convert OKLCH to linear-encoded sRGB in 0..1 (may fall outside gamut)."""
    hue = math.radians(hue_degrees)
    a = chroma * math.cos(hue)
    b = chroma * math.sin(hue)

    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    long, medium, short = l_**3, m_**3, s_**3

    red = 4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short
    green = -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short
    blue = -0.0041960863 * long - 0.7034186147 * medium + 1.7076147010 * short

    def encode(value: float) -> float:
        if value <= 0.0031308:
            return 12.92 * value
        return 1.055 * (abs(value) ** (1 / 2.4)) - 0.055 if value > 0 else 0.0

    return encode(red), encode(green), encode(blue)


def _max_chroma(lightness: float, hue: float) -> float:
    """Greatest chroma this hue can carry at this lightness inside sRGB.

    Found by bisection rather than by clipping the RGB channels afterwards,
    because clipping shifts both hue and lightness and would defeat the point of
    working in OKLCH.
    """
    low, high = 0.0, _PALETTE_CHROMA
    for _ in range(20):
        middle = (low + high) / 2
        candidate = _oklch_to_srgb(lightness, middle, hue)
        if all(-1e-4 <= channel <= 1 + 1e-4 for channel in candidate):
            low = middle
        else:
            high = middle
    return low


def class_colour(class_id: int) -> tuple[int, int, int]:
    """Deterministic, perceptually even RGB colour for a class index."""
    hue = (class_id * _HUE_STEP) % 360.0
    lightness = _PALETTE_BANDS[(class_id // 9) % len(_PALETTE_BANDS)]
    chroma = _max_chroma(lightness, hue)

    # Cyans and blues cannot hold much chroma at every lightness, and forcing
    # them to one fixed lightness turned them grey. Where that happens the
    # lightness slides within the band that stays legible on a white panel until
    # the hue is vivid again.
    if chroma < _MIN_USEFUL_CHROMA:
        low, high = _LIGHTNESS_RANGE
        steps = int(round((high - low) / 0.02))
        for index in range(steps + 1):
            candidate_lightness = low + index * 0.02
            candidate_chroma = _max_chroma(candidate_lightness, hue)
            if candidate_chroma > chroma:
                lightness, chroma = candidate_lightness, candidate_chroma
            if chroma >= 0.145:
                break

    channels = _oklch_to_srgb(lightness, chroma, hue)
    return tuple(max(0, min(255, round(channel * 255))) for channel in channels)  # type: ignore[return-value]


def class_colour_hex(class_id: int) -> str:
    red, green, blue = class_colour(class_id)
    return f"#{red:02x}{green:02x}{blue:02x}"


# --------------------------------------------------------------------------
# Letterbox geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Letterbox:
    """The geometry of one resize-and-pad, together with its inverse."""

    scale: float
    pad_x: float
    pad_y: float
    source_width: int
    source_height: int

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        """Map (N, 4) xyxy boxes from network space back to source pixels."""
        if len(boxes) == 0:
            return boxes.reshape(0, 4)
        out = boxes.astype(np.float64).copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad_x) / self.scale
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad_y) / self.scale
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, self.source_width)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, self.source_height)
        return out


def letterbox(image: np.ndarray, size: int, *, fill: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Resize preserving aspect ratio and pad to a square.

    Padding rather than stretching: YOLO models are trained on letterboxed
    input with this exact grey fill, so the padding reads as padding to the
    network instead of as content.
    """
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((size, size, 3), fill, dtype=image.dtype)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    return canvas, Letterbox(
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
        source_width=width,
        source_height=height,
    )


# --------------------------------------------------------------------------
# Decoding and NMS
# --------------------------------------------------------------------------


def decode_yolo(
    output: np.ndarray, num_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv8/v11 detection head into (boxes_xyxy, scores, class_ids).

    The head emits `(1, 4 + num_classes [+ mask_coeffs], num_anchors)` with boxes
    in centre form at network scale and per-class scores already activated --
    there is no separate objectness term. Exporters differ on whether the tensor
    is transposed, so orientation is decided by matching the channel count
    rather than by trusting a flag.
    """
    if output.ndim == 3:
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f"Unexpected detection output shape {output.shape}")

    # Exporters disagree on whether the tensor is transposed, so the channel
    # axis is identified by what it must contain -- at least 4 box values plus
    # one score per class -- rather than by assuming it is the smaller axis.
    # Assuming that holds for every real model but breaks on small inputs, and
    # a wrongly transposed tensor decodes into plausible, wrong boxes.
    needed = 4 + num_classes
    candidates = [axis for axis in (0, 1) if output.shape[axis] >= needed]

    if not candidates:
        raise ValueError(
            f"Detection output {output.shape} has too few channels for "
            f"{num_classes} classes"
        )

    # When either axis could be the channels -- the usual case, since the anchor
    # count also exceeds the channel count -- the channel axis is the smaller.
    # A segmentation head makes it 4 + classes + 32, still far below the
    # thousands of anchors.
    axis = min(candidates, key=lambda index: output.shape[index])
    channels_first = output if axis == 0 else output.T

    boxes = channels_first[:4].T.astype(np.float64)
    scores_per_class = channels_first[4 : 4 + num_classes]

    class_ids = scores_per_class.argmax(axis=0).astype(np.int32)
    scores = scores_per_class.max(axis=0).astype(np.float64)

    return _xywh_to_xyxy(boxes), scores, class_ids


def is_end2end(output: np.ndarray) -> bool:
    """Whether a detection head has already done its own NMS.

    YOLOv10 and YOLO26 are NMS-free: instead of `(1, 4 + classes, anchors)`
    needing decoding and suppression, they emit a fixed-length `(1, N, 6)` list
    of final detections as `x1, y1, x2, y2, score, class_id`. The two are told
    apart by the trailing axis being exactly 6 while the middle axis is the
    detection count, which no anchor-style head produces -- its trailing axis is
    the anchor count, in the thousands.
    """
    return output.ndim == 3 and output.shape[2] == 6 and output.shape[1] <= 1000


def decode_end2end(output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an end-to-end head's output into (boxes_xyxy, scores, class_ids).

    Boxes are already in network-input coordinates and already suppressed, so
    the caller unmaps them and does no NMS.
    """
    rows = output[0]
    boxes = rows[:, :4].astype(np.float64)
    scores = rows[:, 4].astype(np.float64)
    class_ids = rows[:, 5].astype(np.int32)
    return boxes, scores, class_ids


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    out = np.empty_like(boxes, dtype=np.float64)
    half_w, half_h = boxes[:, 2] / 2.0, boxes[:, 3] / 2.0
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    *,
    max_detections: int = 300,
) -> np.ndarray:
    """Class-aware greedy non-maximum suppression; returns kept indices.

    Suppression is per class: a person standing in front of a chair produces two
    heavily overlapping boxes that are both correct, and a class-agnostic pass
    would delete one of them.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    keep: list[int] = []
    for class_id in np.unique(class_ids):
        members = np.flatnonzero(class_ids == class_id)
        order = members[np.argsort(-scores[members])]
        while order.size > 0:
            best = int(order[0])
            keep.append(best)
            if order.size == 1:
                break
            overlap = _iou(boxes[best], boxes[order[1:]])
            order = order[1:][overlap <= iou_threshold]

    keep.sort(key=lambda index: -scores[index])
    return np.array(keep[:max_detections], dtype=np.int64)


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    lt = np.maximum(box[:2], others[:, :2])
    rb = np.minimum(box[2:], others[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    intersection = wh[:, 0] * wh[:, 1]

    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.clip(others[:, 2] - others[:, 0], 0, None) * np.clip(
        others[:, 3] - others[:, 1], 0, None
    )
    union = area + areas - intersection
    return np.where(union > 0, intersection / union, 0.0)


def _to_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
) -> list[RawDetection]:
    detections: list[RawDetection] = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        index = int(class_id)
        name = class_names[index] if 0 <= index < len(class_names) else f"class_{index}"
        x_min, y_min, x_max, y_max = (int(round(float(v))) for v in box)
        if x_max <= x_min or y_max <= y_min:
            # A degenerate box carries no information and would render as an
            # invisible marker in the UI; dropping it is better than showing it.
            continue
        detections.append(
            RawDetection(
                class_id=index,
                class_name=name,
                confidence=float(score),
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )
    return detections


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class Detector(Protocol):
    """What the rest of the service may assume about a detection model."""

    info: ModelInfo

    def detect(
        self,
        image: np.ndarray,
        *,
        confidence: float,
        iou: float,
        max_detections: int,
    ) -> tuple[list[RawDetection], float]:
        """Detect in a BGR image. Returns detections and inference milliseconds."""
        ...


def _parse_names(raw: str) -> list[str]:
    """Read an Ultralytics `names` metadata string into an ordered list.

    The value is a Python dict literal keyed by class index, e.g.
    ``{0: 'person', 1: 'bicycle'}``. `ast.literal_eval` parses it without
    executing anything.
    """
    import ast

    parsed = ast.literal_eval(raw)
    if isinstance(parsed, dict):
        return [str(parsed[key]) for key in sorted(parsed, key=int)]
    return [str(item) for item in parsed]


class OnnxDetector:
    """Serves an ONNX export through onnxruntime."""

    def __init__(self, spec: ModelSpec, path: Path, device: str) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        wants_gpu = device in {"auto", "cuda"}
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if wants_gpu and "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )

        self._session = ort.InferenceSession(str(path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        metadata = self._session.get_modelmeta().custom_metadata_map or {}
        names = _parse_names(metadata["names"]) if "names" in metadata else []
        if not names:
            raise DetectorUnavailable(
                f"{path.name} carries no class names in its metadata. Re-export it "
                "with Ultralytics, or add the names to the model registry."
            )

        input_shape = self._session.get_inputs()[0].shape
        static_edge = input_shape[2] if isinstance(input_shape[2], int) else None
        input_size = static_edge or _metadata_imgsz(metadata) or 640

        chosen = "cuda" if "CUDAExecutionProvider" in self._session.get_providers() else "cpu"

        self.info = ModelInfo(
            name=spec.name,
            label=spec.label,
            backend="onnx",
            task=metadata.get("task", "detect"),
            device=chosen,
            input_size=int(input_size),
            class_names=names,
            weights_path=str(path),
            description=spec.description,
        )

    def detect(
        self,
        image: np.ndarray,
        *,
        confidence: float,
        iou: float,
        max_detections: int,
    ) -> tuple[list[RawDetection], float]:
        size = self.info.input_size
        padded, transform = letterbox(image, size)
        tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32)

        started = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: tensor})
        inference_ms = (time.perf_counter() - started) * 1000.0

        raw = outputs[0]

        if is_end2end(raw):
            # Already suppressed by the network; running NMS again would only
            # risk discarding genuine neighbouring objects.
            boxes, scores, class_ids = decode_end2end(raw)
            above = scores >= confidence
            boxes, scores, class_ids = boxes[above], scores[above], class_ids[above]
            order = np.argsort(-scores)[:max_detections]
            boxes, scores, class_ids = boxes[order], scores[order], class_ids[order]
        else:
            boxes, scores, class_ids = decode_yolo(raw, len(self.info.class_names))
            above = scores >= confidence
            boxes, scores, class_ids = boxes[above], scores[above], class_ids[above]
            kept = nms(boxes, scores, class_ids, iou, max_detections=max_detections)
            boxes, scores, class_ids = boxes[kept], scores[kept], class_ids[kept]

        return (
            _to_detections(
                transform.to_source(boxes), scores, class_ids, self.info.class_names
            ),
            inference_ms,
        )


def _metadata_imgsz(metadata: dict[str, str]) -> int | None:
    raw = metadata.get("imgsz")
    if not raw:
        return None
    try:
        import ast

        value = ast.literal_eval(raw)
        return int(value[0]) if isinstance(value, (list, tuple)) else int(value)
    except (ValueError, SyntaxError, TypeError, IndexError):
        return None


class UltralyticsDetector:
    """Serves a `.pt` checkpoint through the Ultralytics API.

    Also the path for open-vocabulary models: when the spec carries `prompts`,
    they become the class vocabulary, which is how construction terms the COCO
    label set has no word for become detectable without training.
    """

    def __init__(self, spec: ModelSpec, path: Path, device: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise DetectorUnavailable(
                "ultralytics is not installed. Install it with "
                "`pip install -r requirements.txt`."
            ) from exc

        resolved_device = _torch_device(device)
        # A bare name such as "yolo11n.pt" is left for Ultralytics to fetch from
        # its own cache or the network; an explicit path must exist.
        source = str(path) if path.exists() else spec.weights
        self._model = YOLO(source)
        self._device = resolved_device
        self._prompts = list(spec.prompts)
        self._class_thresholds = dict(spec.class_thresholds)

        if self._prompts:
            setter = getattr(self._model, "set_classes", None)
            if setter is None:
                raise DetectorUnavailable(
                    f"Model {spec.name!r} declares text prompts, but this checkpoint "
                    "is not open-vocabulary. Use a YOLO-World style checkpoint."
                )
            setter(self._prompts)
            names, self._prompt_to_class = build_vocabulary(
                self._prompts, spec.label_map
            )
        else:
            names = _names_from_model(self._model)
            self._prompt_to_class = np.arange(len(names), dtype=np.int32)

        self.info = ModelInfo(
            name=spec.name,
            label=spec.label,
            backend="ultralytics",
            task=str(getattr(self._model, "task", "detect")),
            device=resolved_device,
            input_size=640,
            class_names=names,
            weights_path=str(path if path.exists() else spec.weights),
            description=spec.description,
        )

    def detect(
        self,
        image: np.ndarray,
        *,
        confidence: float,
        iou: float,
        max_detections: int,
    ) -> tuple[list[RawDetection], float]:
        started = time.perf_counter()
        results = self._model.predict(
            source=image,
            imgsz=self.info.input_size,
            conf=confidence,
            iou=iou,
            max_det=max_detections,
            device=self._device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0

        boxes = np.empty((0, 4))
        scores = np.empty(0)
        class_ids = np.empty(0, dtype=np.int32)

        if results:
            found = results[0].boxes
            if found is not None and len(found):
                boxes = found.xyxy.cpu().numpy().astype(np.float64)
                scores = found.conf.cpu().numpy().astype(np.float64)
                class_ids = found.cls.cpu().numpy().astype(np.int32)

        # Ultralytics returns boxes in source coordinates and has already run
        # NMS -- but per *prompt*, which is not the same as per class once
        # synonyms collapse. One light fixture matched by "strip light" and
        # "tube light" survives that pass as two boxes and would be counted
        # twice, so the vocabulary is applied first and NMS repeated over the
        # display classes.
        if len(class_ids):
            mapped = self._prompt_to_class[class_ids]
            keep = mapped >= 0
            boxes, scores, class_ids = boxes[keep], scores[keep], mapped[keep]

            if self._class_thresholds and len(class_ids):
                floors = np.array(
                    [
                        self._class_thresholds.get(self.info.class_names[index], 0.0)
                        for index in class_ids
                    ]
                )
                above = scores >= floors
                boxes, scores, class_ids = boxes[above], scores[above], class_ids[above]

            if len(class_ids):
                kept = nms(boxes, scores, class_ids, iou, max_detections=max_detections)
                boxes, scores, class_ids = boxes[kept], scores[kept], class_ids[kept]

        return (
            _to_detections(boxes, scores, class_ids, self.info.class_names),
            inference_ms,
        )


def build_vocabulary(
    prompts: list[str], label_map: dict[str, str]
) -> tuple[list[str], np.ndarray]:
    """Turn prompts plus a synonym map into display classes and an index map.

    Returns the display class names and an array taking a prompt index to its
    display-class index, or to -1 for a prompt that is scored but not reported.

    Display classes are ordered by first appearance in the prompt list, which
    makes the ordering a property of the registry file rather than of a dict's
    iteration: the same vocabulary therefore gets the same colours every run.
    """
    names: list[str] = []
    index: dict[str, int] = {}
    mapping: list[int] = []

    for prompt in prompts:
        target = label_map.get(prompt, prompt)
        if not target:
            mapping.append(-1)
            continue
        if target not in index:
            index[target] = len(names)
            names.append(target)
        mapping.append(index[target])

    return names, np.array(mapping, dtype=np.int32)


def _names_from_model(model: Any) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=int)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def available_memory_bytes() -> int | None:
    """Memory a new allocation could actually obtain, or None if unknown.

    On Windows this is the *commit* headroom rather than free RAM, because that
    is what actually fails: a machine with 4 GB of RAM free and a full page file
    cannot commit another gigabyte, and the allocation that discovers this is
    typically deep inside a native library.
    """
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(min(status.ullAvailPageFile, status.ullAvailPhys))

    try:
        import os

        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):  # pragma: no cover - platform dependent
        return None


# Measured need for the PyTorch path: the framework itself, a checkpoint, and
# for open-vocabulary models a 338 MB CLIP text encoder that is read into memory
# whole to be hash-checked.
_TORCH_MEMORY_BYTES = 1_200_000_000
_OPEN_VOCAB_MEMORY_BYTES = 2_000_000_000


def _check_memory_headroom(spec: ModelSpec) -> None:
    """Refuse a load that the machine cannot complete.

    This exists because of an observed crash, not a hypothetical one: when the
    allocation fails inside torch or CLIP rather than in Python, the process can
    be left unusable and the whole server dies -- taking with it results the
    user had not finished looking at. Declining up front turns a segfault into a
    sentence, and leaves the ONNX models serving.
    """
    needed = _OPEN_VOCAB_MEMORY_BYTES if spec.prompts else _TORCH_MEMORY_BYTES
    available = available_memory_bytes()
    if available is None or available >= needed:
        return

    raise DetectorUnavailable(
        f"Not enough memory to load {spec.label!r}: about "
        f"{needed / 1e9:.1f} GB is needed and {available / 1e9:.1f} GB is "
        f"available. Close some applications and try again, or use an ONNX "
        f"model such as 'yolo11n', which needs far less. Attempting the load "
        f"anyway risks crashing the server."
    )


def _torch_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover - torch ships with ultralytics
        pass
    if device == "cuda":
        # Asked for explicitly but genuinely absent. Falling back silently would
        # make a GPU deployment look healthy while running on the CPU.
        raise DetectorUnavailable(
            "VISIONAI_DEVICE=cuda was requested but no CUDA device is available "
            "to PyTorch. Install a CUDA build of torch, or use VISIONAI_DEVICE=cpu."
        )
    return "cpu"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass
class _Entry:
    spec: ModelSpec
    detector: Detector | None = None
    error: str | None = None
    load_seconds: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class ModelRegistry:
    """Loads models on first use and keeps them for the process lifetime.

    Reloading a network per request would dominate the response time -- model
    construction costs seconds where inference costs tens of milliseconds -- so
    the first request for a model pays for it and the rest do not. The per-entry
    lock means two concurrent first requests load it once rather than twice.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._entries = {spec.name: _Entry(spec=spec) for spec in settings.models}
        self._resident: list[str] = []
        """Loaded model names, least recently used first."""

        self._resident_lock = threading.Lock()

    @property
    def default_name(self) -> str:
        return self._settings.default_model

    def names(self) -> list[str]:
        return list(self._entries)

    def loaded(self) -> list[str]:
        with self._resident_lock:
            return list(self._resident)

    def get(self, name: str | None = None) -> Detector:
        """Return a ready detector, loading it if this is its first use."""
        key = name or self._settings.default_model
        entry = self._entries.get(key)
        if entry is None:
            known = ", ".join(self._entries) or "none configured"
            raise DetectorUnavailable(f"Unknown model {key!r}. Available: {known}.")

        with entry.lock:
            if entry.detector is not None:
                self._touch(key)
                return entry.detector

            self._evict_for(key)
            started = time.perf_counter()
            try:
                entry.detector = self._build(entry.spec)
            except DetectorUnavailable as exc:
                entry.error = str(exc)
                raise
            except MemoryError as exc:
                # Worth its own message. A bare "MemoryError: bad allocation"
                # reads like a bug in the application, when in fact the model
                # is fine and the machine simply cannot hold it -- and the user
                # has a real choice to make about which model to run.
                entry.error = (
                    f"Not enough memory to load {entry.spec.label!r}. The "
                    f"open-vocabulary models need roughly 2 GB free (PyTorch "
                    f"plus a 338 MB CLIP text encoder). Close some applications "
                    f"and retry, or use an ONNX model such as 'yolo11n', which "
                    f"needs far less."
                )
                raise DetectorUnavailable(entry.error) from exc
            except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the caller
                entry.error = f"{type(exc).__name__}: {exc}"
                raise DetectorUnavailable(entry.error) from exc
            entry.error = None
            entry.load_seconds = time.perf_counter() - started
            self._touch(key)
            return entry.detector

    def _touch(self, name: str) -> None:
        with self._resident_lock:
            if name in self._resident:
                self._resident.remove(name)
            self._resident.append(name)

    def _evict_for(self, incoming: str) -> None:
        """Release least-recently-used models to make room for `incoming`."""
        cap = max(1, self._settings.max_loaded_models)
        while True:
            with self._resident_lock:
                if len(self._resident) < cap:
                    return
                victim = next(
                    (name for name in self._resident if name != incoming), None
                )
                if victim is None:
                    return
                self._resident.remove(victim)

            # Dropping the only reference is what frees the session; the
            # collection is what makes it happen now rather than whenever the
            # allocator next runs, which is the point of evicting at all.
            entry = self._entries[victim]
            entry.detector = None
            entry.load_seconds = None
            import gc

            gc.collect()

    def _build(self, spec: ModelSpec) -> Detector:
        path = self._settings.resolve_weights(spec.weights)
        backend = spec.backend.lower()

        if backend == "onnx":
            if not path.exists():
                raise DetectorUnavailable(
                    f"Weights for {spec.name!r} not found at {path}. Export them with "
                    "`python -m scripts.export_model`, or set VISIONAI_WEIGHTS_DIR."
                )
            return OnnxDetector(spec, path, self._settings.device)

        if backend == "ultralytics":
            _check_memory_headroom(spec)
            return UltralyticsDetector(spec, path, self._settings.device)

        raise DetectorUnavailable(
            f"Unknown backend {spec.backend!r} for model {spec.name!r}. "
            "Supported backends: onnx, ultralytics."
        )

    def describe(self) -> list[dict[str, Any]]:
        """Registry summary for /api/models.

        Reports a model that has not been loaded yet as `status: "available"`
        when its weights are present, rather than loading every model just to
        answer the question -- which would cost seconds and a lot of memory.
        """
        summary: list[dict[str, Any]] = []
        for entry in self._entries.values():
            spec = entry.spec
            path = self._settings.resolve_weights(spec.weights)

            if entry.detector is not None:
                info = entry.detector.info
                summary.append(
                    {
                        "name": spec.name,
                        "label": spec.label,
                        "description": spec.description,
                        "backend": info.backend,
                        "task": info.task,
                        "device": info.device,
                        "input_size": info.input_size,
                        "status": "loaded",
                        "detail": f"loaded in {entry.load_seconds:.1f}s",
                        "class_count": len(info.class_names),
                        "classes": info.class_names,
                        "weights_path": info.weights_path,
                        "is_default": spec.name == self._settings.default_model,
                        "recommended_confidence": spec.recommended_confidence,
                    }
                )
                continue

            if entry.error is not None:
                status, detail = "error", entry.error
            elif path.exists():
                status, detail = "available", "not loaded yet; loads on first detection"
            elif spec.backend == "ultralytics":
                status, detail = (
                    "available",
                    f"{spec.weights} is absent locally and will be downloaded on "
                    "first use",
                )
            else:
                status, detail = "missing", f"weights not found at {path}"

            summary.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "description": spec.description,
                    "backend": spec.backend,
                    "task": None,
                    "device": None,
                    "input_size": None,
                    "status": status,
                    "detail": detail,
                    "class_count": None,
                    "classes": [],
                    "weights_path": str(path),
                    "is_default": spec.name == self._settings.default_model,
                    "recommended_confidence": spec.recommended_confidence,
                }
            )
        return summary
