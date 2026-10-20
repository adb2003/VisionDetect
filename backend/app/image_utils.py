"""Image decoding, validation and annotation rendering.

Validation happens here rather than at the route so that the same rules apply
however an image arrives. The checks are deliberately content-based: a file
named `.png` that is really a text file, or a 60-megapixel panorama that would
exhaust memory during inference, both get a specific message instead of a 500.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from app.detector import RawDetection, class_colour

# Magic bytes, checked instead of trusting the extension or the declared MIME
# type. WEBP is a RIFF container whose fourth word is "WEBP".
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
)


class ImageError(ValueError):
    """An upload that cannot be used, with a message fit to show a user."""


class EncodingError(RuntimeError):
    """The annotated image could not be encoded.

    Distinct from `ImageError` because the cause is on this side: OpenCV returns
    failure from `imencode` when it cannot allocate the output buffer, which
    happens under memory pressure rather than because anything is wrong with the
    image. Conflating the two would report a server resource problem as a
    rejected upload.
    """


@dataclass(frozen=True)
class DecodedImage:
    """A validated image, ready for inference."""

    pixels: np.ndarray
    """BGR, HxWx3, uint8."""

    width: int
    height: int
    detected_format: str
    size_bytes: int


def sniff_format(data: bytes) -> str | None:
    """Identify an image container from its leading bytes."""
    for signature, name in _SIGNATURES:
        if data.startswith(signature):
            return name
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def decode_image(
    data: bytes,
    *,
    filename: str,
    max_bytes: int,
    max_pixels: int,
    allowed_formats: tuple[str, ...],
) -> DecodedImage:
    """Validate and decode an uploaded image.

    Raises `ImageError` with an explanatory message for anything unusable.
    """
    if not data:
        raise ImageError("The uploaded file is empty.")

    if len(data) > max_bytes:
        raise ImageError(
            f"Image is {len(data) / 1024 / 1024:.2f} MB, which exceeds the "
            f"{max_bytes / 1024 / 1024:.0f} MB limit."
        )

    detected = sniff_format(data)
    if detected is None:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
        raise ImageError(
            f"{filename!r} is not a readable JPG, PNG or WEBP image "
            f"(its contents do not match any supported format; extension was "
            f".{suffix})."
        )

    permitted = {fmt.lower() for fmt in allowed_formats}
    # The two spellings describe one format; callers configure either.
    if detected == "jpeg":
        permitted_hit = {"jpg", "jpeg"} & permitted
    else:
        permitted_hit = {detected} & permitted
    if not permitted_hit:
        raise ImageError(
            f"{detected.upper()} images are not accepted. Supported formats: "
            f"{', '.join(sorted(permitted)).upper()}."
        )

    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ImageError(
            f"{filename!r} looks like a {detected.upper()} file but could not be "
            "decoded. It may be truncated or corrupt."
        )

    height, width = pixels.shape[:2]
    if height * width > max_pixels:
        raise ImageError(
            f"Image is {width}x{height} ({width * height / 1e6:.1f} megapixels), "
            f"above the {max_pixels / 1e6:.0f} megapixel limit."
        )

    return DecodedImage(
        pixels=pixels,
        width=width,
        height=height,
        detected_format=detected,
        size_bytes=len(data),
    )


# --------------------------------------------------------------------------
# Annotation
# --------------------------------------------------------------------------


def _render_scale(width: int, height: int) -> tuple[float, int]:
    """Font scale and line thickness proportionate to the image.

    A fixed 1 px box is invisible on a 4000 px photo and a fixed 20 px one
    swallows a thumbnail, so both scale with the image's diagonal.
    """
    diagonal = float(np.hypot(width, height))
    thickness = max(1, int(round(diagonal / 900.0)))
    font_scale = max(0.4, min(1.2, diagonal / 2200.0))
    return font_scale, thickness


def annotate(
    image: np.ndarray,
    detections: list[RawDetection],
    *,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw bounding boxes, class labels and confidences onto a copy of `image`.

    Boxes are drawn in the image's own pixel coordinates, which is the whole
    point of unmapping them in the detector: what the dashboard displays and
    what a downloaded annotated image shows are the same geometry.
    """
    canvas = image.copy()
    if not show_boxes and not show_labels:
        return canvas

    height, width = canvas.shape[:2]
    font_scale, thickness = _render_scale(width, height)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Lowest confidence first, so the most certain labels end up on top where
    # detections overlap.
    for detection in sorted(detections, key=lambda d: d.confidence):
        red, green, blue = class_colour(detection.class_id)
        colour = (blue, green, red)  # OpenCV is BGR

        if show_boxes:
            cv2.rectangle(
                canvas,
                (detection.x_min, detection.y_min),
                (detection.x_max, detection.y_max),
                colour,
                thickness,
            )

        if not show_labels:
            continue

        text = detection.class_name
        if show_confidence:
            text = f"{text} {detection.confidence * 100:.0f}%"

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pad = max(2, thickness)

        # Prefer the label above the box; drop it inside when there is no room,
        # so a detection touching the top edge keeps its label visible.
        label_bottom = detection.y_min - pad
        if label_bottom - text_h - baseline < 0:
            label_bottom = min(height - 1, detection.y_min + text_h + baseline + pad)
        label_top = label_bottom - text_h - baseline
        label_left = max(0, min(detection.x_min, width - text_w - 2 * pad))

        cv2.rectangle(
            canvas,
            (label_left, label_top),
            (label_left + text_w + 2 * pad, label_bottom),
            colour,
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (label_left + pad, label_bottom - baseline),
            font,
            font_scale,
            _contrasting_text_colour(red, green, blue),
            thickness,
            cv2.LINE_AA,
        )

    return canvas


def _contrasting_text_colour(red: int, green: int, blue: int) -> tuple[int, int, int]:
    """Black or white text, whichever stays readable on this label colour."""
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def encode_image(image: np.ndarray, fmt: str = "png", *, jpeg_quality: int = 92) -> bytes:
    """Encode a BGR array to PNG or JPEG bytes.

    Retried once after reclaiming memory, because the observed failure mode is
    an allocation failure inside the encoder on a loaded machine rather than
    anything repeatable about the image.
    """
    normalised = fmt.lower()
    if normalised not in {"png", "jpg", "jpeg"}:
        raise ImageError(f"Unsupported output format {fmt!r}; use png or jpeg.")

    extension = ".png" if normalised == "png" else ".jpg"
    params = [] if normalised == "png" else [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    for attempt in (1, 2):
        success, buffer = cv2.imencode(extension, image, params)
        if success:
            return buffer.tobytes()
        if attempt == 1:
            import gc

            gc.collect()

    raise EncodingError(
        f"Could not encode the annotated image as {normalised}. This is usually "
        "the server running short of memory; retry the request, or restart the "
        "backend if it persists."
    )


def media_type(fmt: str) -> str:
    return "image/jpeg" if fmt.lower() in {"jpg", "jpeg"} else "image/png"


def to_data_url(data: bytes, fmt: str = "png") -> str:
    """Wrap encoded image bytes as a data URL for direct use in an `img` tag."""
    return f"data:{media_type(fmt)};base64,{base64.b64encode(data).decode('ascii')}"
