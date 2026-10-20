"""Export detection weights to ONNX for the serving path.

    python -m scripts.export_model --model yolo11n
    python -m scripts.export_model --model yolo11n --imgsz 640 --out models/

Ultralytics (and therefore PyTorch) is needed here and only here. The serving
container runs `onnxruntime` against the artefact this produces, so the 2 GB
framework never ships to production.

Re-running is safe and deterministic for a given checkpoint and image size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(model_name: str, imgsz: int, out_dir: Path, opset: int) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is required to export weights.\n"
            "  pip install -e '.[cv-torch]'\n"
            "The serving path only needs onnxruntime ('.[cv]').",
            file=sys.stderr,
        )
        raise SystemExit(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = f"{model_name}.pt"

    model = YOLO(checkpoint)
    class_names = [model.names[i] for i in range(len(model.names))]

    exported = Path(
        model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False)
    )
    target = out_dir / exported.name
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), target)

    # Keep the checkpoint beside the ONNX: it is what reference-comparison tests
    # validate the numpy pipeline against.
    source_pt = Path(checkpoint)
    if source_pt.exists() and source_pt.resolve().parent != out_dir.resolve():
        shutil.move(str(source_pt), out_dir / source_pt.name)

    return {
        "model": model_name,
        "onnx": str(target.relative_to(REPO_ROOT)) if target.is_relative_to(REPO_ROOT) else str(target),
        "sha256": sha256(target),
        "imgsz": imgsz,
        "opset": opset,
        "num_classes": len(class_names),
        "class_names": class_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n", help="checkpoint stem, e.g. yolo11n")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models")
    parser.add_argument(
        "--print-classes",
        action="store_true",
        help="emit the class list as JSON, for building a detector config",
    )
    args = parser.parse_args()

    info = export(args.model, args.imgsz, args.out, args.opset)
    print(f"exported {info['onnx']}")
    print(f"  sha256      {info['sha256']}")
    print(f"  input       {info['imgsz']}x{info['imgsz']}, opset {info['opset']}")
    print(f"  classes     {info['num_classes']}")
    if args.print_classes:
        print(json.dumps(info["class_names"], indent=2))


if __name__ == "__main__":
    main()
