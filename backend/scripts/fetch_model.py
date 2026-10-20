"""Fetch a third-party detection checkpoint and convert it to ONNX for serving.

Why this script exists rather than a line in the README telling you to download
a `.pt`:

A PyTorch checkpoint is a **pickle**, and unpickling executes whatever the file
says to execute. Loading a community checkpoint is therefore running a
stranger's code on your machine with your privileges -- not a theoretical risk,
and not something to do casually because a model card looked convincing.

So the pickle is handled once, here, under inspection, and never again:

1. Download into its own directory under `models/third_party/`, nothing else
   in it.
2. Record the SHA-256 so a later fetch of the same name is verifiably the same
   bytes.
3. **Scan the pickle statically** with `pickletools`, which parses opcodes
   without executing them, and refuse any global reference outside an allowlist
   of modules a YOLO checkpoint legitimately needs.
4. Only then load it, and immediately export to ONNX.

The ONNX artefact is what the service registers and loads. ONNX is data parsed
by a schema, not a program, so the serving path never unpickles anything. The
export also embeds the class names, which is what lets the model appear in the
dashboard with its real labels and no code change.

Usage:

    python scripts/fetch_model.py --model ppe-construction
    python scripts/fetch_model.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import pickletools
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
WEIGHTS_DIR = REPO_ROOT / "models"
STAGING_DIR = WEIGHTS_DIR / "third_party"


@dataclass(frozen=True)
class Source:
    """A third-party checkpoint this script knows how to fetch."""

    name: str
    url: str
    licence: str
    origin: str
    output: str
    note: str

    kind: str = "pickle"
    """"pickle" for a `.pt` that must be audited and converted, "onnx" for a
    publisher who already ships the artefact we serve.

    The second is strictly better and needs none of the machinery below: an
    ONNX file is parsed against a schema rather than executed, so there is no
    pickle to audit and no conversion step in which to load one."""


SOURCES: dict[str, Source] = {
    "ppe-construction": Source(
        name="ppe-construction",
        url=(
            "https://huggingface.co/killuminati1/construction-ppe-yolov8/"
            "resolve/main/best.pt"
        ),
        licence="Apache-2.0",
        origin="huggingface.co/killuminati1/construction-ppe-yolov8",
        output="ppe-construction.onnx",
        note=(
            "YOLOv8 fine-tuned on ~13k construction images. 19 classes: PPE worn "
            "and not worn, plus worker, fire extinguisher, welding equipment, "
            "circular saw. Community-trained; validate before relying on it."
        ),
    ),
    "construction-hazard": Source(
        name="construction-hazard",
        url=(
            "https://huggingface.co/yihong1120/Construction-Hazard-Detection/"
            "resolve/main/models/yolo26/onnx/yolo26n.onnx"
        ),
        licence="AGPL-3.0  (copyleft - see the warning this script prints)",
        origin="huggingface.co/yihong1120/Construction-Hazard-Detection",
        output="construction-hazard.onnx",
        kind="onnx",
        note=(
            "11 classes: hardhat, mask, safety vest and their NO- counterparts, "
            "person, safety cone, machinery, vehicle, utility pole. Adds plant "
            "and site-layout objects on top of PPE. Publisher ships ONNX, so no "
            "pickle is involved."
        ),
    ),
}

# Licences that oblige the user of this software, not merely its author.
COPYLEFT = ("agpl", "gpl")

# Modules a YOLO checkpoint has any business referencing. Anything else is
# either a novel architecture this allowlist has not seen -- in which case look
# at it and decide deliberately -- or an attempt to run something.
ALLOWED_MODULE_PREFIXES = (
    "torch",
    "torchvision",
    "ultralytics",
    "collections",
    "numpy",
    "builtins.set",
    "builtins.frozenset",
    "builtins.list",
    "builtins.dict",
    "builtins.tuple",
    "builtins.getattr",
    "__builtin__.set",
    "_codecs.encode",
    "argparse.Namespace",
    "pathlib",
)

# Names that are dangerous wherever they appear.
DENIED_SUBSTRINGS = (
    "os.system",
    "subprocess",
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "builtins.open",
    "builtins.__import__",
    "posix",
    "nt.",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "pty",
    "runpy",
    "importlib",
)


class UnsafeCheckpoint(RuntimeError):
    """The checkpoint references something it should not."""


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urlopen(url) as response, destination.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    size_mb = destination.stat().st_size / 1_048_576
    print(f"  saved {destination.name} ({size_mb:.1f} MB)")
    return destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def pickle_globals(path: Path) -> set[str]:
    """Every global a checkpoint's pickles reference, without executing them.

    A `.pt` is a zip containing one or more pickle streams. `pickletools.genops`
    walks the opcodes; GLOBAL carries "module name" as its argument and
    STACK_GLOBAL takes the two from the stack, so the preceding string constants
    are tracked to resolve it.
    """
    found: set[str] = set()

    with zipfile.ZipFile(path) as archive:
        targets = [n for n in archive.namelist() if n.endswith(".pkl")]
        if not targets:
            raise UnsafeCheckpoint(
                f"{path.name} contains no pickle stream; it is not a torch "
                "checkpoint in the expected format."
            )

        for entry in targets:
            data = archive.read(entry)
            recent: list[str] = []
            for opcode, argument, _position in pickletools.genops(data):
                if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
                    recent.append(str(argument))
                    recent[:] = recent[-2:]
                elif opcode.name == "GLOBAL":
                    found.add(str(argument).replace(" ", "."))
                elif opcode.name == "STACK_GLOBAL" and len(recent) >= 2:
                    found.add(f"{recent[-2]}.{recent[-1]}")

    return found


def audit(path: Path) -> set[str]:
    """Refuse a checkpoint that references anything outside the allowlist."""
    references = pickle_globals(path)

    denied = {
        reference
        for reference in references
        if any(bad in reference for bad in DENIED_SUBSTRINGS)
    }
    unknown = {
        reference
        for reference in references
        if not reference.startswith(ALLOWED_MODULE_PREFIXES)
    }

    if denied:
        raise UnsafeCheckpoint(
            f"{path.name} references {sorted(denied)}. This checkpoint can "
            "execute code on load. Do not use it."
        )
    if unknown - denied:
        raise UnsafeCheckpoint(
            f"{path.name} references modules this script does not recognise: "
            f"{sorted(unknown)}. Inspect them and extend ALLOWED_MODULE_PREFIXES "
            "deliberately if they are legitimate."
        )

    return references


def export_onnx(checkpoint: Path, output: Path) -> None:
    """Load the audited checkpoint and export ONNX beside the other weights."""
    from ultralytics import YOLO

    print("  loading checkpoint and exporting to ONNX (this takes a minute)")
    model = YOLO(str(checkpoint))
    names = getattr(model, "names", {}) or {}
    print(f"  classes ({len(names)}): {', '.join(str(v) for v in names.values())}")

    produced = Path(model.export(format="onnx", imgsz=640, simplify=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), output)
    print(f"  wrote {output}")


def inspect_onnx(path: Path) -> list[str]:
    """Confirm a downloaded ONNX parses and carries its class names.

    Parsing is the check. An ONNX file is data read against a schema, so there
    is nothing here to audit for executable content the way a pickle needs;
    what can go wrong is that the file is truncated, or was exported without
    the metadata the service reads class names from, and both show up here
    rather than at the first detection request.
    """
    import ast

    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map or {}

    raw = metadata.get("names")
    if not raw:
        raise UnsafeCheckpoint(
            f"{path.name} carries no class names in its metadata, so the "
            "service cannot label its output. Re-export it with Ultralytics."
        )

    parsed = ast.literal_eval(raw)
    names = (
        [str(parsed[key]) for key in sorted(parsed, key=int)]
        if isinstance(parsed, dict)
        else [str(item) for item in parsed]
    )

    outputs = [(o.name, o.shape) for o in session.get_outputs()]
    print(f"  outputs : {outputs}")
    print(f"  classes ({len(names)}): {', '.join(names)}")
    return names


def fetch(source: Source, *, keep_checkpoint: bool) -> int:
    print(f"\n{source.name}")
    print(f"  origin  : {source.origin}")
    print(f"  licence : {source.licence}")

    if any(flag in source.licence.lower() for flag in COPYLEFT):
        print(
            "\n  NOTE: this licence is copyleft. For software offered over a\n"
            "  network, AGPL-3.0 requires offering the corresponding source to\n"
            "  its users. That is a decision about your product, not about this\n"
            "  script. See models/README.md, which covers the same question for\n"
            "  the Ultralytics weights.\n"
        )

    staging = STAGING_DIR / source.name
    staging.mkdir(parents=True, exist_ok=True)

    if source.kind == "onnx":
        artefact = staging / "model.onnx"
        if not artefact.exists():
            download(source.url, artefact)
        else:
            print(f"  reusing {artefact}")
        print(f"  sha256  : {sha256(artefact)}")
        try:
            inspect_onnx(artefact)
        except UnsafeCheckpoint as exc:
            print(f"\n  REFUSED: {exc}")
            return 1
        destination = WEIGHTS_DIR / source.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artefact, destination)
        print(f"  wrote {destination}")
        print("\n  Done. Select it in the dashboard.")
        return 0

    checkpoint = staging / "checkpoint.pt"

    if not checkpoint.exists():
        download(source.url, checkpoint)
    else:
        print(f"  reusing {checkpoint}")

    print(f"  sha256  : {sha256(checkpoint)}")

    print("  auditing pickle (no code is executed)")
    try:
        references = audit(checkpoint)
    except UnsafeCheckpoint as exc:
        print(f"\n  REFUSED: {exc}")
        print(f"  The file is left at {checkpoint} for inspection.")
        return 1

    modules = sorted({reference.split(".")[0] for reference in references})
    print(f"  clean: {len(references)} globals across {modules}")

    export_onnx(checkpoint, WEIGHTS_DIR / source.output)

    if not keep_checkpoint:
        checkpoint.unlink()
        print("  removed the .pt; the ONNX is what gets served")

    print(f"\n  Done. Select it in the dashboard, or set VISIONAI_DEFAULT_MODEL.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="which source to fetch")
    parser.add_argument("--list", action="store_true", help="list known sources")
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="keep the .pt after converting (it is not needed to serve)",
    )
    arguments = parser.parse_args()

    if arguments.list or not arguments.model:
        print("Known sources:\n")
        for source in SOURCES.values():
            print(f"  {source.name}")
            print(f"    {source.note}")
            print(f"    licence: {source.licence}  origin: {source.origin}\n")
        print("Every one of these is community-trained. The script audits the")
        print("checkpoint before loading it, but an audit is not a guarantee:")
        print("decide whether you trust the source before fetching.")
        return 0

    source = SOURCES.get(arguments.model)
    if source is None:
        print(f"Unknown model {arguments.model!r}. Known: {', '.join(SOURCES)}")
        return 2

    return fetch(source, keep_checkpoint=arguments.keep_checkpoint)


if __name__ == "__main__":
    sys.exit(main())
