# Model weights

**Weights are not committed.** `models/*.onnx` and `models/*.pt` are gitignored.
Two reasons, one practical and one legal.

## Practical

Binaries bloat git history permanently, cannot be diffed, and are exactly
reproducible from a checkpoint name plus an export command. Regenerate the
base COCO models:

```bash
pip install -r backend/requirements.txt   # includes ultralytics
python -m scripts.export_model --model yolo11n
python -m scripts.export_model --model yolo11n-seg
```

That writes the `.onnx` serving artefacts. Serving itself needs only
`onnxruntime` — no PyTorch — once the export has been done once.

Construction-specific checkpoints (`ppe-construction`, `construction-hazard`)
are fetched and converted separately — see
[backend/models.construction.md](../backend/models.construction.md) and
`backend/scripts/fetch_model.py`. The open-vocabulary models
(`construction-calibrated`, `fitout-extended`) use `yolov8s-worldv2.pt`, which
Ultralytics downloads on first use, plus a ~338 MB CLIP text-encoder download
required for inference.

## Licensing — read before shipping

**Ultralytics YOLO (v8, v11) and its pretrained weights are AGPL-3.0.** That
licence has real consequences for a commercial product: for software offered
over a network, AGPL-3.0 requires making the corresponding source available to
users of that service. Ultralytics sells an Enterprise licence as the
alternative.

`construction-hazard` (YOLO26, from a third-party fine-tune) is also
AGPL-3.0. `ppe-construction` is Apache-2.0.

This is not legal advice, and it is worth a proper review before this reaches
a customer. The practical options:

| Option | Implication |
|---|---|
| Ultralytics Enterprise licence | Paid; removes the AGPL obligation |
| Comply with AGPL-3.0 | Offer corresponding source to service users |
| Use a permissively licensed model | e.g. RT-DETR (Apache-2.0), or DETR/OWL-ViT variants; requires re-export and re-validation |
| Serve the ONNX path only, no Ultralytics at runtime | The default `yolo11n`/`yolo11n-seg` ONNX models need only `onnxruntime`, which carries no such obligation |

The architecture is deliberately model-agnostic to keep this a swap rather
than a rewrite: the serving path (`backend/app/detector.py`) consumes any
ONNX graph with a YOLO-style head or a YOLO26-style end-to-end head, and the
class mapping comes from the checkpoint's own metadata or `models.construction.json`.
Changing model is a new registry entry plus a new weights file.

`scripts/export_model.py` and `backend/scripts/fetch_model.py` are the only
places that touch Ultralytics or a third-party checkpoint directly, so
replacing either is self-contained.

## What the shipped config actually detects

The default model, `yolo11n`, is COCO-trained and has **no class for any
building or MEP element** — no ceiling, wall, duct, pipe, conduit, cable tray,
sprinkler, light fitting or slab. See
[backend/models.construction.md](../backend/models.construction.md) for the
models that add construction-specific classes, and exactly how much evidence
stands behind each one.
