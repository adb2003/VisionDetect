# Construction detection vocabulary — what is measured and what is not

`models.construction.json` adds construction-relevant models to the base COCO
registry. This file records how much evidence stands behind each class,
because the honest answer differs sharply between them and the difference
matters on site: a class that has never been validated reporting nothing is
not evidence that the thing is absent.

## `ppe-construction` — a fine-tuned model, and the one that actually works

The only model here that was *trained* on construction imagery rather than
prompted. YOLOv8 fine-tuned on roughly 13,000 images, 19 classes:

> Boots · Ear-Protection · Glass · Glove · Hard_hat · Mask · Worker · Vest ·
> No-Boots · No-Ear-Protection · No-Glass · No-Glove · No-Helmet · No-Mask ·
> No-Vest · Circular_Saw · Fire_Extinguisher · Fire_prevention_Net ·
> Welding_Equipment

Apache-2.0, from `huggingface.co/killuminati1/construction-ppe-yolov8`.
Uploader-reported mAP@50 ≈ 0.76. Obtain it with:

```bash
python scripts/fetch_model.py --model ppe-construction
```

### Measured here

On `bus.jpg` at a 0.40 threshold it returns 18 objects: Worker ×3 (max 0.853),
Boots ×4 (0.801), No-Glove ×4 (0.799), No-Vest ×2 (0.748), No-Helmet ×2
(0.646), No-Mask ×2, No-Glass ×1. People at a bus stop are indeed workers-shaped
and are indeed wearing no PPE, so the *absence* classes firing is the correct
answer rather than a failure. `zidane.jpg` behaves the same way. (Both are the
public sample images shipped inside the `ultralytics` package.)

Set against the open-vocabulary results below, the contrast is the whole
argument for fine-tuning: `helmet` and `hi-vis vest` scored **0.000** on every
prompt phrasing tried against sample construction photography, while this
model reports them, and their violations, above 0.6.

### What it is not

It detects **people and their equipment**, not building fabric. It has no class
for a pipe, duct, cable tray, conduit or sprinkler — that still needs either a
fine-tune on MEP imagery or an open-vocabulary model, with the caveats below.

It is also community-trained on merged public datasets. The uploader's metrics
are self-reported, the `Ear-Protection` class is noted as under-represented,
and nothing here has been independently re-validated.

### Why it is served as ONNX

`fetch_model.py` downloads the `.pt`, audits its pickle statically, converts to
ONNX and deletes the checkpoint. A PyTorch checkpoint is a pickle, and
unpickling runs whatever the file says to run — loading a community checkpoint
is executing a stranger's code. The audit parses the opcodes without executing
them and refuses anything referencing a module outside an allowlist. For this
file it found 24 globals across `torch`, `ultralytics`, `collections` and
`__builtin__`, and nothing else. The served artefact is ONNX, which is parsed
as data, so normal operation never unpickles anything.

An audit is not a proof. It is the difference between checking and hoping.

## `construction-hazard` — site safety, with plant and layout

YOLO26 nano fine-tuned for construction hazards, 11 classes:

> Hardhat · Mask · Safety Vest · NO-Hardhat · NO-Mask · NO-Safety Vest ·
> Person · Safety Cone · machinery · vehicle · utility pole

From `huggingface.co/yihong1120/Construction-Hazard-Detection`. It overlaps
`ppe-construction` on PPE and adds what that model has no class for: plant,
vehicles, cones and utility poles — the things that make a site photograph a
site rather than a room with people in it.

```bash
python scripts/fetch_model.py --model construction-hazard
```

**AGPL-3.0.** That is copyleft: for software offered over a network it
requires offering the corresponding source to its users. `models/README.md`
covers the same question for the Ultralytics-licensed weights, and the answer
has to be the same one. The fetch script prints the warning rather than
letting it pass unnoticed.

### Measured here

On `bus.jpg` at 0.35 it returns `vehicle` (0.740) over the bus, Person ×3,
NO-Hardhat ×2 on the two visible heads, NO-Safety Vest and NO-Mask. On
`zidane.jpg`, Person ×3 with NO-Hardhat boxes on the heads and NO-Safety Vest
on the torsos. Every box falls inside the frame.

The `vehicle` box lands at (8, 209, 808, 761) where the COCO model independently
puts `bus` at (12, 228, 798, 734) — two unrelated networks agreeing on the same
object to within about twenty pixels, which is a useful check that the
coordinate handling is right for this head.

### It needs a different decoder

The publisher ships ONNX, so there is no pickle and the fetch script only
verifies the file parses and carries its class names. But its output shape is
`(1, 300, 6)`, not the `(1, 4 + classes, 8400)` that YOLOv8 and YOLO11 emit.

YOLO26 is **end-to-end**: suppression happens inside the network, and the
output is a finished list of `x1, y1, x2, y2, score, class_id` rather than
per-anchor predictions needing decoding and NMS. `is_end2end` distinguishes the
two by the trailing axis being exactly 6 while the detection count is small,
and the end-to-end path skips NMS entirely — running it again would only risk
discarding genuine neighbouring objects the network already decided to keep.

## How open-vocabulary detection works here

YOLO-World takes a list of text prompts at inference time and scores every
region against each. There is no training step, which is the only reason a
vocabulary of construction terms is possible at all without a labelled dataset
— but also the reason the results are uneven. The model was never taught what
a cable tray is; it is matching a CLIP text embedding against image features
and hoping.

Two consequences shape the design of the registry, both found while calibrating
`construction-calibrated` and `fitout-extended` against a small set of real
construction photographs (used only for threshold-tuning during development;
not included in this repository):

**Phrasing is not interchangeable, and the winner is not the obvious one.**
`unfinished ceiling` scored well above `exposed concrete soffit` — a more
precise description of the same thing — which scored effectively zero on the
same images. So each class is asked in several phrasings, and `label_map`
collapses them onto one reported class.

**Scores depend on the whole prompt set, not just the prompt being scored.**
The same prompt measured over an order of magnitude higher with a 40-prompt
set than with a 14-prompt set, on an identical image. Prompts therefore cannot
be added or removed freely: doing so invalidates every threshold. In
`construction-calibrated` the unreported prompts are load-bearing context,
which is why they are mapped to `""` — scored, never reported.

## `construction-calibrated` — four classes, all measured

Reports only the four classes that, in that calibration pass, had a threshold
that reliably separated images where the element was present from images
where it was not: `exposed slab`, `light`, `column`, `partition`.

**Measured and failed** — asked for, and did not reliably separate present
from absent: sprinkler, pipe, conduit, cable tray, air diffuser, smoke
detector, window, debris. They are deliberately not reported by this
configuration, which is why `fitout-extended` exists as the broader, less
validated alternative below.

Confidences top out well under the levels a closed-set model reports, so even
the passing classes are indicative rather than dependable — the recommended
confidence for this model is 0.15, not the dashboard's 0.35 default.

## `fitout-extended` — 38 classes, broad but largely unvalidated

118 prompts covering interior fit-out: structure, finishes, MEP, temporary
works, safety equipment and stored materials. A handful of classes (exposed
slab, light, column, wall) are reasonably consistent in calibration; most of
the vocabulary is either measured-and-failing (notably the whole MEP group —
pipe, sprinkler, conduit — which is the gap that matters most for fit-out
progress) or simply untested because nothing in the small calibration set
contained it. Several classes carry suppression floors in `class_thresholds`
because an empty scene produced confident false positives for them (`ladder`,
`door`, `fire extinguisher` among others).

**The real fix is a fine-tune.** Open-vocabulary detection is how you get a
construction vocabulary without a dataset; it is not how you get a reliable
one. A model trained on labelled site imagery would detect these classes
properly, and this service loads one through the same registry mechanism with
no code change — class names are read from the checkpoint's own metadata.
