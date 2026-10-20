"""Tests for the open-vocabulary layer.

These cover the parts that are pure logic -- collapsing prompt synonyms onto
display classes, dropping context-only prompts, and suppressing the duplicate
boxes that collapsing creates. They need no model weights, which matters because
the open-vocabulary checkpoint cannot be loaded on every machine.

What they deliberately do NOT cover is whether the vocabulary actually finds
scaffolding in a photograph. That is a question about the model, answerable only
by running it against labelled imagery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import ModelSpec
from app.detector import build_vocabulary, nms

REGISTRY = Path(__file__).resolve().parents[1] / "models.construction.json"


# --------------------------------------------------------------------------
# Collapsing synonyms
# --------------------------------------------------------------------------


def test_synonyms_collapse_onto_one_display_class() -> None:
    names, mapping = build_vocabulary(
        ["strip light", "tube light", "concrete column"],
        {"strip light": "light", "tube light": "light", "concrete column": "column"},
    )
    assert names == ["light", "column"]
    assert mapping.tolist() == [0, 0, 1]


def test_an_unmapped_prompt_keeps_its_own_name() -> None:
    names, mapping = build_vocabulary(["scaffolding", "ladder"], {})
    assert names == ["scaffolding", "ladder"]
    assert mapping.tolist() == [0, 1]


def test_a_prompt_mapped_to_empty_is_scored_but_not_reported() -> None:
    """Context-only prompts must not become classes.

    The calibrated set depends on near-synonyms being present at inference time
    -- removing them changes the scores of the prompts that are reported -- but
    they are not themselves findings.
    """
    names, mapping = build_vocabulary(
        ["unfinished ceiling", "exposed concrete soffit", "bare concrete ceiling"],
        {
            "unfinished ceiling": "exposed slab",
            "exposed concrete soffit": "",
            "bare concrete ceiling": "",
        },
    )
    assert names == ["exposed slab"]
    assert mapping.tolist() == [0, -1, -1]


def test_display_class_order_follows_the_prompt_list() -> None:
    """Order fixes the colours, so it must come from the file, not a dict."""
    prompts = ["b prompt", "a prompt", "c prompt"]
    labels = {"b prompt": "beta", "a prompt": "alpha", "c prompt": "gamma"}
    names, _ = build_vocabulary(prompts, labels)
    assert names == ["beta", "alpha", "gamma"]


def test_an_empty_vocabulary_is_handled() -> None:
    names, mapping = build_vocabulary([], {})
    assert names == []
    assert mapping.tolist() == []


# --------------------------------------------------------------------------
# Duplicate suppression after collapsing
# --------------------------------------------------------------------------


def test_collapsed_synonyms_on_one_object_are_suppressed() -> None:
    """The reason collapsing needs a second NMS pass.

    Ultralytics suppresses per prompt, so one light fixture matched by two
    phrasings survives as two boxes. Mapped to one class they would be counted
    as two lights, which is the headline number on the dashboard.
    """
    boxes = np.array([[100.0, 100.0, 200.0, 140.0], [102.0, 98.0, 198.0, 142.0]])
    scores = np.array([0.42, 0.37])
    # Both now the same display class, having been "strip light" and "tube light".
    class_ids = np.array([0, 0], dtype=np.int32)

    kept = nms(boxes, scores, class_ids, 0.45, max_detections=300)
    assert len(kept) == 1
    assert kept[0] == 0, "the higher-scoring phrasing should survive"


def test_different_classes_that_overlap_are_both_kept() -> None:
    """A light fixture mounted on a ceiling is two findings, not one."""
    boxes = np.array([[100.0, 100.0, 200.0, 140.0], [102.0, 98.0, 198.0, 142.0]])
    scores = np.array([0.42, 0.37])
    class_ids = np.array([0, 1], dtype=np.int32)

    kept = nms(boxes, scores, class_ids, 0.45, max_detections=300)
    assert len(kept) == 2


# --------------------------------------------------------------------------
# The shipped registry
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> list[ModelSpec]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [ModelSpec.model_validate(entry) for entry in data]


def test_the_shipped_registry_is_valid(registry: list[ModelSpec]) -> None:
    names = [spec.name for spec in registry]
    assert "yolo11n" in names, "the default COCO model must remain available"
    assert "fitout-extended" in names
    assert "construction-calibrated" in names


def test_every_label_map_key_is_a_real_prompt(registry: list[ModelSpec]) -> None:
    """A typo in the map would silently leave a synonym as its own class."""
    for spec in registry:
        unknown = set(spec.label_map) - set(spec.prompts)
        assert not unknown, f"{spec.name} maps prompts it never asks for: {unknown}"


def test_every_threshold_names_a_real_display_class(registry: list[ModelSpec]) -> None:
    """A threshold on a class that cannot be produced would never apply."""
    for spec in registry:
        if not spec.prompts:
            continue
        names, _ = build_vocabulary(spec.prompts, spec.label_map)
        unknown = set(spec.class_thresholds) - set(names)
        assert not unknown, f"{spec.name} sets thresholds for absent classes: {unknown}"


def test_the_calibrated_model_reports_only_what_was_measured(
    registry: list[ModelSpec],
) -> None:
    """Its whole point: four evidence-backed classes and nothing else."""
    spec = next(s for s in registry if s.name == "construction-calibrated")
    names, mapping = build_vocabulary(spec.prompts, spec.label_map)

    assert sorted(names) == ["column", "exposed slab", "light", "partition"]
    assert (mapping < 0).sum() == 35, "the context prompts must still be asked"
    # Every reported class carries the threshold measured for it.
    assert set(spec.class_thresholds) == set(names)


def test_the_fitout_model_covers_the_requested_categories(
    registry: list[ModelSpec],
) -> None:
    spec = next(s for s in registry if s.name == "fitout-extended")
    names, _ = build_vocabulary(spec.prompts, spec.label_map)

    for expected in [
        "scaffolding",
        "wiring",
        "cable tray",
        "wall",
        "window",
        "glass partition",
        "helmet",
        "hi-vis vest",
        "duct",
        "pipe",
        "conduit",
        "sprinkler",
        "light",
        "suspended ceiling",
        "socket / switch",
        "electrical panel",
    ]:
        assert expected in names, f"{expected!r} missing from the fit-out vocabulary"


def test_no_display_class_is_also_a_bare_prompt_name(
    registry: list[ModelSpec],
) -> None:
    """Guards a subtle duplicate: a prompt left unmapped while a synonym of it
    maps to a class of the same name would produce that class twice."""
    for spec in registry:
        if not spec.prompts:
            continue
        names, _ = build_vocabulary(spec.prompts, spec.label_map)
        assert len(names) == len(set(names))
