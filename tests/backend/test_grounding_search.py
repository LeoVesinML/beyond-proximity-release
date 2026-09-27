"""Tests for object-level grounding search variants."""
from __future__ import annotations

import numpy as np

from backend.query.grounding import (
    GroundingCandidate,
    available_variants,
    build_candidates,
    parse_relation_query,
    relation_predicate,
    search_grounding,
    validate_bbox_3d,
)


def _object(label, bbox_2d, bbox_3d, attributes=None):
    return {
        "label": label,
        "bbox_2d": bbox_2d,
        "bbox_3d": bbox_3d,
        "attributes": attributes or [],
    }


def _box(center, size=(1.0, 1.0, 1.0)):
    return {"center": center, "size": size, "label": ""}


def _bundle():
    views = {
        "v1": {
            "view_id": "v1",
            "visible_objects": [
                _object("chair", [0.05, 0.3, 0.25, 0.7], _box([0.0, 0.0, 0.0]), ["near table"]),
                _object("table", [0.55, 0.3, 0.85, 0.8], _box([3.0, 0.0, 0.0])),
            ],
            "spatial_relations": ["chair is to the left of table"],
        },
        "v2": {
            "view_id": "v2",
            "visible_objects": [
                _object("floor lamp", [0.7, 0.1, 0.9, 0.8], _box([8.0, 1.0, 2.0])),
                _object("sofa", [0.2, 0.3, 0.65, 0.8], _box([6.0, 0.0, 2.0])),
            ],
        },
    }
    tree = {
        "root": {
            "node_id": "root",
            "node_type": "root",
            "name": "scene",
            "parent_id": None,
            "children_ids": ["zone_a", "zone_b"],
        },
        "zone_a": {
            "node_id": "zone_a",
            "node_type": "zone",
            "name": "dining area",
            "summary": "chairs and tables",
            "parent_id": "root",
            "children_ids": ["object_v1_001_chair", "object_v1_002_table"],
        },
        "zone_b": {
            "node_id": "zone_b",
            "node_type": "zone",
            "name": "lounge",
            "summary": "soft seating",
            "parent_id": "root",
            "children_ids": ["object_v2_001_floor_lamp", "object_v2_002_sofa"],
        },
        "object_v1_001_chair": {
            "node_id": "object_v1_001_chair",
            "node_type": "object",
            "name": "chair",
            "parent_id": "zone_a",
            "view_ids": ["v1"],
        },
        "object_v1_002_table": {
            "node_id": "object_v1_002_table",
            "node_type": "object",
            "name": "table",
            "parent_id": "zone_a",
            "view_ids": ["v1"],
        },
        "object_v2_001_floor_lamp": {
            "node_id": "object_v2_001_floor_lamp",
            "node_type": "object",
            "name": "floor lamp",
            "parent_id": "zone_b",
            "view_ids": ["v2"],
        },
        "object_v2_002_sofa": {
            "node_id": "object_v2_002_sofa",
            "node_type": "object",
            "name": "sofa",
            "parent_id": "zone_b",
            "view_ids": ["v2"],
        },
    }
    return {"views": views, "tree": tree}


def _candidate(label, center):
    return GroundingCandidate(
        object_id=label,
        node_id=None,
        view_id="v1",
        ancestor_node_ids=[],
        label=label,
        text=label,
        bbox_2d=None,
        bbox_3d=_box(center),
        relations=[],
    )


def test_relation_parser_and_predicates_cover_supported_axes():
    parsed = parse_relation_query("Find the chair to the left of the table")
    assert parsed.target == "chair"
    assert parsed.relation == "left"
    assert parsed.anchor == "table"

    origin = _candidate("anchor", [0.0, 0.0, 0.0])
    cases = {
        "left": [-2.0, 0.0, 0.0],
        "right": [2.0, 0.0, 0.0],
        "above": [0.0, 2.0, 0.0],
        "below": [0.0, -2.0, 0.0],
        "front": [0.0, 0.0, -2.0],
        "behind": [0.0, 0.0, 2.0],
        "near": [0.2, 0.0, 0.0],
    }
    for relation, center in cases.items():
        matched, detail = relation_predicate(_candidate(relation, center), origin, relation)
        assert matched is True, (relation, detail)


def test_dataset_world_boxes_do_not_fake_viewpoint_relative_axes():
    target = _candidate("chair", [-2.0, 0.0, 0.0])
    anchor = _candidate("table", [0.0, 0.0, 0.0])
    target.bbox_3d["coordinate_frame"] = "scannet_axis_aligned_mesh"
    anchor.bbox_3d["coordinate_frame"] = "scannet_axis_aligned_mesh"

    matched, detail = relation_predicate(target, anchor, "left")

    assert matched is None
    assert detail["reason"] == "viewpoint_axis_unavailable"


def test_relation_parser_covers_public_benchmark_distance_and_between_phrases():
    closest = parse_relation_query(
        "select the table that is closer to the kitchen cabinets"
    )
    farthest = parse_relation_query(
        "select the trash can that is far from the kitchen cabinets"
    )
    between = parse_relation_query(
        "choose the picture that is in the center of the toilet paper and the tv"
    )
    above = parse_relation_query(
        "choose the dispenser that is on top of the other dispenser"
    )

    assert (closest.target, closest.relation, closest.anchor) == (
        "table",
        "closest",
        "kitchen cabinets",
    )
    assert (farthest.target, farthest.relation, farthest.anchor) == (
        "trash can",
        "farthest",
        "kitchen cabinets",
    )
    assert (
        between.target,
        between.relation,
        between.anchor,
        between.anchor_secondary,
    ) == ("picture", "between", "toilet paper", "tv")
    assert (above.target, above.relation, above.anchor) == (
        "dispenser",
        "above",
        "other dispenser",
    )


def test_relation_reranking_prefers_matching_target():
    result = search_grounding(
        _bundle(),
        "Find the chair left of the table",
        variant="flat_lexical",
    )
    assert result.status == "ok"
    assert result.candidates[0].label == "chair"
    assert result.candidates[0].score_details["relation_score"] == 1.0
    assert result.candidates[0].score_details["relation_source"] in {
        "spatial_relation_string",
        "bbox_3d",
    }
    table = next(candidate for candidate in result.candidates if candidate.label == "table")
    assert table.score_details["relation_score"] == 0.0
    assert result.metadata["relation_audit"]["applied_candidate_count"] == 1


def test_relation_cannot_override_stronger_label_outside_score_margin():
    bundle = {
        "views": {
            "v1": {
                "view_id": "v1",
                "visible_objects": [
                    _object("chair", None, _box([2.0, 0.0, 0.0])),
                    _object("red chair", None, _box([-2.0, 0.0, 0.0])),
                    _object("table", None, _box([0.0, 0.0, 0.0])),
                ],
            }
        },
        "tree": {},
    }

    result = search_grounding(
        bundle,
        "Find the chair left of the table",
        variant="flat_lexical",
    )

    assert result.candidates[0].label == "chair"
    red = next(candidate for candidate in result.candidates if candidate.label == "red chair")
    assert red.score_details["relation_score"] == 1.0
    assert red.score_details["relation_score_margin_eligible"] is False
    assert red.score_details["relation_applied"] is False


def test_ambiguous_anchor_geometry_is_below_relation_confidence_gate():
    bundle = {
        "views": {
            "v1": {
                "view_id": "v1",
                "visible_objects": [
                    _object("chair", None, _box([-2.0, 0.0, 0.0])),
                    _object("table", None, _box([0.0, 0.0, 0.0])),
                    _object("table", None, _box([4.0, 0.0, 0.0])),
                ],
            }
        },
        "tree": {},
    }

    result = search_grounding(
        bundle,
        "Find the chair left of the table",
        variant="flat_lexical",
    )

    chair = next(candidate for candidate in result.candidates if candidate.label == "chair")
    assert chair.score_details["anchor_evidence"]["status"] == "ambiguous"
    assert chair.score_details["relation_confidence"] < 0.65
    assert chair.score_details["relation_applied"] is False
    assert result.metadata["relation_unresolved"] is True


def test_unapplied_relation_confidence_cannot_break_lexical_ties():
    bundle = {
        "views": {
            "v1": {
                "view_id": "v1",
                "visible_objects": [
                    {
                        **_object("chair", None, _box([2.0, 0.0, 0.0])),
                        "object_id": "a_lexical_first",
                    },
                    {
                        **_object("chair", None, _box([-2.0, 0.0, 0.0])),
                        "object_id": "z_low_confidence_relation",
                    },
                    {
                        **_object("table", None, _box([0.0, 0.0, 0.0])),
                        "object_id": "anchor_1",
                    },
                    {
                        **_object("table", None, _box([4.0, 0.0, 0.0])),
                        "object_id": "anchor_2",
                    },
                ],
            }
        },
        "tree": {},
    }

    result = search_grounding(
        bundle,
        "Find the chair left of the table",
        variant="flat_lexical",
    )

    assert result.candidates[0].object_id == "a_lexical_first"
    low_confidence = next(
        candidate
        for candidate in result.candidates
        if candidate.object_id == "z_low_confidence_relation"
    )
    assert low_confidence.score_details["relation_confidence"] < 0.65
    assert low_confidence.score_details["relation_applied"] is False


def test_relative_distance_relations_choose_closest_and_farthest_target():
    bundle = {
        "views": {
            "v1": {
                "view_id": "v1",
                "visible_objects": [
                    {**_object("chair", None, _box([1.0, 0.0, 0.0])), "object_id": "near"},
                    {**_object("chair", None, _box([5.0, 0.0, 0.0])), "object_id": "far"},
                    {**_object("table", None, _box([0.0, 0.0, 0.0])), "object_id": "anchor"},
                ],
            }
        },
        "tree": {},
    }

    closest = search_grounding(
        bundle,
        "select the chair that is closest to the table",
        variant="flat_lexical",
    )
    farthest = search_grounding(
        bundle,
        "select the chair that is farthest from the table",
        variant="flat_lexical",
    )

    assert closest.candidates[0].object_id == "near"
    assert farthest.candidates[0].object_id == "far"
    assert closest.candidates[0].score_details["relation_applied"] is True
    assert farthest.candidates[0].score_details["relation_applied"] is True


def test_between_relation_requires_two_valid_anchors():
    bundle = {
        "views": {
            "v1": {
                "view_id": "v1",
                "visible_objects": [
                    _object("chair", None, _box([5.0, 0.0, 0.0])),
                    _object("table", None, _box([0.0, 0.0, 0.0])),
                    _object("sofa", None, _box([10.0, 0.0, 0.0])),
                ],
            }
        },
        "tree": {},
    }

    result = search_grounding(
        bundle,
        "find the chair between the table and the sofa",
        variant="flat_lexical",
    )

    assert result.candidates[0].label == "chair"
    assert result.candidates[0].score_details["relation_applied"] is True
    assert result.metadata["relation_unresolved"] is False


def test_graph_fallback_expands_after_low_confidence_pruning():
    bundle = _bundle()
    # No branch summary says "lamp", so deterministic graph pruning starts in
    # zone_a; fallback must expand and recover the exact-label object in zone_b.
    result = search_grounding(bundle, "Find the floor lamp", variant="graph_fallback")
    assert result.fallback_triggered is True
    assert "low_confidence" in result.fallback_reasons
    assert result.searched_candidate_count == result.candidate_count
    assert result.candidates[0].label == "floor lamp"

    no_fallback = search_grounding(
        bundle,
        "Find the floor lamp",
        variant="graph_fallback",
        no_fallback=True,
    )
    assert no_fallback.fallback_triggered is False
    assert no_fallback.candidates[0].label != "floor lamp"


def test_bbox_quality_gate_rejects_invalid_values_and_scene_overflow():
    assert validate_bbox_3d(_box([0, 0, 0])) == (True, "valid")
    assert validate_bbox_3d(_box([np.nan, 0, 0]))[1] == "non_finite_bbox_3d"
    assert validate_bbox_3d(_box([0, 0, 0], [1, 0, 1]))[1] == "non_positive_bbox_3d_extent"
    valid, reason = validate_bbox_3d(
        _box([9, 0, 0], [4, 1, 1]),
        {"min": [-10, -10, -10], "max": [10, 10, 10]},
    )
    assert valid is False
    assert reason == "bbox_3d_outside_scene_bounds"

    bundle = _bundle()
    bundle["views"]["v1"]["visible_objects"][0]["bbox_3d"] = _box([np.inf, 0, 0])
    bundle["views"]["v1"]["visible_objects"][0]["bbox_2d"] = None
    result = search_grounding(
        bundle,
        "chair left of table",
        variant="graph_fallback",
    )
    assert result.fallback_triggered is True
    assert "bbox_quality_rejected" in result.fallback_reasons


class DeterministicEmbedder:
    model_name = "deterministic-test-v1"

    def encode(self, texts, **_kwargs):
        vectors = {
            "reading light": [1.0, 0.0, 0.0],
            "floor lamp": [0.95, 0.05, 0.0],
            "chair near table chair is to the left of table": [0.0, 1.0, 0.0],
            "table chair is to the left of table": [0.0, 0.9, 0.1],
            "sofa": [0.1, 0.1, 0.8],
        }
        return [vectors.get(text, [0.0, 0.0, 1.0]) for text in texts]


def test_embedding_ranking_uses_injected_backend_and_reports_metadata():
    result = search_grounding(
        _bundle(),
        "reading light",
        variant="flat_embedding",
        embedder=DeterministicEmbedder(),
    )
    assert result.status == "ok"
    assert result.candidates[0].label == "floor lamp"
    assert result.metadata["embedding"] == {
        "backend": "injected",
        "model": "deterministic-test-v1",
        "cache_folder": None,
        "device": "cpu",
    }


def test_variants_use_same_canonical_input_and_expose_ablations():
    bundle = _bundle()
    canonical = build_candidates(bundle["views"], bundle["tree"])
    graph = search_grounding(bundle, "chair", variant="graph")
    flat = search_grounding(bundle, "chair", variant="flat_lexical")
    embedding = search_grounding(
        bundle,
        "chair",
        variant="flat_embedding",
        embedder=DeterministicEmbedder(),
    )
    expected_ids = [candidate.object_id for candidate in canonical]
    assert graph.metadata["same_input_candidate_ids"] == expected_ids
    assert flat.metadata["same_input_candidate_ids"] == expected_ids
    assert embedding.metadata["same_input_candidate_ids"] == expected_ids
    assert set(available_variants()) == {
        "graph",
        "graph_fallback",
        "flat_lexical",
        "flat_embedding",
    }


def test_official_object_id_and_object_root_branch_are_preserved():
    bundle = {
        "views": {
            "v1": {
                "view_id": "v1",
                "visible_objects": [
                    _object("cabinet", None, {**_box([0, 0, 0]), "object_id": "42"})
                ],
            }
        },
        "tree": {
            "root": {
                "node_id": "root",
                "node_type": "root",
                "children_ids": ["object_v1_001_cabinet"],
            },
            "object_v1_001_cabinet": {
                "node_id": "object_v1_001_cabinet",
                "node_type": "object",
                "name": "cabinet",
                "parent_id": "root",
                "view_ids": ["v1"],
                "bbox_3d": {**_box([0, 0, 0]), "object_id": "42"},
            },
        },
    }
    candidates = build_candidates(bundle["views"], bundle["tree"])
    assert candidates[0].object_id == "42"
    assert candidates[0].node_id == "object_v1_001_cabinet"

    result = search_grounding(bundle, "cabinet", variant="graph")
    assert result.fallback_triggered is False
    assert result.searched_candidate_count == 1
    assert result.candidates[0].object_id == "42"


def test_embedding_unavailable_is_explicitly_blocked(monkeypatch):
    def unavailable(_model_name, _cache_folder):
        raise ImportError("sentence-transformers is intentionally absent")

    monkeypatch.setattr("backend.query.grounding._load_sentence_transformer", unavailable)
    result = search_grounding(_bundle(), "chair", variant="flat_embedding")
    assert result.status == "blocked"
    assert result.candidates == []
    assert "ImportError" in result.metadata["embedding"]["blocked_reason"]
