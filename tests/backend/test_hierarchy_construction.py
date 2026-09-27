import pytest

from scripts.evaluate_hierarchy_construction import (
    _agent_zones,
    _tree_with_zones,
    _zone_assignment_metrics,
)


def test_zone_assignment_metrics_detect_orphans_duplicates_and_pair_agreement():
    reference = [
        {"zone_id": "a", "view_ids": ["v1", "v2"]},
        {"zone_id": "b", "view_ids": ["v3"]},
    ]
    predicted = [
        {"zone_id": "x", "view_ids": ["v1", "v2"]},
        {"zone_id": "y", "view_ids": ["v3"]},
    ]

    metrics = _zone_assignment_metrics(
        predicted,
        reference,
        {"v1", "v2", "v3"},
    )

    assert metrics["pairwise_f1"] == 1.0
    assert metrics["weighted_best_zone_jaccard"] == 1.0
    assert metrics["orphan_view_ids"] == []
    assert metrics["duplicate_view_ids"] == []


def test_agent_zones_require_direct_mcp_provenance_and_exact_coverage():
    decisions = {
        "constraints": {
            "direct_mcp_calls": True,
            "mcp_tool_call_count": 5,
        },
        "scenes": [
            {
                "scene_id": "scene",
                "zones": [
                    {
                        "zone_id": "z",
                        "zone_name": "Zone",
                        "view_ids": ["v1", "v2"],
                    }
                ],
            }
        ],
    }

    assert _agent_zones(decisions, "scene", {"v1", "v2"})[0]["zone_id"] == "z"

    decisions["constraints"]["direct_mcp_calls"] = False
    with pytest.raises(ValueError, match="direct_mcp_calls"):
        _agent_zones(decisions, "scene", {"v1", "v2"})


def test_tree_with_zones_reparents_semantic_nodes_without_mutating_source():
    source = {
        "scene_id": "scene",
        "views": {"v1": {}, "v2": {}},
        "manifest": {"root_node_id": "root"},
        "tree": {
            "root": {
                "node_id": "root",
                "node_type": "root",
                "children_ids": ["manual", "object"],
            },
            "manual": {
                "node_id": "manual",
                "node_type": "zone",
                "children_ids": ["object"],
            },
            "object": {
                "node_id": "object",
                "node_type": "object",
                "parent_id": "manual",
                "view_ids": ["v1"],
                "children_ids": [],
            },
        },
    }
    zones = [
        {
            "zone_id": "new_zone",
            "zone_name": "New zone",
            "view_ids": ["v1", "v2"],
            "rationale": "fixture",
        }
    ]

    rebuilt = _tree_with_zones(source, zones, variant="fixture")

    assert rebuilt["tree"]["object"]["parent_id"] == "new_zone"
    assert rebuilt["tree"]["new_zone"]["children_ids"] == ["object"]
    assert source["tree"]["object"]["parent_id"] == "manual"
