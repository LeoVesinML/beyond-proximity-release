import random

from scripts.run_matched_hierarchy_controls import (
    _disjoint_projection,
    _random_partition,
)
from scripts.run_matched_relation_ablation import _bootstrap


def test_matched_relation_bootstrap_is_paired_and_deterministic() -> None:
    rows = [
        {"with": True, "without": False},
        {"with": False, "without": False},
        {"with": True, "without": True},
        {"with": False, "without": True},
    ]

    first = _bootstrap(rows, "with", "without", seed=42)
    second = _bootstrap(rows, "with", "without", seed=42)

    assert first == second
    assert first["estimate"] == 0.0
    assert first["samples"] == 10_000


def test_random_partition_preserves_membership_and_requested_sizes() -> None:
    view_ids = {f"v{index}" for index in range(8)}
    zones = _random_partition(view_ids, [3, 3, 2], random.Random(7))

    memberships = [view_id for zone in zones for view_id in zone["view_ids"]]
    assert [len(zone["view_ids"]) for zone in zones] == [3, 3, 2]
    assert set(memberships) == view_ids
    assert len(memberships) == len(set(memberships))


def test_disjoint_projection_assigns_each_view_once() -> None:
    reference = [
        {"zone_id": "a", "view_ids": ["v1", "v2"], "query_reference_count": 2},
        {"zone_id": "b", "view_ids": ["v2", "v3"], "query_reference_count": 5},
    ]

    projected = _disjoint_projection(reference, {"v1", "v2", "v3", "v4"})
    memberships = [view_id for zone in projected for view_id in zone["view_ids"]]

    assert sorted(memberships) == ["v1", "v2", "v3", "v4"]
    assert len(memberships) == len(set(memberships))
    assert next(zone for zone in projected if zone["zone_id"] == "b")["view_ids"] == ["v2", "v3"]
    assert next(zone for zone in projected if zone["zone_id"] == "projected_orphan")["view_ids"] == ["v4"]
