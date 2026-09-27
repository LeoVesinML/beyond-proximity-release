import csv

from scripts.run_ieee_hierarchy_scorer_grid import (
    _hierarchy_priority,
    _matched_rank,
    _quality,
    _write_csv,
)


def test_hierarchy_priority_orders_branches_then_fills_uncovered_views():
    scores = {"v1": 0.9, "v2": 0.8, "v3": 0.7, "v4": 0.6}
    zones = [
        {"zone_id": "low", "view_ids": ["v4", "v3"]},
        {"zone_id": "high", "view_ids": ["v2", "v1"]},
    ]

    assert _hierarchy_priority(scores, zones) == ["v1", "v2", "v3", "v4"]


def test_hierarchy_priority_deduplicates_overlap():
    scores = {"v1": 0.9, "v2": 0.8, "v3": 0.7}
    zones = [
        {"zone_id": "a", "view_ids": ["v1", "v2"]},
        {"zone_id": "b", "view_ids": ["v2", "v3"]},
    ]

    assert _hierarchy_priority(scores, zones) == ["v1", "v2", "v3"]


def test_matched_rank_recomputes_candidate_union_for_budget():
    scores = {"v1": 0.9, "v2": 0.8, "v3": 0.7, "v4": 0.6}
    zones = [
        {"zone_id": "top", "view_ids": ["v1", "v4"]},
        {"zone_id": "next", "view_ids": ["v2", "v3"]},
    ]

    assert _matched_rank(scores, zones, 2) == (["v1", "v4"], 2)
    assert _matched_rank(scores, zones, 3) == (["v1", "v2", "v3"], 4)


def test_quality_uses_rank_position_and_handles_unlabeled_query():
    assert _quality(["v2", "v1"], {"v1"}) == (False, True, 0.5)
    assert _quality(["v2"], set()) == (None, None, None)


def test_write_csv_accepts_heterogeneous_rows(tmp_path):
    target = tmp_path / "rows.csv"
    _write_csv(target, [{"method": "manual"}, {"method": "raw", "ci": {"low": -0.1}}])

    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["ci"] == ""
    assert rows[1]["ci"] == '{"low": -0.1}'
