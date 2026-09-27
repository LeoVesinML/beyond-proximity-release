"""Tests for captured-scene graph-vs-flat benchmark."""
from pathlib import Path

import pytest

from backend.query.graph_vs_flat import (
    DEFAULT_SCENE_MAP,
    compare_modes_for_query,
    filter_five_scene_queries,
    load_benchmark_queries,
    load_scene_bundle,
    run_five_scene_benchmark,
    simulate_graph_search,
)


ROOT = Path(__file__).resolve().parents[2]
SCENE_ROOT = ROOT / "backend" / "data" / "scenes"
BENCHMARK = ROOT / "docs" / "benchmarks" / "benchmark_queries_v1.json"
BENCHMARK_V2 = ROOT / "docs" / "benchmarks" / "benchmark_queries_v2.json"


def scene_assets_available(scene: Path) -> bool:
    view_count = len(list((scene / "views").glob("*.json")))
    return (
        view_count > 0
        and len(list((scene / "images").glob("*.png"))) >= view_count
        and len(list((scene / "depths").glob("*.npy"))) >= view_count
    )


CONFERENCE_ASSETS_AVAILABLE = scene_assets_available(
    SCENE_ROOT / "ConferenceHall-capture-pilot"
)
ALL_SCENE_ASSETS_AVAILABLE = all(
    scene_assets_available(SCENE_ROOT / path) for path in DEFAULT_SCENE_MAP.values()
)


@pytest.mark.skipif(
    not CONFERENCE_ASSETS_AVAILABLE,
    reason="Complete captured ConferenceHall RGB-D assets are not present",
)
def test_load_scene_bundle_has_views_and_tree():
    bundle = load_scene_bundle(SCENE_ROOT / "ConferenceHall-capture-pilot")
    assert bundle["views"]
    assert bundle["tree"]
    assert bundle["manifest"]["root_node_id"] == "root" or bundle["manifest"]["root_id"] == "root"


@pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmark_queries_v1.json missing")
def test_filter_five_scene_queries_returns_forty():
    queries = filter_five_scene_queries(load_benchmark_queries(BENCHMARK))
    assert len(queries) == 40
    assert {q["scene_id"] for q in queries} == set(DEFAULT_SCENE_MAP)


@pytest.mark.skipif(
    not CONFERENCE_ASSETS_AVAILABLE or not BENCHMARK.exists(),
    reason="Complete captured scene assets or benchmark are missing",
)
def test_graph_checks_fewer_views_than_flat_on_chair_query():
    bundle = load_scene_bundle(SCENE_ROOT / "ConferenceHall-capture-pilot")
    query = next(
        q for q in filter_five_scene_queries(load_benchmark_queries(BENCHMARK))
        if q["query_id"] == "q051"
    )
    row = compare_modes_for_query(bundle, query)
    assert row["flat"]["views_checked"] >= row["graph"]["views_checked"]
    assert row["flat"]["input_tokens"] >= row["graph"]["input_tokens"]


@pytest.mark.skipif(
    not CONFERENCE_ASSETS_AVAILABLE or not BENCHMARK.exists(),
    reason="Complete captured scene assets or benchmark are missing",
)
def test_graph_pruning_knobs_are_recorded():
    bundle = load_scene_bundle(SCENE_ROOT / "ConferenceHall-capture-pilot")
    query = next(
        q for q in filter_five_scene_queries(load_benchmark_queries(BENCHMARK))
        if q["query_id"] == "q051"
    )
    result = simulate_graph_search(
        bundle,
        query["query"],
        query_id=query["query_id"],
        benchmark_scene_id=query["scene_id"],
        branch_keep_ratio=1.0,
        fallback_child_limit=1,
        mode_suffix="_tight",
    )
    assert result.mode == "graph_lexical_tight"
    assert "branch_keep_ratio=1.0" in result.warnings[-1]


@pytest.mark.skipif(
    not ALL_SCENE_ASSETS_AVAILABLE or not BENCHMARK.exists(),
    reason="Complete assets for all five captured scenes or benchmark are missing",
)
def test_run_five_scene_benchmark_summary():
    run = run_five_scene_benchmark(
        benchmark_path=BENCHMARK,
        scene_root=SCENE_ROOT,
        include_affordance=True,
    )
    assert run["summary"]["query_count"] == 40
    assert run["summary"]["averages"]["flat_views_checked"] > 0
    assert "savings_views_pct" in run["summary"]["averages"]


@pytest.mark.skipif(
    not ALL_SCENE_ASSETS_AVAILABLE or not BENCHMARK_V2.exists(),
    reason="Complete assets for all five captured scenes or benchmark_queries_v2.json are missing",
)
def test_run_v2_benchmark_has_gt_quality_metrics():
    run = run_five_scene_benchmark(
        benchmark_path=BENCHMARK_V2,
        scene_root=SCENE_ROOT,
        include_affordance=True,
    )
    summary = run["summary"]
    assert summary["query_count"] == 150
    assert summary["gt_query_count"] == 125
    assert summary["quality"]["hit_at_1_graph"] is not None
    assert summary["quality"]["hit_at_3_flat"] is not None
    gt_rows = [row for row in run["queries"] if row["quality_graph"]["gt_available"]]
    assert len(gt_rows) == 125
