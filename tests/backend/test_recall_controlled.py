"""Tests for recall-controlled hierarchical retrieval."""
from pathlib import Path

import pytest

from backend.query.graph_vs_flat import load_benchmark_queries, load_scene_bundle
from backend.query.recall_controlled import (
    RecallControlledConfig,
    paired_bootstrap_difference,
    prepare_scene,
    search_flat_prepared_scene,
    search_prepared_scene,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[2]
SCENE_ROOT = ROOT / "backend" / "data" / "scenes"
BENCHMARK = ROOT / "docs" / "benchmarks" / "benchmark_queries_v2.json"


def _scene_assets_available(scene: Path) -> bool:
    view_count = len(list((scene / "views").glob("*.json")))
    return view_count > 0 and (scene / "tree" / "manifest.json").is_file()


@pytest.mark.skipif(
    not _scene_assets_available(SCENE_ROOT / "ConferenceHall-capture-pilot")
    or not BENCHMARK.is_file(),
    reason="ConferenceHall index or v2 benchmark missing",
)
def test_recall_controlled_result_has_auditable_cost_and_quality():
    bundle = load_scene_bundle(SCENE_ROOT / "ConferenceHall-capture-pilot")
    scene = prepare_scene(bundle)
    query = next(
        row for row in load_benchmark_queries(BENCHMARK)
        if row["scene_id"] == "ConferenceHall" and row.get("expected_view_ids")
    )
    config = RecallControlledConfig(
        branch_mass_threshold=0.8,
        candidate_budget=6,
        safety_shortlist_size=2,
    )
    graph = search_prepared_scene(scene, query, config)
    flat = search_flat_prepared_scene(scene, query)

    assert graph["query_id"] == query["query_id"]
    assert graph["records_checked"] <= flat["records_checked"]
    assert graph["proxy_view_comparisons"] == len(scene.view_ids)
    assert graph["serialized_context_tokens"] >= 0
    assert graph["hit_at_1"] in {True, False}
    assert graph["hit_at_3"] in {True, False}
    assert graph["selected_branches"]


def test_statistics_are_deterministic_and_bounded():
    low, high = wilson_interval(8, 10)
    assert 0 <= low < 0.8 < high <= 1

    first = paired_bootstrap_difference([1.0, 0.0, 1.0], [0.0, 0.0, 1.0], samples=200)
    second = paired_bootstrap_difference([1.0, 0.0, 1.0], [0.0, 0.0, 1.0], samples=200)
    assert first == second
    assert first["estimate"] == pytest.approx(1 / 3, abs=1e-6)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="branch_mass_threshold"):
        RecallControlledConfig(branch_mass_threshold=0).validate()
