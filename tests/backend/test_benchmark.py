"""Tests for graph vs flat query benchmark."""
import pytest

from backend.config import DATA_DIR
from backend.io.view_store import load_view_analysis
from backend.query.benchmark import (
    _leaf_match_candidate,
    compare_modes,
    simulate_graph_search,
    simulate_flat_search,
)
from backend.query.benchmark_reasoning import _extract_target


def test_flat_checks_more_views_than_graph():
    r = compare_modes("default", "take me to the projector")
    assert r["flat"]["views_checked"] >= r["graph"]["views_checked"]
    assert r["flat"]["input_tokens"] >= r["graph"]["input_tokens"]


def test_graph_finds_projector():
    g = simulate_graph_search("default", "take me to the projector")
    assert g.views_checked < 19
    assert g.llm_calls <= 20


def test_find_sofa_picks_usable_view():
    g = simulate_graph_search("default", "Find the sofa")
    assert g.found
    assert g.view_id is not None
    if not (DATA_DIR / "default" / "images" / f"{g.view_id}.png").is_file():
        pytest.skip("raw scene images are not distributed with the public code release")
    from backend.io.annotator import view_image_usable
    assert view_image_usable("default", g.view_id)


def test_find_column_uses_column_target_and_bbox():
    view = load_view_analysis(str(DATA_DIR / "default"), "v002")
    assert view is not None

    candidate = _leaf_match_candidate(
        "default",
        "leaf_lobby_corridor_lounge",
        "v002",
        "find the column",
        view.model_dump(),
    )

    assert _extract_target("find the column") == "column"
    assert candidate["matched_object"] == "column"
    assert candidate["bbox_2d"] is not None


def test_find_collumn_typo_normalizes_to_column():
    g = simulate_graph_search("default", "find a collumn")

    assert _extract_target("find a collumn") == "column"
    assert g.found
    assert g.view_id is not None


def test_flat_checks_all_views():
    f = simulate_flat_search("default", "find the bar")
    assert f.views_checked == 19
    assert f.llm_calls == 20  # 1 decomposition + 19 leaf checks
