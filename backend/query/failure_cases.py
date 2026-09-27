"""Curated failure / edge-case queries with room-scoped ground truth.

These cases demonstrate where flat search returns the wrong room while graph
traversal scopes to the correct zone first.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from backend.query.benchmark import simulate_graph_search, simulate_flat_search
from backend.query.timing import estimate_timing
from backend.tree.storage import load_all_nodes


@dataclass
class FailureCase:
    query: str
    room_label: str
    positive_view_ids: List[str]
    wrong_room_view_ids: List[str]
    expected_leaf_nodes: List[str]
    why_it_matters: str
    category: str = "room_disambiguation"


FAILURE_CASES: List[FailureCase] = [
    FailureCase(
        query="where is the screen in the conference room",
        room_label="Conference room / stage (presentation end)",
        positive_view_ids=["v017", "v018"],
        wrong_room_view_ids=["v012", "v015"],
        expected_leaf_nodes=["leaf_ballroom_stage_front", "leaf_stage"],
        why_it_matters=(
            "Multiple projection screens exist in the hall. Flat search returns the "
            "first keyword match (often v012 — ballroom floor through doors) without "
            "respecting the room constraint. Graph traversal enters the stage/screen "
            "leaf and confirms v017/v018."
        ),
        category="room_disambiguation",
    ),
    FailureCase(
        query="where is the screen in the conference room located",
        room_label="Conference room / stage (presentation end)",
        positive_view_ids=["v017", "v018"],
        wrong_room_view_ids=["v012", "v015"],
        expected_leaf_nodes=["leaf_ballroom_stage_front", "leaf_stage"],
        why_it_matters=(
            "Same room constraint with natural phrasing ('located'). Tests whether "
            "filler words break matching and whether graph still scopes correctly."
        ),
        category="room_disambiguation",
    ),
    FailureCase(
        query="where is the projection screen in the ballroom",
        room_label="Ballroom (any zone)",
        positive_view_ids=["v012", "v015", "v017", "v018"],
        wrong_room_view_ids=["v001", "v002"],
        expected_leaf_nodes=["leaf_ballroom_main_floor", "leaf_ballroom_stage_front"],
        why_it_matters=(
            "Ballroom-wide query — both floor and stage screens are valid. Graph should "
            "stay inside zone_ballroom; flat may still work but checks all 19 views."
        ),
        category="room_scoped_search",
    ),
    FailureCase(
        query="find the projector in the ballroom",
        room_label="Ballroom AV (side wall projector)",
        positive_view_ids=["v013"],
        wrong_room_view_ids=["v001", "v002", "v009"],
        expected_leaf_nodes=["leaf_ballroom_main_floor"],
        why_it_matters=(
            "Projector on AV stand (v013) vs projection screens elsewhere. Room + object "
            "disambiguation — flat may confuse screen views with projector view."
        ),
        category="room_disambiguation",
    ),
    FailureCase(
        query="where is the exit sign in the lobby",
        room_label="Lobby / reception",
        positive_view_ids=["v008", "v009", "v011"],
        wrong_room_view_ids=["v012", "v017", "v018"],
        expected_leaf_nodes=[
            "leaf_lobby_corridor_lounge",
            "leaf_lobby_foyer_prefunction",
            "leaf_lobby_bar_reception",
        ],
        why_it_matters=(
            "Exit signs appear throughout the venue. Lobby-specific query should not "
            "return ballroom or stage signs that flat search hits when scanning all views."
        ),
        category="room_disambiguation",
    ),
]


def _view_in_leaves(view_id: Optional[str], leaf_ids: List[str], scene_dir: str) -> bool:
    if not view_id:
        return False
    nodes = load_all_nodes(scene_dir)
    for lid in leaf_ids:
        node = nodes.get(lid)
        if node and view_id in node.view_ids:
            return True
    return False


def _classify_result(
    case: FailureCase,
    *,
    found: bool,
    view_id: Optional[str],
    scene_dir: str,
) -> str:
    if not found:
        return "miss"
    if view_id in case.positive_view_ids:
        return "correct_room"
    if view_id in case.wrong_room_view_ids:
        return "wrong_room"
    if _view_in_leaves(view_id, case.expected_leaf_nodes, scene_dir):
        return "correct_zone"
    return "other"


def run_failure_suite(scene_id: str = "default") -> Dict[str, Any]:
    """Run graph + flat on curated failure cases; classify room accuracy."""
    from backend.config import DATA_DIR

    scene_dir = str(DATA_DIR / scene_id)
    results: List[Dict[str, Any]] = []

    for case in FAILURE_CASES:
        g = simulate_graph_search(scene_id, case.query)
        f = simulate_flat_search(scene_id, case.query)

        g_cls = _classify_result(case, found=g.found, view_id=g.view_id, scene_dir=scene_dir)
        f_cls = _classify_result(case, found=f.found, view_id=f.view_id, scene_dir=scene_dir)

        graph_wins = (
            g_cls in ("correct_room", "correct_zone")
            and f_cls in ("wrong_room", "other", "miss")
        ) or (
            g_cls == "correct_room" and f_cls != "correct_room"
        )

        results.append({
            "case": asdict(case),
            "graph": {
                **g.__dict__,
                "classification": g_cls,
                "timing": estimate_timing(g.elapsed_ms, g.llm_calls, g.input_tokens).to_dict(),
            },
            "flat": {
                **f.__dict__,
                "classification": f_cls,
                "timing": estimate_timing(f.elapsed_ms, f.llm_calls, f.input_tokens).to_dict(),
            },
            "graph_wins_room": graph_wins,
            "flat_wrong_room": f_cls == "wrong_room",
            "demo_highlight": case.category == "room_disambiguation" and graph_wins,
        })

    n = len(results)
    graph_room_ok = sum(
        1 for r in results if r["graph"]["classification"] in ("correct_room", "correct_zone")
    )
    flat_room_ok = sum(
        1 for r in results if r["flat"]["classification"] in ("correct_room", "correct_zone")
    )
    flat_wrong = sum(1 for r in results if r["flat_wrong_room"])
    graph_wins = sum(1 for r in results if r["graph_wins_room"])

    return {
        "scene_id": scene_id,
        "cases_run": n,
        "summary": {
            "graph_room_correct": graph_room_ok,
            "flat_room_correct": flat_room_ok,
            "flat_wrong_room_count": flat_wrong,
            "graph_wins_room_count": graph_wins,
            "highlight_case": next(
                (r["case"]["query"] for r in results if r["demo_highlight"]),
                None,
            ),
        },
        "results": results,
    }
