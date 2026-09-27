"""Benchmark graph vs flat query search — token and call accounting.

Both modes use the same prompt builders as the production pipeline (§7).
Graph mode simulates LLM branch pruning via keyword overlap on child summaries.
Flat mode checks every analyzed view (no tree pruning).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from backend.config import DATA_DIR
from backend.io.annotator import view_image_usable
from backend.io.view_store import list_view_ids, load_view_analysis
from backend.query.pipeline import (
    build_decomposition_prompt,
    build_leaf_confirmation_for_view,
    build_traversal_step,
    rank_leaf_results,
)
from backend.query.benchmark_reasoning import _extract_target
from backend.query.timing import attach_timing
from backend.query.tokens import count_tokens
from backend.tree.nodes import NodeType
from backend.tree.storage import get_root_node_id, load_all_nodes, load_node

_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "to", "is", "in", "on", "at", "for", "of", "and", "or",
    "find", "where", "what", "which", "how", "why", "me", "my", "we", "can",
    "could", "please", "tell", "show", "get", "locate", "look", "there", "all",
    "take", "go", "located", "location", "situated", "positioned",
})

_QUERY_ALIASES = {
    "collumn": "column",
    "collumns": "columns",
    "pillar": "column",
    "pillars": "columns",
}


def _keywords(query: str) -> List[str]:
    tokens = [
        _QUERY_ALIASES.get(t, t)
        for t in re.split(r"[^\w]+", query.lower())
        if len(t) >= 2
    ]
    kws = [t for t in tokens if t not in _QUERY_STOPWORDS]
    return kws or tokens


def _score_text(text: str, query: str) -> float:
    kws = _keywords(query)
    if not kws:
        return 0.0
    hay = text.lower()
    return sum(1 for k in kws if k in hay) / len(kws)


def _node_for_view(scene_dir: str, view_id: str) -> Optional[str]:
    for node in load_all_nodes(scene_dir).values():
        if view_id in node.view_ids:
            return node.node_id
    return None


@dataclass
class BenchmarkResult:
    mode: str
    query: str
    found: bool
    view_id: Optional[str]
    llm_calls: int = 0
    input_tokens: int = 0
    views_checked: int = 0
    views_checked_ids: List[str] = field(default_factory=list)
    nodes_visited: int = 0
    traversal_steps: int = 0
    elapsed_ms: float = 0.0
    path: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def _add_prompt(metrics: Dict[str, Any], prompt: str) -> None:
    metrics["llm_calls"] += 1
    metrics["input_tokens"] += count_tokens(prompt)


def _result_dict(result: BenchmarkResult) -> Dict[str, Any]:
    d = result.__dict__.copy()
    attach_timing(d)
    return d


def get_graph_stats(scene_id: str) -> Dict[str, Any]:
    """Tree size metrics for scalability reporting."""
    scene_dir = str(DATA_DIR / scene_id)
    nodes = load_all_nodes(scene_dir)
    if not nodes:
        return {"has_tree": False}

    by_type: Dict[str, int] = {}
    max_depth = 0
    views_in_tree: Set[str] = set()
    for n in nodes.values():
        by_type[n.node_type.value] = by_type.get(n.node_type.value, 0) + 1
        max_depth = max(max_depth, n.depth)
        views_in_tree.update(n.view_ids)

    return {
        "has_tree": True,
        "node_count": len(nodes),
        "max_depth": max_depth,
        "root_id": get_root_node_id(scene_dir),
        "nodes_by_type": by_type,
        "leaf_count": by_type.get(NodeType.LEAF.value, 0),
        "zone_count": by_type.get(NodeType.ZONE.value, 0),
        "views_in_tree": len(views_in_tree),
    }


def simulate_graph_search(
    scene_id: str,
    query: str,
    structured_plan: Optional[dict] = None,
    *,
    descend_override: Optional[Dict[str, List[str]]] = None,
    active_view_ids: Optional[Set[str]] = None,
) -> BenchmarkResult:
    """Graph search: decomposition → pruned traversal → leaf checks only on visited leaves."""
    scene_dir = str(DATA_DIR / scene_id)
    plan = structured_plan or {}
    t0 = time.perf_counter()

    metrics: Dict[str, Any] = {
        "llm_calls": 0,
        "input_tokens": 0,
        "views_checked": 0,
        "nodes_visited": 0,
        "traversal_steps": 0,
    }
    checked_ids: List[str] = []
    path: List[str] = []
    found_view: Optional[str] = None
    found = False

    dec = build_decomposition_prompt(query)
    _add_prompt(metrics, dec["prompt"])

    root_id = get_root_node_id(scene_dir)
    if root_id is None:
        return BenchmarkResult(
            mode="graph",
            query=query,
            found=False,
            view_id=None,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            details={"error": "no tree"},
        )

    stack = [root_id]
    visited: Set[str] = set()

    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        metrics["nodes_visited"] += 1
        path.append(node_id)

        step = build_traversal_step(scene_id, node_id, query, plan)
        if "error" in step:
            continue

        if step.get("is_leaf"):
            view_ids = step.get("view_ids", [])
            if active_view_ids is not None:
                view_ids = [v for v in view_ids if v in active_view_ids]
            leaf_candidates: List[dict] = []
            for vid in view_ids:
                leaf = build_leaf_confirmation_for_view(scene_id, node_id, vid, query, plan)
                if "error" in leaf:
                    continue
                _add_prompt(metrics, leaf["prompt"])
                metrics["views_checked"] += 1
                checked_ids.append(vid)
                v = load_view_analysis(scene_dir, vid)
                if v and _view_matches(v.model_dump(), query):
                    leaf_candidates.append(
                        _leaf_match_candidate(scene_id, node_id, vid, query, v.model_dump())
                    )
            best = _pick_best_leaf_match(scene_id, leaf_candidates)
            if best:
                found = True
                found_view = best["view_id"]
                break
            continue

        _add_prompt(metrics, step["prompt"])
        metrics["traversal_steps"] += 1

        children = step.get("children", [])
        if descend_override and node_id in descend_override:
            chosen_ids = set(descend_override[node_id])
            to_visit = [c for c in children if c["node_id"] in chosen_ids]
        else:
            scored = [(c, _score_text(f"{c.get('name', '')} {c.get('summary', '')}", query)) for c in children]
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored and scored[0][1] > 0:
                top = scored[0][1]
                to_visit = [c for c, s in scored if s >= top * 0.5]
            else:
                to_visit = children  # no signal — visit all (worst case)

        for child in reversed(to_visit):
            stack.append(child["node_id"])

    elapsed = (time.perf_counter() - t0) * 1000
    return BenchmarkResult(
        mode="graph",
        query=query,
        found=found,
        view_id=found_view,
        llm_calls=metrics["llm_calls"],
        input_tokens=metrics["input_tokens"],
        views_checked=metrics["views_checked"],
        views_checked_ids=checked_ids,
        nodes_visited=metrics["nodes_visited"],
        traversal_steps=metrics["traversal_steps"],
        elapsed_ms=elapsed,
        path=path,
    )


def simulate_flat_search(
    scene_id: str,
    query: str,
    structured_plan: Optional[dict] = None,
    *,
    active_view_ids: Optional[Set[str]] = None,
) -> BenchmarkResult:
    """Flat search: decomposition → leaf confirmation on EVERY view (no graph pruning)."""
    scene_dir = str(DATA_DIR / scene_id)
    plan = structured_plan or {}
    t0 = time.perf_counter()

    llm_calls = 0
    input_tokens = 0
    views_checked = 0
    checked_ids: List[str] = []
    found_view: Optional[str] = None
    found = False

    dec = build_decomposition_prompt(query)
    llm_calls += 1
    input_tokens += count_tokens(dec["prompt"])

    view_ids = list_view_ids(scene_dir)
    if active_view_ids is not None:
        view_ids = [v for v in view_ids if v in active_view_ids]

    flat_candidates: List[dict] = []
    for vid in view_ids:
        node_id = _node_for_view(scene_dir, vid)
        if node_id is None:
            continue
        leaf = build_leaf_confirmation_for_view(scene_id, node_id, vid, query, plan)
        if "error" in leaf:
            continue
        llm_calls += 1
        input_tokens += count_tokens(leaf["prompt"])
        views_checked += 1
        checked_ids.append(vid)
        v = load_view_analysis(scene_dir, vid)
        if v and _view_matches(v.model_dump(), query):
            flat_candidates.append(
                _leaf_match_candidate(scene_id, node_id, vid, query, v.model_dump())
            )

    best = _pick_best_leaf_match(scene_id, flat_candidates)
    if best:
        found = True
        found_view = best["view_id"]

    elapsed = (time.perf_counter() - t0) * 1000
    return BenchmarkResult(
        mode="flat",
        query=query,
        found=found,
        view_id=found_view,
        llm_calls=llm_calls,
        input_tokens=input_tokens,
        views_checked=views_checked,
        views_checked_ids=checked_ids,
        nodes_visited=0,
        traversal_steps=0,
        elapsed_ms=elapsed,
        path=["all_views"],
    )


def _view_matches(view_dict: dict, query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    haystack = json.dumps(view_dict, ensure_ascii=False).lower()
    if q in haystack:
        return True
    kws = _keywords(query)
    return bool(kws) and all(k in haystack for k in kws)


def _target_object_in_view(view_dict: dict, query: str) -> Optional[dict]:
    """Best-matching object entry for the query target, if any."""
    target = _extract_target(query).lower()
    best: Optional[dict] = None
    for obj in view_dict.get("objects", []):
        label = obj.get("label", "").lower()
        if target in label or label in target or target in label.replace("_", " "):
            conf = float(obj.get("confidence", 0))
            if best is None or conf > float(best.get("confidence", 0)):
                best = obj
    return best


def _leaf_match_candidate(
    scene_id: str,
    node_id: str,
    view_id: str,
    query: str,
    view_dict: dict,
) -> dict:
    obj = _target_object_in_view(view_dict, query)
    conf = float(obj.get("confidence", 0)) if obj else 0.0
    confidence = "high" if conf >= 0.85 else "medium" if conf >= 0.7 else "low"
    return {
        "view_id": view_id,
        "node_id": node_id,
        "found": True,
        "confidence": confidence,
        "bbox_2d": obj.get("bbox_2d") if obj else None,
        "matched_object": (obj or {}).get("label") or _extract_target(query),
        "explanation": view_dict.get("scene_summary", ""),
    }


def _pick_best_leaf_match(scene_id: str, candidates: List[dict]) -> Optional[dict]:
    """Prefer views with real PNGs and strong bbox scores for Query Flow display."""
    if not candidates:
        return None
    usable = [c for c in candidates if view_image_usable(scene_id, c["view_id"])]
    pool = usable if usable else candidates
    ranked = rank_leaf_results(pool, scene_id)
    return ranked[0] if ranked else None


def replay_graph_from_session(session_path: Path) -> Optional[Dict[str, List[str]]]:
    """Extract descend_into map from a saved query session JSON."""
    if not session_path.exists():
        return None
    with open(session_path) as fh:
        data = json.load(fh)
    override: Dict[str, List[str]] = {}
    for step in data.get("steps", []):
        if step.get("step_type") == "traversal" and step.get("descend_into"):
            override[step["node_id"]] = step["descend_into"]
    return override or None


def compare_modes(
    scene_id: str,
    query: str,
    *,
    session_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run graph + flat for one query and return side-by-side metrics."""
    descend_override = replay_graph_from_session(session_path) if session_path else None
    graph = simulate_graph_search(scene_id, query, descend_override=descend_override)
    flat = simulate_flat_search(scene_id, query)

    total_views = len(list_view_ids(str(DATA_DIR / scene_id)))

    def _times_less(flat_val: float, graph_val: float) -> Optional[float]:
        if graph_val <= 0:
            return None
        return round(flat_val / graph_val, 1)

    return {
        "query": query,
        "total_views": total_views,
        "graph": _result_dict(graph),
        "flat": _result_dict(flat),
        "savings": {
            "tokens_pct": round(100 * (1 - graph.input_tokens / flat.input_tokens), 1) if flat.input_tokens else 0,
            "calls_pct": round(100 * (1 - graph.llm_calls / flat.llm_calls), 1) if flat.llm_calls else 0,
            "views_pct": round(100 * (1 - graph.views_checked / flat.views_checked), 1) if flat.views_checked else 0,
            "tokens_times_less": _times_less(flat.input_tokens, graph.input_tokens),
            "calls_times_less": _times_less(flat.llm_calls, graph.llm_calls),
            "views_times_faster": _times_less(flat.views_checked, graph.views_checked),
        },
        "graph_wins_on_tokens": graph.input_tokens < flat.input_tokens,
        "graph_wins_on_calls": graph.llm_calls < flat.llm_calls,
        "graph_wins_on_views": graph.views_checked < flat.views_checked,
    }


DEFAULT_BENCHMARK_QUERIES = [
    "take me to the projector",
    "find the bar",
    "find the red sofa",
    "where is the exit sign?",
    "find the piano",
]


def collect_queries_from_logs(scene_id: str) -> List[str]:
    """Unique queries from saved session logs + defaults."""
    seen: Set[str] = set()
    out: List[str] = []
    for q in DEFAULT_BENCHMARK_QUERIES:
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    log_dir = DATA_DIR / scene_id / "queries"
    if log_dir.exists():
        for p in sorted(log_dir.glob("*.json")):
            try:
                with open(p) as fh:
                    data = json.load(fh)
                q = (data.get("original_query") or "").strip()
                if q and q.lower() not in seen:
                    seen.add(q.lower())
                    out.append(q)
            except Exception:
                continue
    return out


def compare_all_modes(
    scene_id: str,
    query: str,
    *,
    session_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Graph (auto-prune), graph_oracle (logged LLM path), and flat."""
    base = compare_modes(scene_id, query, session_path=None)
    oracle: Optional[Dict[str, Any]] = None
    if session_path and session_path.exists():
        g = simulate_graph_search(
            scene_id, query,
            descend_override=replay_graph_from_session(session_path),
        )
        f = base["flat"]
        def _tl(fv: float, gv: float) -> Optional[float]:
            return round(fv / gv, 1) if gv > 0 else None

        oracle = {
            "graph_oracle": _result_dict(g),
            "savings_vs_flat": {
                "tokens_pct": round(100 * (1 - g.input_tokens / f["input_tokens"]), 1) if f["input_tokens"] else 0,
                "views_pct": round(100 * (1 - g.views_checked / f["views_checked"]), 1) if f["views_checked"] else 0,
                "calls_pct": round(100 * (1 - g.llm_calls / f["llm_calls"]), 1) if f["llm_calls"] else 0,
                "tokens_times_less": _tl(f["input_tokens"], g.input_tokens),
                "views_times_faster": _tl(f["views_checked"], g.views_checked),
                "calls_times_less": _tl(f["llm_calls"], g.llm_calls),
            },
        }
    return {**base, "oracle": oracle, "session_file": str(session_path) if session_path else None}


SCALING_VIEW_COUNTS = [3, 5, 10, 15, 19]


def run_scaling_benchmark(
    scene_id: str = "default",
    query: str = "find the bar",
) -> Dict[str, Any]:
    """Sweep active view count to show what drives flat vs graph cost."""
    scene_dir = str(DATA_DIR / scene_id)
    all_views = list_view_ids(scene_dir)
    graph_stats = get_graph_stats(scene_id)

    rows: List[Dict[str, Any]] = []
    for k in SCALING_VIEW_COUNTS:
        k = min(k, len(all_views))
        active = set(all_views[:k])
        g = simulate_graph_search(scene_id, query, active_view_ids=active)
        f = simulate_flat_search(scene_id, query, active_view_ids=active)
        g_dict = _result_dict(g)
        f_dict = _result_dict(f)
        rows.append({
            "views_available": k,
            "graph": {
                "views_checked": g.views_checked,
                "nodes_visited": g.nodes_visited,
                "input_tokens": g.input_tokens,
                "llm_calls": g.llm_calls,
                "total_sec": g_dict["timing"]["total_sec"],
            },
            "flat": {
                "views_checked": f.views_checked,
                "input_tokens": f.input_tokens,
                "llm_calls": f.llm_calls,
                "total_sec": f_dict["timing"]["total_sec"],
            },
            "speedup_sec": round(
                f_dict["timing"]["total_sec"] / g_dict["timing"]["total_sec"], 1
            ) if g_dict["timing"]["total_sec"] > 0 else None,
        })

    if len(rows) >= 2:
        flat_sec_delta = rows[-1]["flat"]["total_sec"] - rows[0]["flat"]["total_sec"]
        graph_sec_delta = rows[-1]["graph"]["total_sec"] - rows[0]["graph"]["total_sec"]
        view_delta = rows[-1]["views_available"] - rows[0]["views_available"]
        flat_per_view = round(flat_sec_delta / view_delta, 2) if view_delta else 0
        graph_per_view = round(graph_sec_delta / view_delta, 2) if view_delta else 0
    else:
        flat_per_view = graph_per_view = 0

    return {
        "query": query,
        "graph_stats": graph_stats,
        "rows": rows,
        "interpretation": {
            "flat_driver": "view_count",
            "flat_sec_per_extra_view": flat_per_view,
            "graph_driver": "tree_depth_and_branch_pruning",
            "graph_sec_per_extra_view": graph_per_view,
            "note": (
                "Flat time grows ~linearly with view count (checks every view). "
                "Graph time grows slowly — dominated by tree depth (~{depth} levels, "
                "{leaves} leaves) and query-specific branches, not total views."
            ).format(
                depth=graph_stats.get("max_depth", "?"),
                leaves=graph_stats.get("leaf_count", "?"),
            ),
        },
    }


def run_scene_benchmark(scene_id: str = "default") -> Dict[str, Any]:
    """Run graph vs flat on every query; attach oracle when a session log exists."""
    scene_dir = DATA_DIR / scene_id
    queries = collect_queries_from_logs(scene_id)
    log_dir = scene_dir / "queries"

    # Map query (lower) -> best session (prefer found=True, more traversal steps)
    session_by_query: Dict[str, Path] = {}
    session_score: Dict[str, int] = {}
    if log_dir.exists():
        for p in log_dir.glob("*.json"):
            try:
                with open(p) as fh:
                    data = json.load(fh)
                q = (data.get("original_query") or "").strip().lower()
                if not q:
                    continue
                found = 1 if (data.get("result") or {}).get("found") else 0
                steps = len(data.get("steps") or [])
                score = found * 1000 + steps
                if q not in session_by_query or score > session_score[q]:
                    session_by_query[q] = p
                    session_score[q] = score
            except Exception:
                continue

    results = []
    for q in queries:
        session = session_by_query.get(q.strip().lower())
        results.append(compare_all_modes(scene_id, q, session_path=session))

    n = len(results)

    def _avg(key: str) -> float:
        return round(sum(r["savings"][key] for r in results) / n, 1) if n else 0.0

    def _avg_ratio(key: str) -> float:
        vals = [r["savings"][key] for r in results if r["savings"].get(key)]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    oracle_rows = [r for r in results if r.get("oracle")]

    def _oracle_avg(key: str) -> float:
        vals = [r["oracle"]["savings_vs_flat"][key] for r in oracle_rows if r["oracle"]["savings_vs_flat"].get(key)]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    avg_graph_tokens = round(sum(r["graph"]["input_tokens"] for r in results) / n) if n else 0
    avg_flat_tokens = round(sum(r["flat"]["input_tokens"] for r in results) / n) if n else 0
    avg_graph_views = round(sum(r["graph"]["views_checked"] for r in results) / n, 1) if n else 0

    def _avg_timing(mode: str, field: str) -> float:
        vals = [r[mode]["timing"][field] for r in results if r[mode].get("timing")]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    avg_graph_sec = _avg_timing("graph", "total_sec")
    avg_flat_sec = _avg_timing("flat", "total_sec")
    sec_ratio = round(avg_flat_sec / avg_graph_sec, 1) if avg_graph_sec > 0 else None

    graph_stats = get_graph_stats(scene_id)
    scaling = run_scaling_benchmark(scene_id)

    return {
        "scene_id": scene_id,
        "total_views": len(list_view_ids(str(scene_dir))),
        "graph_stats": graph_stats,
        "queries_run": n,
        "results": results,
        "scaling": scaling,
        "timing_model": {
            "mode": "fast_llm_no_thinking",
            "sec_per_call": 0.35,
            "tokens_per_sec": 140,
            "formula": "total_sec = infra_sec + llm_calls*0.35 + input_tokens/140",
        },
        "summary": {
            "avg_token_savings_pct": _avg("tokens_pct"),
            "avg_view_savings_pct": _avg("views_pct"),
            "avg_call_savings_pct": _avg("calls_pct"),
            "avg_tokens_times_less": _avg_ratio("tokens_times_less"),
            "avg_views_times_faster": _avg_ratio("views_times_faster"),
            "avg_calls_times_less": _avg_ratio("calls_times_less"),
            "avg_graph_tokens": avg_graph_tokens,
            "avg_flat_tokens": avg_flat_tokens,
            "avg_graph_views_checked": avg_graph_views,
            "avg_graph_total_sec": avg_graph_sec,
            "avg_flat_total_sec": avg_flat_sec,
            "avg_sec_times_faster": sec_ratio,
            "oracle_queries": len(oracle_rows),
            "oracle_avg_token_savings_pct": _oracle_avg("tokens_pct"),
            "oracle_avg_tokens_times_less": _oracle_avg("tokens_times_less"),
            "oracle_avg_views_times_faster": _oracle_avg("views_times_faster"),
            "oracle_avg_calls_times_less": _oracle_avg("calls_times_less"),
            "graph_wins_all_queries": all(r["graph_wins_on_tokens"] for r in results),
        },
    }
