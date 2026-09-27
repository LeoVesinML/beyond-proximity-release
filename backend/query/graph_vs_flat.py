"""Same-input graph vs flat benchmark over captured semantic indexes."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.io.captured_semantic_index import load_captured_scene_index
from backend.query.affordance import expand_query_tokens, score_text_with_tokens, tokenize
from backend.query.model_client import _view_objects, _view_text
from backend.query.tokens import count_tokens

FIVE_SCENE_BENCHMARK_IDS = frozenset({
    "ConferenceHall",
    "Museume",
    "outdoor-drone",
    "outdoor-street",
    "Theater",
})

DEFAULT_SCENE_MAP: dict[str, str] = {
    "ConferenceHall": "ConferenceHall-capture-pilot",
    "Museume": "Museume-capture",
    "outdoor-drone": "outdoor-drone-capture",
    "outdoor-street": "outdoor-street-capture",
    "Theater": "Theater-capture",
}


@dataclass
class SearchMetrics:
    mode: str
    scene_id: str
    benchmark_scene_id: str
    query_id: str
    query: str
    expanded_query: str
    affordance_keys: list[str]
    found: bool
    selected_view_id: str | None
    selected_node_id: str | None
    matched_object: str | None
    views_checked: int = 0
    views_checked_ids: list[str] = field(default_factory=list)
    nodes_visited: int = 0
    visited_node_ids: list[str] = field(default_factory=list)
    semantic_entries_scanned: int = 0
    regions_scanned: int = 0
    objects_scanned: int = 0
    landmarks_scanned: int = 0
    context_chars: int = 0
    input_tokens: int = 0
    elapsed_ms: float = 0.0
    ranked_view_ids: list[str] = field(default_factory=list)
    all_ranked_view_ids: list[str] = field(default_factory=list)
    model_backend: str = "none"
    model_name: str | None = None
    model_call_count: int = 0
    model_input_tokens: int = 0
    model_output_tokens: int = 0
    model_inference_ms: float = 0.0
    model_token_method: str = "not_applicable"
    process_rss_mb_observed: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _top_ranked_views(scored: list[tuple[int, float, str]], *, k: int = 3) -> list[str]:
    """Return up to k view ids ordered by lexical score (overlap, ratio)."""
    ordered = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
    return [view_id for overlap, _, view_id in ordered[:k] if overlap > 0]


def _all_ranked_views(scored: list[tuple[int, float, str]]) -> list[str]:
    """Return every positively scored view in stable lexical rank order."""
    ordered = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
    return [view_id for overlap, _, view_id in ordered if overlap > 0]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root_id(manifest: dict[str, Any]) -> str:
    root = manifest.get("root_node_id") or manifest.get("root_id")
    if not isinstance(root, str) or not root:
        raise ValueError("Captured tree manifest is missing root_node_id")
    return root


def _node_text(node: dict[str, Any]) -> str:
    parts = [str(node.get("name", "")), str(node.get("summary", ""))]
    attrs = node.get("attributes")
    if isinstance(attrs, list):
        parts.extend(str(item) for item in attrs)
    return " ".join(part for part in parts if part)


def _count_semantic_items(view: dict[str, Any]) -> tuple[int, int, int]:
    regions = view.get("visible_regions") if isinstance(view.get("visible_regions"), list) else []
    objects = _view_objects(view)
    landmarks = view.get("landmarks") if isinstance(view.get("landmarks"), list) else []
    return len(regions), len(objects), len(landmarks)


def _add_context(metrics: SearchMetrics, text: str) -> None:
    if not text:
        return
    metrics.context_chars += len(text)
    metrics.input_tokens += count_tokens(text)


def _is_leaf_node(node: dict[str, Any]) -> bool:
    children = node.get("children_ids") if isinstance(node.get("children_ids"), list) else []
    node_type = str(node.get("node_type", ""))
    if not children:
        return True
    return node_type in {"object", "landmark", "leaf"}


def _best_view_for_node(
    node: dict[str, Any],
    views: dict[str, dict[str, Any]],
    query_tokens: set[str],
    metrics: SearchMetrics,
) -> tuple[int, float, str | None, str | None]:
    best: tuple[int, float, str | None, str | None] = (0, 0.0, None, None)
    view_ids = [str(v) for v in node.get("view_ids", []) if str(v) in views]
    for view_id in view_ids:
        if view_id in metrics.views_checked_ids:
            continue
        view = views[view_id]
        metrics.views_checked += 1
        metrics.views_checked_ids.append(view_id)
        regions, objects, landmarks = _count_semantic_items(view)
        metrics.regions_scanned += regions
        metrics.objects_scanned += objects
        metrics.landmarks_scanned += landmarks
        metrics.semantic_entries_scanned += regions + objects + landmarks
        view_text = _view_text(view)
        _add_context(metrics, view_text)
        overlap, ratio = score_text_with_tokens(view_text, query_tokens)
        if (overlap, ratio, view_id) > (best[0], best[1], best[2] or ""):
            matched = None
            if overlap > 0:
                for obj in _view_objects(view):
                    obj_overlap, _ = score_text_with_tokens(str(obj.get("label", "")), query_tokens)
                    if obj_overlap > 0:
                        matched = str(obj.get("label"))
                        break
            best = (overlap, ratio, view_id, matched)
    return best


def simulate_flat_search(
    bundle: dict[str, Any],
    query: str,
    *,
    query_id: str = "",
    benchmark_scene_id: str = "",
    use_affordance: bool = False,
) -> SearchMetrics:
    """Exhaustive lexical scan over every indexed view (same semantic items)."""
    t0 = time.perf_counter()
    views: dict[str, dict[str, Any]] = bundle["views"]
    query_tokens, affordance_keys = expand_query_tokens(query) if use_affordance else (tokenize(query), [])
    expanded = query if not use_affordance else f"{query} [{' '.join(sorted(query_tokens - tokenize(query)))}]"

    metrics = SearchMetrics(
        mode="graph_affordance_flat" if use_affordance else "flat_lexical",
        scene_id=str(bundle["scene_id"]),
        benchmark_scene_id=benchmark_scene_id or str(bundle["scene_id"]),
        query_id=query_id,
        query=query,
        expanded_query=expanded,
        affordance_keys=affordance_keys,
        found=False,
        selected_view_id=None,
        selected_node_id=None,
        matched_object=None,
    )

    best_overlap, best_ratio, best_view, best_object = 0, 0.0, None, None
    scored_views: list[tuple[int, float, str]] = []
    for view_id in sorted(views):
        view = views[view_id]
        regions, objects, landmarks = _count_semantic_items(view)
        metrics.views_checked += 1
        metrics.views_checked_ids.append(view_id)
        metrics.regions_scanned += regions
        metrics.objects_scanned += objects
        metrics.landmarks_scanned += landmarks
        metrics.semantic_entries_scanned += regions + objects + landmarks
        view_text = _view_text(view)
        _add_context(metrics, view_text)
        overlap, ratio = score_text_with_tokens(view_text, query_tokens)
        scored_views.append((overlap, ratio, view_id))
        if (overlap, ratio) > (best_overlap, best_ratio):
            best_overlap, best_ratio = overlap, ratio
            best_view = view_id
            best_object = None
            for obj in _view_objects(view):
                obj_overlap, _ = score_text_with_tokens(str(obj.get("label", "")), query_tokens)
                if obj_overlap > 0:
                    best_object = str(obj.get("label"))
                    break

    metrics.all_ranked_view_ids = _all_ranked_views(scored_views)
    metrics.ranked_view_ids = metrics.all_ranked_view_ids[:3]
    if best_overlap > 0 and best_view:
        metrics.found = True
        metrics.selected_view_id = best_view
        metrics.matched_object = best_object
    metrics.elapsed_ms = (time.perf_counter() - t0) * 1000
    metrics.warnings.append("Flat mode scanned every indexed view with the same lexical scorer.")
    return metrics


def simulate_graph_search(
    bundle: dict[str, Any],
    query: str,
    *,
    query_id: str = "",
    benchmark_scene_id: str = "",
    use_affordance: bool = False,
    branch_keep_ratio: float = 0.5,
    fallback_child_limit: int = 3,
    mode_suffix: str = "",
) -> SearchMetrics:
    """Tree-guided lexical search with branch pruning over the captured semantic tree."""
    if branch_keep_ratio < 0:
        raise ValueError("branch_keep_ratio must be non-negative")
    if fallback_child_limit < 1:
        raise ValueError("fallback_child_limit must be at least 1")

    t0 = time.perf_counter()
    tree: dict[str, dict[str, Any]] = bundle["tree"]
    views: dict[str, dict[str, Any]] = bundle["views"]
    manifest: dict[str, Any] = bundle["manifest"]
    root_id = _root_id(manifest)
    query_tokens, affordance_keys = expand_query_tokens(query) if use_affordance else (tokenize(query), [])
    expanded = query if not use_affordance else f"{query} [{' '.join(sorted(query_tokens - tokenize(query)))}]"

    metrics = SearchMetrics(
        mode=("graph_affordance" if use_affordance else "graph_lexical") + mode_suffix,
        scene_id=str(bundle["scene_id"]),
        benchmark_scene_id=benchmark_scene_id or str(bundle["scene_id"]),
        query_id=query_id,
        query=query,
        expanded_query=expanded,
        affordance_keys=affordance_keys,
        found=False,
        selected_view_id=None,
        selected_node_id=None,
        matched_object=None,
    )

    stack = [root_id]
    visited: set[str] = set()
    best_overlap, best_ratio = 0, 0.0
    best_view: str | None = None
    best_node: str | None = None
    best_object: str | None = None
    scored_views: list[tuple[int, float, str]] = []

    while stack:
        node_id = stack.pop()
        if node_id in visited or node_id not in tree:
            continue
        visited.add(node_id)
        node = tree[node_id]
        metrics.nodes_visited += 1
        metrics.visited_node_ids.append(node_id)
        node_text = _node_text(node)
        _add_context(metrics, node_text)

        if _is_leaf_node(node):
            overlap, ratio, view_id, matched = _best_view_for_node(node, views, query_tokens, metrics)
            if view_id:
                scored_views.append((overlap, ratio, view_id))
            if view_id and (overlap, ratio) > (best_overlap, best_ratio):
                best_overlap, best_ratio = overlap, ratio
                best_view, best_node, best_object = view_id, node_id, matched
            continue

        children = [str(child) for child in node.get("children_ids", []) if str(child) in tree]
        scored = []
        for child_id in children:
            child = tree[child_id]
            overlap, ratio = score_text_with_tokens(_node_text(child), query_tokens)
            scored.append((overlap, ratio, child_id))
        scored.sort(reverse=True)
        if scored and scored[0][0] > 0:
            top = scored[0][0]
            chosen = [child_id for overlap, _, child_id in scored if overlap >= top * branch_keep_ratio]
        else:
            chosen = children[: min(fallback_child_limit, len(children))]
        for child_id in reversed(chosen):
            stack.append(child_id)

    metrics.all_ranked_view_ids = _all_ranked_views(scored_views)
    metrics.ranked_view_ids = metrics.all_ranked_view_ids[:3]
    if best_overlap > 0 and best_view:
        metrics.found = True
        metrics.selected_view_id = best_view
        metrics.selected_node_id = best_node
        metrics.matched_object = best_object
    metrics.elapsed_ms = (time.perf_counter() - t0) * 1000
    metrics.warnings.append(
        "Graph mode prunes sibling branches using the same lexical scorer as flat mode; "
        f"branch_keep_ratio={branch_keep_ratio}, fallback_child_limit={fallback_child_limit}."
    )
    return metrics


def load_scene_bundle(scene_dir: Path, *, allow_unindexed_frames: bool = True) -> dict[str, Any]:
    index = load_captured_scene_index(
        scene_dir.resolve(),
        require_complete=True,
        allow_unindexed_frames=allow_unindexed_frames,
    )
    return {
        "scene_id": scene_dir.name,
        "scene_dir": scene_dir.resolve(),
        "views": index["views"],
        "tree": index["tree"],
        "manifest": index["manifest"],
        "counts": index["counts"],
        "warnings": index.get("warnings", []),
    }


def load_benchmark_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"Benchmark file must contain queries array: {path}")
    return [item for item in queries if isinstance(item, dict)]


def filter_five_scene_queries(queries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [q for q in queries if str(q.get("scene_id")) in FIVE_SCENE_BENCHMARK_IDS]


def resolve_scene_dir(
    benchmark_scene_id: str,
    scene_root: Path,
    scene_map: dict[str, str] | None = None,
) -> Path:
    mapping = scene_map or DEFAULT_SCENE_MAP
    scene_id = mapping.get(benchmark_scene_id, benchmark_scene_id)
    path = scene_root / scene_id
    if not path.is_dir():
        raise FileNotFoundError(f"Scene directory not found for {benchmark_scene_id}: {path}")
    return path


def query_has_view_gt(query: dict[str, Any]) -> bool:
    status = str(query.get("verification_status", ""))
    expected = query.get("expected_view_ids")
    return status.startswith("verified_") and isinstance(expected, list) and len(expected) > 0


def hit_at_k(result_views: list[str], expected_views: list[str], k: int) -> bool | None:
    if not expected_views:
        return None
    expected = {str(item) for item in expected_views}
    ranked = [str(item) for item in result_views[:k] if item]
    return any(view in expected for view in ranked)


def compare_modes_for_query(
    bundle: dict[str, Any],
    query: dict[str, Any],
    *,
    include_affordance: bool = True,
) -> dict[str, Any]:
    text = str(query.get("query", ""))
    qid = str(query.get("query_id", ""))
    benchmark_scene_id = str(query.get("scene_id", ""))
    flat = simulate_flat_search(
        bundle, text, query_id=qid, benchmark_scene_id=benchmark_scene_id, use_affordance=False
    )
    graph = simulate_graph_search(
        bundle, text, query_id=qid, benchmark_scene_id=benchmark_scene_id, use_affordance=False
    )
    affordance_graph = (
        simulate_graph_search(
            bundle, text, query_id=qid, benchmark_scene_id=benchmark_scene_id, use_affordance=True
        )
        if include_affordance
        else None
    )

    gt_views = [str(v) for v in query.get("expected_view_ids", []) if v]
    gt_available = query_has_view_gt(query)

    def _quality(metrics: SearchMetrics) -> dict[str, Any]:
        ranked = metrics.ranked_view_ids or ([metrics.selected_view_id] if metrics.selected_view_id else [])
        return {
            "hit_at_1": hit_at_k(ranked, gt_views, 1) if gt_available else None,
            "hit_at_3": hit_at_k(ranked, gt_views, 3) if gt_available else None,
            "gt_available": gt_available,
            "ranked_view_ids": ranked,
        }

    savings = {
        "views_pct": round(100 * (1 - graph.views_checked / flat.views_checked), 1) if flat.views_checked else 0.0,
        "tokens_pct": round(100 * (1 - graph.input_tokens / flat.input_tokens), 1) if flat.input_tokens else 0.0,
        "context_chars_pct": round(100 * (1 - graph.context_chars / flat.context_chars), 1)
        if flat.context_chars
        else 0.0,
        "runtime_pct": round(100 * (1 - graph.elapsed_ms / flat.elapsed_ms), 1) if flat.elapsed_ms else 0.0,
    }

    row = {
        "query_id": qid,
        "benchmark_scene_id": benchmark_scene_id,
        "scene_id": bundle["scene_id"],
        "query": text,
        "query_type": query.get("query_type"),
        "flat": flat.to_dict(),
        "graph": graph.to_dict(),
        "savings_graph_vs_flat": savings,
        "quality_graph": _quality(graph),
        "quality_flat": _quality(flat),
    }
    if affordance_graph is not None:
        row["graph_affordance"] = affordance_graph.to_dict()
        row["quality_graph_affordance"] = _quality(affordance_graph)
    return row


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"query_count": 0}

    def _avg(mode_key: str, field: str) -> float:
        vals = [row[mode_key][field] for row in rows if isinstance(row.get(mode_key), dict)]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    gt_rows = [row for row in rows if row.get("quality_graph", {}).get("gt_available")]
    hit1_graph = [row["quality_graph"]["hit_at_1"] for row in gt_rows if row["quality_graph"]["hit_at_1"] is not None]
    hit1_flat = [row["quality_flat"]["hit_at_1"] for row in gt_rows if row["quality_flat"]["hit_at_1"] is not None]
    hit3_graph = [row["quality_graph"]["hit_at_3"] for row in gt_rows if row["quality_graph"]["hit_at_3"] is not None]
    hit3_flat = [row["quality_flat"]["hit_at_3"] for row in gt_rows if row["quality_flat"]["hit_at_3"] is not None]

    return {
        "query_count": len(rows),
        "gt_query_count": len(gt_rows),
        "gt_excluded_no_view_ids": len(rows) - len(gt_rows),
        "averages": {
            "flat_views_checked": _avg("flat", "views_checked"),
            "graph_views_checked": _avg("graph", "views_checked"),
            "flat_input_tokens": _avg("flat", "input_tokens"),
            "graph_input_tokens": _avg("graph", "input_tokens"),
            "flat_context_chars": _avg("flat", "context_chars"),
            "graph_context_chars": _avg("graph", "context_chars"),
            "flat_elapsed_ms": _avg("flat", "elapsed_ms"),
            "graph_elapsed_ms": _avg("graph", "elapsed_ms"),
            "graph_nodes_visited": _avg("graph", "nodes_visited"),
            "savings_views_pct": round(
                sum(row["savings_graph_vs_flat"]["views_pct"] for row in rows) / len(rows), 1
            ),
            "savings_tokens_pct": round(
                sum(row["savings_graph_vs_flat"]["tokens_pct"] for row in rows) / len(rows), 1
            ),
        },
        "quality": {
            "hit_at_1_graph": round(sum(hit1_graph) / len(hit1_graph), 4) if hit1_graph else None,
            "hit_at_1_flat": round(sum(hit1_flat) / len(hit1_flat), 4) if hit1_flat else None,
            "hit_at_3_graph": round(sum(hit3_graph) / len(hit3_graph), 4) if hit3_graph else None,
            "hit_at_3_flat": round(sum(hit3_flat) / len(hit3_flat), 4) if hit3_flat else None,
            "denominator_note": (
                "Quality metrics use only queries with verified expected_view_ids "
                f"({len(gt_rows)} of {len(rows)} queries)."
            ),
        },
    }


def run_five_scene_benchmark(
    *,
    benchmark_path: Path,
    scene_root: Path,
    scene_map: dict[str, str] | None = None,
    include_affordance: bool = True,
) -> dict[str, Any]:
    queries = filter_five_scene_queries(load_benchmark_queries(benchmark_path))
    bundles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for query in queries:
        benchmark_scene_id = str(query["scene_id"])
        if benchmark_scene_id not in bundles:
            scene_dir = resolve_scene_dir(benchmark_scene_id, scene_root, scene_map)
            bundles[benchmark_scene_id] = load_scene_bundle(scene_dir)
        rows.append(
            compare_modes_for_query(
                bundles[benchmark_scene_id],
                query,
                include_affordance=include_affordance,
            )
        )
    return {
        "generated_at": _utc_now(),
        "benchmark_path": str(benchmark_path),
        "scene_root": str(scene_root),
        "scene_map": scene_map or DEFAULT_SCENE_MAP,
        "queries": rows,
        "summary": summarize_results(rows),
    }


def write_run_outputs(run: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run.get("summary", {})
    (out_dir / "metrics_summary.json").write_text(
        json.dumps({"summary": summary, "generated_at": run.get("generated_at")}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "per_query_results.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "benchmark_path": run.get("benchmark_path"),
                "scene_root": run.get("scene_root"),
                "scene_map": run.get("scene_map"),
                "generated_at": run.get("generated_at"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Graph vs flat — five-scene run",
        "",
        f"- Generated: {run.get('generated_at')}",
        f"- Queries: {summary.get('query_count', 0)}",
        f"- GT queries with view labels: {summary.get('gt_query_count', 0)}",
        "",
        "## Average savings (graph lexical vs flat lexical)",
        "",
    ]
    averages = summary.get("averages", {})
    for key in (
        "flat_views_checked",
        "graph_views_checked",
        "savings_views_pct",
        "flat_input_tokens",
        "graph_input_tokens",
        "savings_tokens_pct",
        "flat_elapsed_ms",
        "graph_elapsed_ms",
    ):
        if key in averages:
            lines.append(f"- `{key}`: {averages[key]}")
    quality = summary.get("quality", {})
    if quality.get("hit_at_1_graph") is not None:
        lines.extend(
            [
                "",
                "## Quality (GT-aware denominator only)",
                "",
                f"- hit@1 graph: {quality.get('hit_at_1_graph')}",
                f"- hit@1 flat: {quality.get('hit_at_1_flat')}",
                f"- hit@3 graph: {quality.get('hit_at_3_graph')}",
                f"- hit@3 flat: {quality.get('hit_at_3_flat')}",
                f"- {quality.get('denominator_note', '')}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Quality",
                "",
                "No verified view GT in the current benchmark subset; report efficiency only.",
            ]
        )
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv_tables(out_dir, run)


def _write_csv_tables(out_dir: Path, run: dict[str, Any]) -> None:
    """Export paper-ready CSV tables (Person 1 day 9-10)."""
    import csv

    summary = run.get("summary", {})
    averages = summary.get("averages", {})

    with (out_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["metric", "flat_avg", "graph_avg", "savings_pct"])
        pairs = [
            ("views_checked", "flat_views_checked", "graph_views_checked", "savings_views_pct"),
            ("input_tokens", "flat_input_tokens", "graph_input_tokens", "savings_tokens_pct"),
            ("context_chars", "flat_context_chars", "graph_context_chars", None),
            ("elapsed_ms", "flat_elapsed_ms", "graph_elapsed_ms", None),
        ]
        for label, flat_key, graph_key, savings_key in pairs:
            flat_val = averages.get(flat_key, "")
            graph_val = averages.get(graph_key, "")
            savings = averages.get(savings_key, "") if savings_key else ""
            if savings_key is None and flat_val and graph_val:
                try:
                    savings = round(100 * (1 - float(graph_val) / float(flat_val)), 1)
                except (TypeError, ZeroDivisionError):
                    savings = ""
            writer.writerow([label, flat_val, graph_val, savings])

    rows = run.get("queries", [])
    if rows:
        with (out_dir / "per_query_summary.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow([
                "query_id", "scene_id", "query_type", "flat_views", "graph_views",
                "flat_tokens", "graph_tokens", "savings_views_pct", "savings_tokens_pct",
                "graph_found", "flat_found",
                "hit_at_1_graph", "hit_at_1_flat", "hit_at_3_graph", "hit_at_3_flat",
            ])
            for row in rows:
                savings = row.get("savings_graph_vs_flat", {})
                qg = row.get("quality_graph", {})
                qf = row.get("quality_flat", {})
                writer.writerow([
                    row.get("query_id"),
                    row.get("benchmark_scene_id"),
                    row.get("query_type"),
                    row.get("flat", {}).get("views_checked"),
                    row.get("graph", {}).get("views_checked"),
                    row.get("flat", {}).get("input_tokens"),
                    row.get("graph", {}).get("input_tokens"),
                    savings.get("views_pct"),
                    savings.get("tokens_pct"),
                    row.get("graph", {}).get("found"),
                    row.get("flat", {}).get("found"),
                    qg.get("hit_at_1"),
                    qf.get("hit_at_1"),
                    qg.get("hit_at_3"),
                    qf.get("hit_at_3"),
                ])
