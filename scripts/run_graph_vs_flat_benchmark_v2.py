#!/usr/bin/env python3
"""Run same-input graph-vs-flat retrieval over benchmark v2.

This benchmark uses the same captured-scene semantic tree entries for both
methods. Graph search first narrows by tree-native manual zone nodes; flat search
scores every semantic entry in the scene.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENE_ROOT = ROOT / "backend" / "data" / "scenes"
BENCHMARK_PATH = ROOT / "docs" / "benchmarks" / "benchmark_queries_v2.json"
OUT_DIR = ROOT / "outputs" / "week3" / "graph_vs_flat_benchmark_v2"
REPORT_PATH = ROOT / "docs" / "experiments" / "graph_vs_flat" / "graph_vs_flat_benchmark_v2.md"


STOPWORDS = {
    "a", "an", "the", "to", "is", "in", "on", "at", "for", "of", "and", "or",
    "find", "where", "what", "which", "how", "me", "my", "can", "could", "please",
    "there", "all", "inside", "area", "someone", "people", "guests", "place",
}

ALIASES = {
    "collumn": "column",
    "collumns": "columns",
    "pillar": "column",
    "pillars": "columns",
    "stairs": "stair",
    "staircase": "stair",
    "chairs": "chair",
    "seats": "seat",
    "sofas": "sofa",
    "doors": "door",
    "windows": "window",
    "drinks": "bar",
    "watch": "performance",
    "sit": "seat",
    "sitting": "seat",
    "information": "info",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def terms(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        if raw in STOPWORDS:
            continue
        token = ALIASES.get(raw, raw)
        out.add(token)
        if token.endswith("s") and len(token) > 3:
            out.add(token[:-1])
    return out


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def token_estimate(text: str) -> int:
    return max(1, round(len(text) / 4))


def scene_map(benchmark: dict[str, Any]) -> dict[str, str]:
    return {
        str(public): str(captured)
        for public, captured in zip(
            benchmark.get("scene_scope", []),
            benchmark.get("captured_scene_scope", []),
            strict=True,
        )
    }


def node_text(node: dict[str, Any]) -> str:
    pieces = [
        str(node.get("node_id", "")),
        str(node.get("node_type", "")),
        str(node.get("name", "")),
        str(node.get("summary", "")),
        str(node.get("approx_location", "")),
    ]
    for key in ("attributes", "zone_labels", "observations"):
        value = node.get(key)
        if isinstance(value, dict):
            pieces.extend(str(item) for item in value.values() if item is not None)
        elif isinstance(value, list):
            pieces.extend(compact_json(item) if isinstance(item, dict) else str(item) for item in value if item is not None)
    return " ".join(pieces)


def load_scene_index(captured_scene_id: str) -> dict[str, Any]:
    tree_dir = SCENE_ROOT / captured_scene_id / "tree"
    nodes: dict[str, dict[str, Any]] = {}
    for path in sorted(tree_dir.glob("node_*.json")):
        data = load_json(path)
        nodes[str(data["node_id"])] = data
    root = nodes.get("root")
    if root is None:
        raise ValueError(f"Scene has no root tree node: {captured_scene_id}")

    zones = [
        node for node in nodes.values()
        if str(node.get("node_type")) == "zone"
    ]
    entries = [
        node for node in nodes.values()
        if str(node.get("node_type")) not in {"root", "zone"}
    ]
    for node in entries:
        text = node_text(node)
        node["_search_text"] = text
        node["_tokens"] = terms(text)
    child_text_by_zone: dict[str, list[str]] = defaultdict(list)
    for node in entries:
        parent_id = str(node.get("parent_id") or "")
        if parent_id:
            child_text_by_zone[parent_id].append(str(node.get("_search_text", "")))
    for zone in zones:
        zone_text = " ".join([
            node_text(zone),
            " ".join(child_text_by_zone.get(str(zone["node_id"]), [])),
        ])
        zone["_search_text"] = zone_text
        zone["_tokens"] = terms(zone_text)

    entries_by_zone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unzoned: list[dict[str, Any]] = []
    for node in entries:
        parent_id = str(node.get("parent_id") or "")
        if parent_id and parent_id in nodes and nodes[parent_id].get("node_type") == "zone":
            entries_by_zone[parent_id].append(node)
        else:
            unzoned.append(node)

    return {
        "nodes": nodes,
        "zones": zones,
        "entries": entries,
        "entries_by_zone": entries_by_zone,
        "unzoned": unzoned,
    }


def score(tokens: set[str], candidate_tokens: set[str]) -> tuple[int, float]:
    if not tokens:
        return 0, 0.0
    overlap = len(tokens & candidate_tokens)
    return overlap, overlap / len(tokens)


def selected_view(node: dict[str, Any]) -> str | None:
    view_ids = node.get("view_ids")
    if isinstance(view_ids, list) and view_ids:
        return str(view_ids[0])
    return None


def candidate_payload(node: dict[str, Any], query_tokens: set[str], zone_id: str | None) -> dict[str, Any]:
    overlap, ratio = score(query_tokens, node.get("_tokens", set()))
    return {
        "node_id": str(node["node_id"]),
        "zone_id": zone_id,
        "node_type": str(node.get("node_type", "")),
        "label": str(node.get("name", "")),
        "view_id": selected_view(node),
        "overlap": overlap,
        "ratio": ratio,
        "text": str(node.get("_search_text", "")),
    }


def pick_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    type_priority = {"object": 4, "landmark": 3, "region": 2}
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["overlap"],
            item["ratio"],
            type_priority.get(item["node_type"], 0),
            item["node_id"],
        ),
        reverse=True,
    )
    best = ranked[0]
    return best if best["overlap"] > 0 else None


def quality(query: dict[str, Any], best: dict[str, Any] | None) -> dict[str, Any]:
    expected_negative = query.get("query_type") == "negative" or query.get("expected_output_type") == "not_found"
    expected_views = {str(item) for item in query.get("expected_view_ids", [])}
    expected_nodes = {str(item) for item in query.get("expected_node_ids", [])}
    expected_zones = {str(item) for item in query.get("expected_zone_ids", [])}
    expected_labels = {label.casefold() for label in query.get("expected_object_labels", [])}
    found = best is not None
    label = str(best.get("label", "")).casefold() if best else ""
    return {
        "retrieval_success": (not found if expected_negative else found),
        "expected_view_hit": None if expected_negative or not expected_views else best is not None and best.get("view_id") in expected_views,
        "expected_node_hit": None if expected_negative or not expected_nodes else best is not None and best.get("node_id") in expected_nodes,
        "expected_zone_hit": None if expected_negative or not expected_zones else best is not None and best.get("zone_id") in expected_zones,
        "expected_object_hit": None if expected_negative or not expected_labels else bool(label and any(item in label or label in item for item in expected_labels)),
        "not_found_correctness": (not found if expected_negative else None),
    }


def run_method(
    method: str,
    scene_index: dict[str, Any],
    query: dict[str, Any],
) -> dict[str, Any]:
    query_tokens = terms(str(query["query"]))
    start = time.perf_counter()
    scanned: list[dict[str, Any]] = []
    visited_nodes: list[str] = []

    if method == "flat":
        scanned = list(scene_index["entries"])
        visited_nodes = ["flat_all_entries"]
    elif method == "graph":
        zone_scores = [
            (score(query_tokens, zone.get("_tokens", set())), str(zone["node_id"]), zone)
            for zone in scene_index["zones"]
        ]
        zone_scores.sort(key=lambda item: (item[0][0], item[0][1], item[1]), reverse=True)
        positive_scores = [item for item in zone_scores if item[0][0] > 0]
        if positive_scores:
            top = positive_scores[0][0][0]
            chosen_zones = [item for item in positive_scores if item[0][0] >= max(1, top * 0.5)]
        else:
            chosen_zones = zone_scores
        chosen_zone_ids = [item[1] for item in chosen_zones]
        visited_nodes = ["root", *chosen_zone_ids]
        for zone_id in chosen_zone_ids:
            scanned.extend(scene_index["entries_by_zone"].get(zone_id, []))
        scanned.extend(scene_index["unzoned"])
    else:
        raise ValueError(f"Unknown method: {method}")

    candidates: list[dict[str, Any]] = []
    for node in scanned:
        parent_id = str(node.get("parent_id") or "")
        zone_id = parent_id if parent_id in scene_index["entries_by_zone"] else None
        candidates.append(candidate_payload(node, query_tokens, zone_id))
    best = pick_best(candidates)
    if method == "graph" and best is None and len(scanned) < len(scene_index["entries"]):
        scanned_ids = {str(node["node_id"]) for node in scanned}
        remaining = [
            node for node in scene_index["entries"]
            if str(node["node_id"]) not in scanned_ids
        ]
        scanned.extend(remaining)
        visited_nodes.append("graph_fallback_remaining_entries")
        for node in remaining:
            parent_id = str(node.get("parent_id") or "")
            zone_id = parent_id if parent_id in scene_index["entries_by_zone"] else None
            candidates.append(candidate_payload(node, query_tokens, zone_id))
        best = pick_best(candidates)
    if best is not None and method == "graph":
        visited_nodes.append(best["node_id"])
    elapsed_ms = (time.perf_counter() - start) * 1000

    context = [
        {
            "node_id": str(node["node_id"]),
            "node_type": str(node.get("node_type", "")),
            "name": str(node.get("name", "")),
            "summary": str(node.get("summary", "")),
            "view_ids": node.get("view_ids", []),
            "parent_id": node.get("parent_id"),
        }
        for node in scanned
    ]
    context_text = compact_json({"query": query["query"], "entries": context})
    objects_checked = sum(1 for node in scanned if str(node.get("node_type")) == "object")
    regions_checked = sum(1 for node in scanned if str(node.get("node_type")) == "region")
    views_checked = len({view_id for node in scanned for view_id in node.get("view_ids", [])})
    result = {
        "method": method,
        "found": best is not None,
        "selected_node_id": best.get("node_id") if best else None,
        "selected_zone_id": best.get("zone_id") if best else None,
        "selected_view_id": best.get("view_id") if best else None,
        "matched_label": best.get("label") if best else None,
        "score_overlap": best.get("overlap") if best else 0,
        "score_ratio": best.get("ratio") if best else 0.0,
        "visited_nodes": visited_nodes,
        "latency_ms": round(elapsed_ms, 4),
        "semantic_entries_scanned": len(scanned),
        "objects_checked": objects_checked,
        "regions_checked": regions_checked,
        "views_checked": views_checked,
        "context_size_chars": len(context_text),
        "estimated_input_tokens": token_estimate(context_text),
    }
    result.update(quality(query, best))
    return result


def aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values = [row[method] for row in rows]
    metrics = {}
    for key in (
        "latency_ms",
        "semantic_entries_scanned",
        "objects_checked",
        "regions_checked",
        "views_checked",
        "context_size_chars",
        "estimated_input_tokens",
    ):
        metrics[key] = round(mean(float(item[key]) for item in values), 4) if values else None
    for key in (
        "retrieval_success",
        "expected_view_hit",
        "expected_node_hit",
        "expected_zone_hit",
        "expected_object_hit",
        "not_found_correctness",
    ):
        applicable = [item[key] for item in values if item.get(key) is not None]
        metrics[key] = {
            "value": round(sum(1 for item in applicable if item) / len(applicable), 4) if applicable else None,
            "numerator": sum(1 for item in applicable if item),
            "denominator": len(applicable),
        }
    return metrics


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graph = aggregate(rows, "graph")
    flat = aggregate(rows, "flat")
    speedups = []
    token_reductions = []
    scan_reductions = []
    for row in rows:
        g = row["graph"]
        f = row["flat"]
        if g["latency_ms"] > 0:
            speedups.append(f["latency_ms"] / g["latency_ms"])
        if f["estimated_input_tokens"]:
            token_reductions.append(1 - g["estimated_input_tokens"] / f["estimated_input_tokens"])
        if f["semantic_entries_scanned"]:
            scan_reductions.append(1 - g["semantic_entries_scanned"] / f["semantic_entries_scanned"])
        row["speedup_vs_flat"] = round(f["latency_ms"] / g["latency_ms"], 4) if g["latency_ms"] > 0 else None
        row["token_reduction_vs_flat"] = round(1 - g["estimated_input_tokens"] / f["estimated_input_tokens"], 4) if f["estimated_input_tokens"] else None
        row["scan_reduction_vs_flat"] = round(1 - g["semantic_entries_scanned"] / f["semantic_entries_scanned"], 4) if f["semantic_entries_scanned"] else None
    return {
        "graph": graph,
        "flat": flat,
        "mean_speedup_vs_flat": round(mean(speedups), 4) if speedups else None,
        "mean_token_reduction_vs_flat": round(mean(token_reductions), 4) if token_reductions else None,
        "mean_scan_reduction_vs_flat": round(mean(scan_reductions), 4) if scan_reductions else None,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    graph = summary["graph"]
    flat = summary["flat"]
    context_reduction = 1 - graph["context_size_chars"] / flat["context_size_chars"]
    view_reduction = 1 - graph["views_checked"] / flat["views_checked"]
    lines = [
        "# Graph Vs Flat Benchmark V2",
        "",
        "Status date: 2026-07-01.",
        "",
        "This is a same-input deterministic benchmark over `benchmark_queries_v2.json`. Both methods use the same captured-scene semantic entries and the same manual reference GT.",
        "",
        "## Scope",
        "",
        "- Mode: deterministic local lexical retrieval, no live model calls.",
        "- Graph: select tree-native manual zones, then scan entries under those zones.",
        "- Flat: scan every semantic entry in the scene.",
        "- GT: manual semantic-index reference labels, not independent dataset GT.",
        "",
        "## Efficiency Summary",
        "",
        "| Metric | Graph mean | Flat mean | Graph improvement |",
        "|---|---:|---:|---:|",
        f"| latency_ms | {graph['latency_ms']} | {flat['latency_ms']} | {summary['mean_speedup_vs_flat']}x speedup |",
        f"| semantic_entries_scanned | {graph['semantic_entries_scanned']} | {flat['semantic_entries_scanned']} | {summary['mean_scan_reduction_vs_flat']:.2%} fewer |",
        f"| context_size_chars | {graph['context_size_chars']} | {flat['context_size_chars']} | {context_reduction:.2%} smaller |",
        f"| estimated_input_tokens | {graph['estimated_input_tokens']} | {flat['estimated_input_tokens']} | {summary['mean_token_reduction_vs_flat']:.2%} fewer |",
        f"| views_checked | {graph['views_checked']} | {flat['views_checked']} | {view_reduction:.2%} fewer |",
        "",
        "## Quality Summary",
        "",
        "| Metric | Graph | Flat |",
        "|---|---:|---:|",
    ]
    for key in (
        "retrieval_success",
        "expected_view_hit",
        "expected_node_hit",
        "expected_zone_hit",
        "expected_object_hit",
        "not_found_correctness",
    ):
        g = graph[key]
        f = flat[key]
        lines.append(
            f"| {key} | {g['numerator']}/{g['denominator']} = {g['value']} | "
            f"{f['numerator']}/{f['denominator']} = {f['value']} |"
        )
    lines.extend([
        "",
        "## Evidence",
        "",
        "```text",
        "outputs/week3/graph_vs_flat_benchmark_v2/graph_vs_flat_metrics.json",
        "outputs/week3/graph_vs_flat_benchmark_v2/per_query_results.json",
        "docs/experiments/graph_vs_flat/graph_vs_flat_benchmark_v2.md",
        "```",
        "",
        "## Limitations",
        "",
        "- This proves deterministic same-input graph pruning efficiency, not live provider speed.",
        "- Token counts are estimated from serialized context size.",
        "- Manual labels are reference GT, not independent dataset GT.",
    ])
    return "\n".join(lines) + "\n"


def run(benchmark_path: Path = BENCHMARK_PATH, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    benchmark = load_json(benchmark_path)
    mapping = scene_map(benchmark)
    scene_indexes = {
        public_scene_id: load_scene_index(captured_scene_id)
        for public_scene_id, captured_scene_id in mapping.items()
    }
    rows: list[dict[str, Any]] = []
    type_counts = Counter()
    scene_counts = Counter()
    for query in benchmark["queries"]:
        if not isinstance(query, dict):
            continue
        scene_id = str(query["scene_id"])
        scene_index = scene_indexes[scene_id]
        graph = run_method("graph", scene_index, query)
        flat = run_method("flat", scene_index, query)
        row = {
            "query_id": query["query_id"],
            "scene_id": scene_id,
            "query_type": query["query_type"],
            "query": query["query"],
            "graph": graph,
            "flat": flat,
        }
        rows.append(row)
        type_counts[str(query["query_type"])] += 1
        scene_counts[scene_id] += 1

    summary = build_summary(rows)
    payload = {
        "schema_version": "semanticsplat.graph_vs_flat_benchmark.v2",
        "benchmark_id": benchmark["benchmark_id"],
        "run_id": "graph_vs_flat_benchmark_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "deterministic_local",
        "query_count": len(rows),
        "query_type_counts": dict(sorted(type_counts.items())),
        "scene_query_counts": dict(sorted(scene_counts.items())),
        "summary": summary,
        "per_query": rows,
        "claims_allowed": [
            "same-input deterministic graph search scans fewer entries than flat search",
            "same-input deterministic graph search uses smaller serialized context than flat search",
            "quality is measured only against manual semantic-index reference GT",
        ],
        "claims_not_allowed": [
            "live provider latency",
            "official Replica/ScanNet accuracy",
            "independent semantic accuracy",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "graph_vs_flat_metrics.json", payload)
    write_json(out_dir / "per_query_results.json", {"per_query": rows})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph-vs-flat benchmark v2")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    payload = run(args.benchmark, args.out)
    print(json.dumps({
        "query_count": payload["query_count"],
        "summary": payload["summary"],
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
