#!/usr/bin/env python3
"""Evaluate manual, geometric, deterministic-semantic, and Cursor-MCP hierarchies."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.geometry.spatial_graph import build_spatial_clusters
from backend.io.nerfstudio import load_transforms
from backend.io.view_store import list_view_summaries
from backend.query.graph_vs_flat import (
    load_benchmark_queries,
    load_scene_bundle,
    query_has_view_gt,
    simulate_graph_search,
)
from backend.query.tokens import count_tokens


SCHEMA_VERSION = "semanticsplat.hierarchy_construction_evaluation.v1"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.fmean(rows), 6) if rows else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _manual_zones(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = _read_json(path)
    return {
        str(scene["captured_scene_id"]): [
            zone for zone in scene.get("zones", []) if isinstance(zone, dict)
        ]
        for scene in data.get("scenes", [])
        if isinstance(scene, dict) and scene.get("captured_scene_id")
    }


def _positions(scene_dir: Path) -> dict[str, list[float]]:
    scene = load_transforms(str(scene_dir / "transforms.json"))
    return {
        str(frame.view_id): frame.camera_position()
        for frame in scene.frames
        if frame.view_id
    }


def _cluster_records(
    clusters: list[list[str]],
    summaries: list[dict[str, Any]],
    *,
    variant: str,
) -> list[dict[str, Any]]:
    summary_by_view = {
        str(summary["view_id"]): summary
        for summary in summaries
        if summary.get("view_id")
    }
    records: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        room_types = [
            str(summary_by_view.get(view_id, {}).get("room_type") or "unknown")
            for view_id in cluster
        ]
        known = [value for value in room_types if value != "unknown"]
        room_type = Counter(known).most_common(1)[0][0] if known else "spatial"
        summaries_text = [
            str(summary_by_view.get(view_id, {}).get("summary") or "")
            for view_id in cluster
        ]
        records.append(
            {
                "zone_id": f"{variant}_zone_{index:02d}",
                "zone_name": (
                    f"{room_type.replace('_', ' ')} cluster {index}"
                    if variant == "pose_semantic_merge"
                    else f"pose cluster {index}"
                ),
                "view_ids": sorted(cluster),
                "rationale": " ".join(text for text in summaries_text if text),
            }
        )
    return records


def _agent_zones(
    decisions: dict[str, Any],
    scene_id: str,
    expected_views: set[str],
) -> list[dict[str, Any]]:
    constraints = decisions.get("constraints")
    if not isinstance(constraints, dict) or constraints.get("direct_mcp_calls") is not True:
        raise ValueError("Cursor hierarchy evidence must record direct_mcp_calls=true")
    if constraints.get("mcp_tool_call_count") != 5:
        raise ValueError("Cursor hierarchy evidence must record exactly five MCP tool calls")
    scene = next(
        (
            item
            for item in decisions.get("scenes", [])
            if isinstance(item, dict) and item.get("scene_id") == scene_id
        ),
        None,
    )
    if scene is None:
        raise ValueError(f"Cursor hierarchy evidence is missing {scene_id}")
    zones = [zone for zone in scene.get("zones", []) if isinstance(zone, dict)]
    assigned = [
        str(view_id)
        for zone in zones
        for view_id in zone.get("view_ids", [])
    ]
    duplicates = sorted(
        view_id for view_id, count in Counter(assigned).items() if count > 1
    )
    missing = sorted(expected_views - set(assigned))
    unexpected = sorted(set(assigned) - expected_views)
    if duplicates or missing or unexpected:
        raise ValueError(
            f"Invalid Cursor zones for {scene_id}: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    return [
        {
            "zone_id": str(zone["zone_id"]),
            "zone_name": str(zone.get("zone_name") or zone["zone_id"]),
            "view_ids": sorted(str(view_id) for view_id in zone["view_ids"]),
            "rationale": str(zone.get("rationale") or ""),
        }
        for zone in zones
    ]


def _tree_with_zones(
    bundle: dict[str, Any],
    zones: list[dict[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    source_tree: dict[str, dict[str, Any]] = bundle["tree"]
    tree = {
        node_id: copy.deepcopy(node)
        for node_id, node in source_tree.items()
        if node_id != "root" and str(node.get("node_type")) != "zone"
    }
    zone_children: dict[str, list[str]] = {
        str(zone["zone_id"]): [] for zone in zones
    }
    root_children: list[str] = []
    for node_id, node in tree.items():
        node_views = {str(view_id) for view_id in node.get("view_ids", [])}
        matches = [
            (
                len(node_views & {str(view_id) for view_id in zone["view_ids"]}),
                -len(zone["view_ids"]),
                str(zone["zone_id"]),
            )
            for zone in zones
        ]
        matches = [match for match in matches if match[0] > 0]
        if not matches:
            node["parent_id"] = "root"
            node["depth"] = 1
            root_children.append(node_id)
            continue
        zone_id = max(matches)[2]
        node["parent_id"] = zone_id
        node["depth"] = 2
        zone_children[zone_id].append(node_id)

    for zone in zones:
        zone_id = str(zone["zone_id"])
        tree[zone_id] = {
            "schema_version": "semanticsplat.tree_node.v1",
            "scene_id": bundle["scene_id"],
            "node_id": zone_id,
            "node_type": "zone",
            "name": str(zone["zone_name"]),
            "summary": str(zone.get("rationale") or zone["zone_name"]),
            "semantic_index_mode": "manual",
            "parent_id": "root",
            "children_ids": sorted(zone_children[zone_id]),
            "view_ids": sorted(str(view_id) for view_id in zone["view_ids"]),
            "depth": 1,
            "construction_variant": variant,
        }
    tree["root"] = {
        "schema_version": "semanticsplat.tree_node.v1",
        "scene_id": bundle["scene_id"],
        "node_id": "root",
        "node_type": "root",
        "name": str(bundle["scene_id"]),
        "summary": "Root of the hierarchy-construction evaluation tree.",
        "semantic_index_mode": "manual",
        "parent_id": None,
        "children_ids": [str(zone["zone_id"]) for zone in zones] + sorted(root_children),
        "view_ids": sorted(bundle["views"]),
        "depth": 0,
        "construction_variant": variant,
    }
    manifest = copy.deepcopy(bundle["manifest"])
    manifest.update(
        {
            "root_node_id": "root",
            "root_id": "root",
            "node_count": len(tree),
            "node_ids": sorted(tree),
            "construction_variant": variant,
        }
    )
    return {**bundle, "tree": tree, "manifest": manifest}


def _zone_assignment_metrics(
    zones: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    view_ids: set[str],
) -> dict[str, Any]:
    predicted_memberships: dict[str, set[str]] = defaultdict(set)
    for zone in zones:
        for view_id in zone.get("view_ids", []):
            predicted_memberships[str(view_id)].add(str(zone["zone_id"]))
    reference_memberships: dict[str, set[str]] = defaultdict(set)
    for zone in reference:
        for view_id in zone.get("view_ids", []):
            reference_memberships[str(view_id)].add(str(zone["zone_id"]))

    tp = fp = fn = tn = 0
    ordered = sorted(view_ids)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            predicted_same = bool(
                predicted_memberships.get(left, set())
                & predicted_memberships.get(right, set())
            )
            reference_same = bool(
                reference_memberships.get(left, set())
                & reference_memberships.get(right, set())
            )
            if predicted_same and reference_same:
                tp += 1
            elif predicted_same:
                fp += 1
            elif reference_same:
                fn += 1
            else:
                tn += 1
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 6)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    weighted_jaccard = 0.0
    weighted_jaccard_denominator = 0
    for zone in zones:
        predicted_views = {str(value) for value in zone.get("view_ids", [])}
        best = max(
            (
                len(predicted_views & {str(value) for value in ref.get("view_ids", [])})
                / len(predicted_views | {str(value) for value in ref.get("view_ids", [])})
                for ref in reference
                if predicted_views | {str(value) for value in ref.get("view_ids", [])}
            ),
            default=0.0,
        )
        weighted_jaccard += len(predicted_views) * best
        weighted_jaccard_denominator += len(predicted_views)
    assigned = [
        str(view_id)
        for zone in zones
        for view_id in zone.get("view_ids", [])
    ]
    counts = Counter(assigned)
    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "pairwise_accuracy": _rate(tp + tn, tp + fp + fn + tn),
        "weighted_best_zone_jaccard": round(
            weighted_jaccard / weighted_jaccard_denominator, 6
        ) if weighted_jaccard_denominator else None,
        "pair_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "orphan_view_ids": sorted(view_ids - set(assigned)),
        "duplicate_view_ids": sorted(
            view_id for view_id, count in counts.items() if count > 1
        ),
        "unexpected_view_ids": sorted(set(assigned) - view_ids),
        "assigned_view_count": len(set(assigned) & view_ids),
        "view_denominator": len(view_ids),
    }


def _structure(tree_bundle: dict[str, Any], zones: list[dict[str, Any]]) -> dict[str, Any]:
    tree = tree_bundle["tree"]
    depths = [
        int(node.get("depth", 0))
        for node in tree.values()
        if isinstance(node, dict)
    ]
    return {
        "node_count": len(tree),
        "zone_count": len(zones),
        "max_depth": max(depths, default=0),
        "leaf_count": sum(
            not node.get("children_ids")
            for node in tree.values()
            if isinstance(node, dict)
        ),
    }


def _rank_quality(ranked: list[str], expected: list[str]) -> tuple[bool, bool, float]:
    expected_set = set(expected)
    rank = next(
        (
            index
            for index, view_id in enumerate(ranked, start=1)
            if view_id in expected_set
        ),
        None,
    )
    return (
        rank == 1,
        rank is not None and rank <= 3,
        1.0 / rank if rank is not None else 0.0,
    )


def _query_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["quality_eligible"]]
    return {
        "query_count": len(rows),
        "quality_denominator": len(eligible),
        "hit_at_1": _rate(sum(row["hit_at_1"] for row in eligible), len(eligible)),
        "hit_at_3": _rate(sum(row["hit_at_3"] for row in eligible), len(eligible)),
        "mrr": _mean(float(row["reciprocal_rank"]) for row in eligible),
        "mean_views_checked": _mean(float(row["views_checked"]) for row in rows),
        "mean_entries_scanned": _mean(
            float(row["semantic_entries_scanned"]) for row in rows
        ),
        "mean_input_tokens": _mean(float(row["input_tokens"]) for row in rows),
        "total_input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "mean_elapsed_ms": _mean(float(row["elapsed_ms"]) for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["variant"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Hierarchy construction evaluation",
        "",
        "The manual zones are a human/Cursor-assisted reference, not independent dataset GT. "
        "Cursor Agent used the project MCP server directly and was not shown the manual-zone file.",
        "",
        "| Variant | Zones | Pairwise F1 | Weighted Jaccard | Hit@1 | Hit@3 | MRR | Views/query | Tokens/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, record in summary["variants"].items():
        structure = record["structure"]
        agreement = record["assignment_agreement"]
        query = record["query_metrics"]
        lines.append(
            f"| {variant} | {structure['zone_count']} | "
            f"{agreement['pairwise_f1']} | {agreement['weighted_best_zone_jaccard']} | "
            f"{query['hit_at_1']} | {query['hit_at_3']} | {query['mrr']} | "
            f"{query['mean_views_checked']} | {query['mean_input_tokens']} |"
        )
    lines.extend(
        [
            "",
            "## Construction accounting",
            "",
            "| Variant | Runtime status | Measured local seconds | Provider tokens | Serialized context tokens |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for variant, record in summary["variants"].items():
        cost = record["construction_cost"]
        lines.append(
            f"| {variant} | {cost['runtime_status']} | "
            f"{cost['measured_local_seconds']} | {cost['provider_tokens']} | "
            f"{cost['serialized_context_tokens']} |"
        )
    lines.extend(
        [
            "",
            "Provider tokens remain `N/A` for the historical manual reference and Cursor UI "
            "run because neither exposes an auditable token counter. Deterministic variants "
            "make no provider calls, so their provider-token count is exactly 0.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate(config_path: Path, out_dir: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    scene_root = ROOT / str(config["scene_root"])
    benchmark = load_benchmark_queries(ROOT / str(config["benchmark_path"]))
    reference_by_scene = _manual_zones(
        ROOT / str(config["manual_zone_reference_path"])
    )
    decisions = _read_json(ROOT / str(config["agent_decisions_path"]))
    policy = config["query_policy"]
    scene_map = {
        str(scene_id): str(benchmark_id)
        for scene_id, benchmark_id in config["benchmark_scene_map"].items()
    }
    queries_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in benchmark:
        for scene_id, benchmark_id in scene_map.items():
            if str(query.get("scene_id")) == benchmark_id:
                queries_by_scene[scene_id].append(query)
                break

    variant_scene_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_rows: list[dict[str, Any]] = []
    for scene_id in config["scene_ids"]:
        scene_id = str(scene_id)
        scene_dir = scene_root / scene_id
        bundle = load_scene_bundle(scene_dir)
        view_ids = set(bundle["views"])
        summaries = list_view_summaries(str(scene_dir))
        positions = _positions(scene_dir)
        serialized_tokens = count_tokens(
            json.dumps(
                {
                    "positions": positions,
                    "summaries": summaries,
                },
                sort_keys=True,
            )
        )
        reference = reference_by_scene[scene_id]
        variants: dict[str, tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = {}

        reference_zones = [
            {
                "zone_id": str(zone["zone_id"]),
                "zone_name": str(zone["zone_label"]),
                "view_ids": sorted(str(value) for value in zone["view_ids"]),
                "rationale": str(zone.get("description") or ""),
            }
            for zone in reference
        ]
        variants["manual_reference"] = (
            bundle,
            reference_zones,
            {
                "runtime_status": "historical_human_time_unavailable",
                "measured_local_seconds": None,
                "provider_tokens": None,
                "serialized_context_tokens": serialized_tokens,
            },
        )

        started = time.perf_counter()
        pose_clusters = build_spatial_clusters(
            positions,
            scale_factor=float(config["spatial_scale_factor"]),
        )
        pose_zones = _cluster_records(pose_clusters, summaries, variant="pose_only")
        pose_tree = _tree_with_zones(bundle, pose_zones, variant="pose_only")
        pose_seconds = time.perf_counter() - started
        variants["pose_only"] = (
            pose_tree,
            pose_zones,
            {
                "runtime_status": "measured_local_deterministic",
                "measured_local_seconds": round(pose_seconds, 6),
                "provider_tokens": 0,
                "serialized_context_tokens": serialized_tokens,
            },
        )

        started = time.perf_counter()
        semantic_clusters = build_spatial_clusters(
            positions,
            scale_factor=float(config["spatial_scale_factor"]),
            view_summaries=summaries,
        )
        semantic_zones = _cluster_records(
            semantic_clusters,
            summaries,
            variant="pose_semantic_merge",
        )
        semantic_tree = _tree_with_zones(
            bundle,
            semantic_zones,
            variant="pose_semantic_merge",
        )
        semantic_seconds = time.perf_counter() - started
        variants["pose_semantic_merge"] = (
            semantic_tree,
            semantic_zones,
            {
                "runtime_status": "measured_local_deterministic",
                "measured_local_seconds": round(semantic_seconds, 6),
                "provider_tokens": 0,
                "serialized_context_tokens": serialized_tokens,
            },
        )

        agent_zones = _agent_zones(decisions, scene_id, view_ids)
        variants["cursor_agent_mcp"] = (
            _tree_with_zones(bundle, agent_zones, variant="cursor_agent_mcp"),
            agent_zones,
            {
                "runtime_status": "unavailable_cursor_agent_ui_not_metered",
                "measured_local_seconds": None,
                "provider_tokens": None,
                "serialized_context_tokens": serialized_tokens,
                "direct_mcp_calls": True,
            },
        )

        for variant, (tree_bundle, zones, construction) in variants.items():
            variant_scene_records[variant].append(
                {
                    "scene_id": scene_id,
                    "structure": _structure(tree_bundle, zones),
                    "assignment_agreement": _zone_assignment_metrics(
                        zones,
                        reference,
                        view_ids,
                    ),
                    "construction_cost": construction,
                }
            )
            for query in queries_by_scene[scene_id]:
                metrics = simulate_graph_search(
                    tree_bundle,
                    str(query["query"]),
                    query_id=str(query["query_id"]),
                    benchmark_scene_id=scene_map[scene_id],
                    use_affordance=bool(policy["use_affordance"]),
                    branch_keep_ratio=float(policy["branch_keep_ratio"]),
                    fallback_child_limit=int(policy["fallback_child_limit"]),
                    mode_suffix=f"_{variant}",
                )
                eligible = query_has_view_gt(query)
                expected = [
                    str(value)
                    for value in query.get("expected_view_ids", [])
                    if value
                ]
                hit1, hit3, rr = _rank_quality(
                    metrics.all_ranked_view_ids,
                    expected,
                ) if eligible else (False, False, 0.0)
                query_rows.append(
                    {
                        "variant": variant,
                        "scene_id": scene_id,
                        "query_id": str(query["query_id"]),
                        "query_type": str(query.get("query_type") or "unspecified"),
                        "quality_eligible": eligible,
                        "hit_at_1": hit1,
                        "hit_at_3": hit3,
                        "reciprocal_rank": round(rr, 6),
                        "views_checked": metrics.views_checked,
                        "semantic_entries_scanned": metrics.semantic_entries_scanned,
                        "input_tokens": metrics.input_tokens,
                        "elapsed_ms": round(metrics.elapsed_ms, 6),
                    }
                )

    variants_summary: dict[str, Any] = {}
    flat_construction_rows: list[dict[str, Any]] = []
    for variant in config["variants"]:
        scene_records = variant_scene_records[str(variant)]
        variant_queries = [
            row for row in query_rows if row["variant"] == str(variant)
        ]
        structures = [record["structure"] for record in scene_records]
        agreements = [record["assignment_agreement"] for record in scene_records]
        costs = [record["construction_cost"] for record in scene_records]
        variants_summary[str(variant)] = {
            "scene_count": len(scene_records),
            "structure": {
                "node_count": sum(record["node_count"] for record in structures),
                "zone_count": sum(record["zone_count"] for record in structures),
                "max_depth": max(
                    (record["max_depth"] for record in structures),
                    default=0,
                ),
                "leaf_count": sum(record["leaf_count"] for record in structures),
            },
            "assignment_agreement": {
                "pairwise_precision": _mean(
                    float(record["pairwise_precision"])
                    for record in agreements
                    if record["pairwise_precision"] is not None
                ),
                "pairwise_recall": _mean(
                    float(record["pairwise_recall"])
                    for record in agreements
                    if record["pairwise_recall"] is not None
                ),
                "pairwise_f1": _mean(
                    float(record["pairwise_f1"]) for record in agreements
                ),
                "pairwise_accuracy": _mean(
                    float(record["pairwise_accuracy"])
                    for record in agreements
                    if record["pairwise_accuracy"] is not None
                ),
                "weighted_best_zone_jaccard": _mean(
                    float(record["weighted_best_zone_jaccard"])
                    for record in agreements
                    if record["weighted_best_zone_jaccard"] is not None
                ),
                "orphan_view_count": sum(
                    len(record["orphan_view_ids"]) for record in agreements
                ),
                "duplicate_view_count": sum(
                    len(record["duplicate_view_ids"]) for record in agreements
                ),
                "view_denominator": sum(
                    int(record["view_denominator"]) for record in agreements
                ),
            },
            "query_metrics": _query_summary(variant_queries),
            "construction_cost": {
                "runtime_status": (
                    costs[0]["runtime_status"]
                    if costs
                    and len({cost["runtime_status"] for cost in costs}) == 1
                    else "mixed"
                ),
                "measured_local_seconds": (
                    round(
                        sum(
                            float(cost["measured_local_seconds"])
                            for cost in costs
                            if cost["measured_local_seconds"] is not None
                        ),
                        6,
                    )
                    if any(
                        cost["measured_local_seconds"] is not None
                        for cost in costs
                    )
                    else None
                ),
                "provider_tokens": (
                    sum(int(cost["provider_tokens"]) for cost in costs)
                    if costs
                    and all(cost["provider_tokens"] is not None for cost in costs)
                    else None
                ),
                "serialized_context_tokens": sum(
                    int(cost["serialized_context_tokens"]) for cost in costs
                ),
            },
            "per_scene": scene_records,
        }
        for record in scene_records:
            flat_construction_rows.append(
                {
                    "variant": variant,
                    "scene_id": record["scene_id"],
                    "zone_count": record["structure"]["zone_count"],
                    "node_count": record["structure"]["node_count"],
                    "max_depth": record["structure"]["max_depth"],
                    "pairwise_f1": record["assignment_agreement"]["pairwise_f1"],
                    "weighted_best_zone_jaccard": record[
                        "assignment_agreement"
                    ]["weighted_best_zone_jaccard"],
                    "orphan_view_count": len(
                        record["assignment_agreement"]["orphan_view_ids"]
                    ),
                    "duplicate_view_count": len(
                        record["assignment_agreement"]["duplicate_view_ids"]
                    ),
                    "construction_seconds": record["construction_cost"][
                        "measured_local_seconds"
                    ],
                    "provider_tokens": record["construction_cost"][
                        "provider_tokens"
                    ],
                }
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": _display_path(config_path),
        "benchmark_path": _display_path(
            ROOT / str(config["benchmark_path"])
        ),
        "manual_reference_status": (
            "human/Cursor-assisted reference; not independent dataset GT"
        ),
        "agent_evidence": {
            "path": _display_path(
                ROOT / str(config["agent_decisions_path"])
            ),
            "surface": decisions.get("agent_surface"),
            "mcp_server": decisions.get("mcp_server"),
            "direct_mcp_calls": decisions.get("constraints", {}).get(
                "direct_mcp_calls"
            ),
            "mcp_tool_call_count": decisions.get("constraints", {}).get(
                "mcp_tool_call_count"
            ),
            "provider_tokens": decisions.get("constraints", {}).get(
                "provider_token_count"
            ),
        },
        "fixed_policy": config,
        "variants": variants_summary,
        "per_query": query_rows,
    }
    _write_json(out_dir / "hierarchy_evaluation.json", summary)
    _write_csv(out_dir / "hierarchy_structure.csv", flat_construction_rows)
    _write_csv(out_dir / "hierarchy_query_results.csv", query_rows)
    (out_dir / "hierarchy_evaluation.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "hierarchy_construction_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "hierarchy_construction" / "four_variant_v1",
    )
    args = parser.parse_args()
    summary = evaluate(args.config.resolve(), args.out.resolve())
    print(
        f"Wrote {len(summary['variants'])} hierarchy variants to "
        f"{args.out.resolve()}"
    )


if __name__ == "__main__":
    main()
