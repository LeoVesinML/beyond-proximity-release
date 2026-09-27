#!/usr/bin/env python3
"""Evaluate canonical SemanticSplat query results against benchmark v1."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


METRICS_SCHEMA = "semanticsplat.metrics_record.v1"
QUERY_RESULT_SCHEMA = "semanticsplat.query_result.v1"
VALID_MODES = {"stub", "live", "cached_live"}
QUERY_TYPE_STRATA = ("simple", "compound", "relational", "multi_hop", "functional")
QUERY_TYPE_ALIASES = {
    "simple_object": "simple",
    "attribute": "compound",
    "negative": "simple",
}
FAILURE_CATEGORIES = (
    "wrong room / zone",
    "wrong object",
    "wrong view",
    "missing object in semantic description",
    "bad query parsing",
    "bad traversal",
    "bad bbox",
    "invalid depth",
    "ambiguous ground truth",
    "baseline adapter issue",
    "missing GT",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def validate_result(data: dict[str, Any], query: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = (
        "schema_version",
        "run_id",
        "method",
        "mode",
        "scene_id",
        "query_id",
        "query",
        "query_type",
        "structured_plan",
        "visited_nodes",
        "selected_views",
        "result",
        "metrics_log",
        "warnings",
    )
    for key in required_top:
        if key not in data:
            errors.append(f"missing top-level field: {key}")

    if data.get("schema_version") != QUERY_RESULT_SCHEMA:
        errors.append(f"schema_version must be {QUERY_RESULT_SCHEMA}")
    if data.get("mode") not in VALID_MODES:
        errors.append("mode must be stub, live, or cached_live")
    if data.get("availability_status", "available") not in {"available", "unavailable"}:
        errors.append("availability_status must be available or unavailable")
    if data.get("query_id") != query.get("query_id"):
        errors.append("query_id does not match benchmark record")
    if data.get("scene_id") != query.get("scene_id"):
        errors.append("scene_id does not match benchmark record")
    if data.get("query_type") != query.get("query_type"):
        errors.append("query_type does not match benchmark record")
    if not isinstance(data.get("visited_nodes"), list):
        errors.append("visited_nodes must be a list")
    if not isinstance(data.get("selected_views"), list):
        errors.append("selected_views must be a list")
    if not isinstance(data.get("structured_plan"), dict):
        errors.append("structured_plan must be an object")
    if not isinstance(data.get("warnings"), list):
        errors.append("warnings must be a list")

    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
    else:
        for key in (
            "found",
            "matched_object",
            "selected_node_id",
            "selected_view_id",
            "bbox_2d",
            "bbox_3d",
            "camera_pose",
            "confidence",
            "explanation",
        ):
            if key not in result:
                errors.append(f"missing result field: {key}")
        if "found" in result and not isinstance(result["found"], bool):
            errors.append("result.found must be boolean")

    metrics = data.get("metrics_log")
    if not isinstance(metrics, dict):
        errors.append("metrics_log must be an object")
    else:
        for key in (
            "runtime_seconds",
            "stage_runtime_seconds",
            "token_usage",
            "model_call_count",
            "cache_hits",
            "retry_count",
            "failure_count",
            "visited_node_count",
            "checked_view_count",
            "construction_cost",
        ):
            if key not in metrics:
                errors.append(f"missing metrics_log field: {key}")

    if data.get("mode") == "stub" and isinstance(metrics, dict):
        if numeric(metrics.get("model_call_count")) not in (None, 0.0):
            errors.append("stub result cannot report live model calls")
    return errors


def selected_view_ids(data: dict[str, Any]) -> list[str]:
    selected = as_string_list(data.get("selected_views"))
    result = data.get("result") if isinstance(data.get("result"), dict) else {}
    result_view = result.get("selected_view_id")
    if result_view is not None and str(result_view) not in selected:
        selected.append(str(result_view))
    return selected


def normalized_labels(values: Iterable[Any]) -> set[str]:
    return {str(value).strip().lower() for value in values if str(value).strip()}


def expected_object_ids(query: dict[str, Any]) -> set[str]:
    values = as_string_list(query.get("expected_object_ids"))
    single = query.get("expected_object_id")
    if single is not None:
        values.append(str(single))
    return {value for value in values if value}


def result_object_id(result: dict[str, Any]) -> str | None:
    bbox = result.get("bbox_3d")
    if isinstance(bbox, dict) and bbox.get("object_id") is not None:
        return str(bbox.get("object_id"))
    node_id = result.get("selected_node_id")
    if isinstance(node_id, str):
        match = re.search(r"object_v\d+_(\d+)_", node_id)
        if match:
            return str(int(match.group(1)))
    return None


def box_bounds(box: Any) -> tuple[list[float], list[float]] | None:
    if not isinstance(box, dict):
        return None
    if isinstance(box.get("min"), list) and isinstance(box.get("max"), list):
        lower = box["min"]
        upper = box["max"]
    elif isinstance(box.get("center"), list):
        size = box.get("size", box.get("dimensions"))
        if not isinstance(size, list):
            return None
        center = box["center"]
        if len(center) != 3 or len(size) != 3:
            return None
        lower = [float(center[i]) - float(size[i]) / 2 for i in range(3)]
        upper = [float(center[i]) + float(size[i]) / 2 for i in range(3)]
    else:
        return None
    if len(lower) != 3 or len(upper) != 3:
        return None
    try:
        lo = [float(value) for value in lower]
        hi = [float(value) for value in upper]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) for value in lo + hi):
        return None
    if any(hi[i] <= lo[i] for i in range(3)):
        return None
    return lo, hi


def bbox_iou_3d(predicted: Any, expected: Any) -> float | None:
    pred_bounds = box_bounds(predicted)
    gt_bounds = box_bounds(expected)
    if pred_bounds is None or gt_bounds is None:
        return None
    pred_lo, pred_hi = pred_bounds
    gt_lo, gt_hi = gt_bounds
    intersection = 1.0
    pred_volume = 1.0
    gt_volume = 1.0
    for index in range(3):
        intersection *= max(0.0, min(pred_hi[index], gt_hi[index]) - max(pred_lo[index], gt_lo[index]))
        pred_volume *= pred_hi[index] - pred_lo[index]
        gt_volume *= gt_hi[index] - gt_lo[index]
    union = pred_volume + gt_volume - intersection
    return round(intersection / union, 6) if union > 0 else None


def expected_3d_boxes(query: dict[str, Any]) -> list[Any]:
    boxes = query.get("expected_bboxes_3d")
    if isinstance(boxes, list) and boxes:
        return boxes
    box = query.get("expected_bbox_3d")
    return [box] if box else []


def best_bbox_iou_3d(predicted: Any, query: dict[str, Any]) -> float | None:
    values = [
        value
        for box in expected_3d_boxes(query)
        if (value := bbox_iou_3d(predicted, box)) is not None
    ]
    return max(values) if values else None


def run_depth_reliable(run_config: dict[str, Any]) -> bool:
    direct = run_config.get("depth_validation")
    if isinstance(direct, dict) and direct.get("status") == "reliable":
        return True
    bbox_eval = run_config.get("bbox_3d_evaluation")
    if isinstance(bbox_eval, dict) and bbox_eval.get("status") in {
        "coarse_manual_gt_enabled",
        "official_dataset_gt_enabled",
        "reliable",
    }:
        return True
    manifest = run_config.get("scene_manifest")
    if isinstance(manifest, dict):
        validation = manifest.get("depth_validation")
        return isinstance(validation, dict) and validation.get("status") == "reliable"
    return False


def official_dataset_bbox_gt_enabled(run_config: dict[str, Any]) -> bool:
    bbox_eval = run_config.get("bbox_3d_evaluation")
    return isinstance(bbox_eval, dict) and bbox_eval.get("status") == "official_dataset_gt_enabled"


def bbox_3d_metrics_enabled(run_config: dict[str, Any]) -> bool:
    bbox_eval = run_config.get("bbox_3d_evaluation")
    return not (
        isinstance(bbox_eval, dict)
        and bbox_eval.get("status") == "unavailable_no_predicted_box"
    )


def bbox_evaluation_warning(run_config: dict[str, Any]) -> str | None:
    bbox_eval = run_config.get("bbox_3d_evaluation")
    if not isinstance(bbox_eval, dict):
        return None
    if bbox_eval.get("status") != "coarse_manual_gt_enabled":
        return None
    source = bbox_eval.get("gt_source", "manual bbox GT")
    return (
        f"3D IoU uses coarse manual depth-projected GT boxes from {source}; "
        "this is an internal regression signal, not official dataset GT."
    )


def warning_text(data: dict[str, Any]) -> str:
    warnings = data.get("warnings")
    return " ".join(str(item) for item in warnings).lower() if isinstance(warnings, list) else ""


def query_has_accuracy_gt(query: dict[str, Any]) -> bool:
    return str(query.get("verification_status", "")).startswith("verified_")


def normalized_query_type(query_type: Any) -> str:
    value = str(query_type or "unknown")
    return QUERY_TYPE_ALIASES.get(value, value)


def evaluate_one(
    query: dict[str, Any],
    data: dict[str, Any],
    *,
    depth_reliable: bool,
    official_bbox_gt: bool = False,
) -> dict[str, Any]:
    schema_errors = validate_result(data, query)
    result = data.get("result") if isinstance(data.get("result"), dict) else {}
    metrics = data.get("metrics_log") if isinstance(data.get("metrics_log"), dict) else {}
    result_available = data.get("availability_status", "available") == "available"
    gt_available = query_has_accuracy_gt(query)
    accuracy_eligible = result_available and gt_available and not schema_errors
    expected_negative = query.get("query_type") == "negative" or query.get("expected_output_type") == "not_found"
    found = result.get("found") if isinstance(result.get("found"), bool) else None
    retrieval_success = ((found is False) if expected_negative else (found is True)) if accuracy_eligible else None

    expected_views = set(as_string_list(query.get("expected_view_ids")))
    expected_nodes = set(as_string_list(query.get("expected_node_ids")))
    expected_zones = set(as_string_list(query.get("expected_zone_ids")))
    actual_views = set(selected_view_ids(data))
    actual_node = result.get("selected_node_id")
    visited = set(as_string_list(data.get("visited_nodes")))
    actual_labels = normalized_labels([result.get("matched_object")])
    expected_labels = normalized_labels(query.get("expected_object_labels", []))
    expected_ids = expected_object_ids(query)
    actual_object_id = result_object_id(result)

    view_hit = (bool(expected_views & actual_views) if expected_views else None) if accuracy_eligible else None
    node_hit = (
        str(actual_node) in expected_nodes if expected_nodes and actual_node is not None else (False if expected_nodes else None)
    ) if accuracy_eligible else None
    zone_hit = (bool(expected_zones & visited) if expected_zones else None) if accuracy_eligible else None
    object_hit = None
    object_id_hit = None
    if accuracy_eligible and not expected_negative:
        object_hit = (
            bool(expected_labels & actual_labels)
            if expected_labels and found
            else (None if not expected_labels else False)
        )
        object_id_hit = (
            actual_object_id in expected_ids
            if expected_ids and actual_object_id is not None
            else (False if expected_ids else None)
        )
    not_found_correct = (found is False if expected_negative else None) if accuracy_eligible else None

    iou = None
    iou_eligible = bool(
        result_available
        and depth_reliable
        and query.get("gt_3d_reliable") is True
        and expected_3d_boxes(query)
    )
    if iou_eligible:
        iou = best_bbox_iou_3d(result.get("bbox_3d"), query)

    categories: set[str] = set()
    warnings = warning_text(data)
    verification = query.get("verification_status")
    if schema_errors:
        categories.add("baseline adapter issue")
    if verification == "ambiguous":
        categories.add("ambiguous ground truth")
    elif verification == "missing_gt":
        categories.add("missing GT")
    if (
        not official_bbox_gt
        and ("invalid depth" in warnings or "constant depth" in warnings or "unreliable depth" in warnings)
    ):
        categories.add("invalid depth")
    if "missing object in semantic description" in warnings:
        categories.add("missing object in semantic description")
    if "baseline adapter" in warnings:
        categories.add("baseline adapter issue")

    failed = accuracy_eligible and (
        retrieval_success is False
        or view_hit is False
        or node_hit is False
        or zone_hit is False
        or object_hit is False
        or object_id_hit is False
    )
    if failed:
        if not isinstance(data.get("structured_plan"), dict) or not data.get("structured_plan"):
            categories.add("bad query parsing")
        if expected_zones and zone_hit is False:
            categories.add("wrong room / zone")
        if expected_views and view_hit is False:
            categories.add("wrong view")
        if (expected_labels and found and object_hit is False) or object_id_hit is False:
            categories.add("wrong object")
        if not expected_negative and (not visited or found is False):
            categories.add("bad traversal")
        if expected_negative and found is True:
            categories.add("wrong object")
    if iou_eligible and (iou is None or iou < 0.5):
        categories.add("bad bbox")

    token_usage = metrics.get("token_usage")
    return {
        "query_id": query.get("query_id"),
        "scene_id": query.get("scene_id"),
        "source_dataset": query.get("source_dataset"),
        "query_type": query.get("query_type"),
        "mode": data.get("mode"),
        "status": "invalid_schema" if schema_errors else ("evaluated" if result_available else "unavailable"),
        "ground_truth_available": gt_available,
        "accuracy_metrics_status": (
            "eligible" if accuracy_eligible else ("unavailable_result" if not result_available else "unavailable_gt")
        ),
        "retrieval_success": retrieval_success,
        "expected_view_hit": view_hit,
        "expected_node_hit": node_hit,
        "expected_zone_hit": zone_hit,
        "expected_object_hit": object_hit,
        "expected_object_id_hit": object_id_hit,
        "expected_object_ids": sorted(expected_ids),
        "actual_object_id": actual_object_id,
        "not_found_correct": not_found_correct,
        "runtime_seconds": numeric(metrics.get("runtime_seconds")),
        "token_usage": token_usage if isinstance(token_usage, dict) else None,
        "context_size_chars": numeric(metrics.get("context_size_chars")),
        "prompt_size_chars": numeric(metrics.get("prompt_size_chars")),
        "estimated_input_tokens": numeric(metrics.get("estimated_input_tokens")),
        "checked_view_count": numeric(metrics.get("checked_view_count")),
        "visited_node_count": numeric(metrics.get("visited_node_count")),
        "bbox_3d_iou": iou,
        "bbox_3d_iou_eligible": iou_eligible,
        "failure_categories": sorted(categories),
        "schema_errors": schema_errors,
    }


def aggregate_metric(rows: list[dict[str, Any]], field: str, eligible: Any = None) -> dict[str, Any]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        if eligible is not None and not eligible(row):
            continue
        values.append(bool(value))
    numerator = sum(values)
    return {
        "value": rate(numerator, len(values)),
        "numerator": numerator,
        "denominator": len(values),
        "status": "measured" if values else "unavailable",
        "reason": None if values else "No schema-valid result with applicable ground truth",
    }


def aggregate_tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"input": 0, "output": 0, "total": 0}
    available = 0
    seen_fields = set()
    tokenizers: set[str] = set()
    count_types: set[str] = set()
    billing_flags: set[bool] = set()
    aliases = {
        "input": ("input", "input_tokens", "prompt_tokens"),
        "output": ("output", "output_tokens", "completion_tokens"),
        "total": ("total", "total_tokens"),
    }
    for row in rows:
        usage = row.get("token_usage")
        if not isinstance(usage, dict):
            continue
        if usage.get("tokenizer") is not None:
            tokenizers.add(str(usage["tokenizer"]))
        if usage.get("token_count_type") is not None:
            count_types.add(str(usage["token_count_type"]))
        if isinstance(usage.get("provider_billing_tokens"), bool):
            billing_flags.add(bool(usage["provider_billing_tokens"]))
        row_available = False
        for target, keys in aliases.items():
            for key in keys:
                value = numeric(usage.get(key))
                if value is not None:
                    totals[target] += int(value)
                    seen_fields.add(target)
                    row_available = True
                    break
        if row_available:
            available += 1
    return {
        "input": totals["input"] if "input" in seen_fields else None,
        "output": totals["output"] if "output" in seen_fields else None,
        "total": totals["total"] if "total" in seen_fields else None,
        "denominator": available,
        "status": "measured" if available else "unavailable",
        "tokenizers": sorted(tokenizers),
        "token_count_types": sorted(count_types),
        "provider_billing_tokens": (
            next(iter(billing_flags)) if len(billing_flags) == 1 else None
        ),
    }


def aggregate_numeric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [value for row in rows if (value := numeric(row.get(field))) is not None]
    return {
        "mean": mean(values),
        "total": round(sum(values), 4) if values else None,
        "denominator": len(values),
        "status": "measured" if values else "unavailable",
    }


def aggregate_iou_threshold(ious: list[float], threshold: float) -> dict[str, Any]:
    numerator = sum(value >= threshold for value in ious)
    denominator = len(ious)
    return {
        "value": round(numerator / denominator, 4) if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "threshold": threshold,
        "status": "measured" if denominator else "N/A",
        "reason": None if denominator else "Reliable GT 3D boxes and predictions were not both available",
    }


def eligible_iou_values(rows: Iterable[dict[str, Any]]) -> list[float]:
    """Return one IoU per GT-eligible query, using zero for no valid prediction."""
    values: list[float] = []
    for row in rows:
        if not row.get("bbox_3d_iou_eligible"):
            continue
        value = numeric(row.get("bbox_3d_iou"))
        values.append(float(value) if value is not None else 0.0)
    return values


def source_dataset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source_dataset") or "unspecified")
        grouped[source].append(row)
    result: dict[str, Any] = {}
    for source, source_rows in sorted(grouped.items()):
        evaluated = [row for row in source_rows if row.get("status") == "evaluated"]
        ious = eligible_iou_values(evaluated)
        result[source] = {
            "query_count": len(source_rows),
            "evaluated_count": len(evaluated),
            "retrieval_success": aggregate_metric(evaluated, "retrieval_success"),
            "expected_object_hit": aggregate_metric(evaluated, "expected_object_hit"),
            "expected_object_id_hit": aggregate_metric(evaluated, "expected_object_id_hit"),
            "expected_view_hit": aggregate_metric(evaluated, "expected_view_hit"),
            "expected_node_hit": aggregate_metric(evaluated, "expected_node_hit"),
            "bbox_3d_iou": {
                "value": mean(ious),
                "denominator": len(ious),
                "status": "measured" if ious else "N/A",
            },
            "bbox_acc_at_0_1": aggregate_iou_threshold(ious, 0.1),
            "bbox_acc_at_0_25": aggregate_iou_threshold(ious, 0.25),
            "bbox_acc_at_0_5": aggregate_iou_threshold(ious, 0.5),
            "runtime_seconds": aggregate_numeric(evaluated, "runtime_seconds"),
            "checked_view_count": aggregate_numeric(evaluated, "checked_view_count"),
            "visited_node_count": aggregate_numeric(evaluated, "visited_node_count"),
            "context_size_chars": aggregate_numeric(evaluated, "context_size_chars"),
            "estimated_input_tokens": aggregate_numeric(evaluated, "estimated_input_tokens"),
        }
    return result


def compact_metrics(
    queries: Iterable[dict[str, Any]],
    per_query: list[dict[str, Any]],
) -> dict[str, Any]:
    query_list = list(queries)
    rows_by_id = {str(row.get("query_id")): row for row in per_query if row.get("query_id") is not None}
    per_type: dict[str, dict[str, Any]] = {}
    raw_per_type: dict[str, dict[str, Any]] = {}

    for query_type in QUERY_TYPE_STRATA:
        typed_queries = [
            query for query in query_list if normalized_query_type(query.get("query_type")) == query_type
        ]
        typed_ids = {str(query.get("query_id")) for query in typed_queries if query.get("query_id") is not None}
        typed_rows = [row for query_id, row in rows_by_id.items() if query_id in typed_ids]
        schema_valid_rows = [
            row for row in typed_rows if row.get("status") in {"evaluated", "unavailable"}
        ]
        evaluated_rows = [row for row in typed_rows if row.get("status") == "evaluated"]
        per_type[query_type] = {
            "total_queries": len(typed_queries),
            "runnable_queries": sum(row.get("status") != "missing_result" for row in typed_rows),
            "gt_eligible_queries": sum(query_has_accuracy_gt(query) for query in typed_queries),
            "schema_valid_count": len(schema_valid_rows),
            "retrieval_success": aggregate_metric(evaluated_rows, "retrieval_success"),
            "expected_view_hit": aggregate_metric(evaluated_rows, "expected_view_hit"),
            "expected_node_hit": aggregate_metric(evaluated_rows, "expected_node_hit"),
            "expected_zone_hit": aggregate_metric(evaluated_rows, "expected_zone_hit"),
        }

    for query_type in sorted({str(query.get("query_type", "unknown")) for query in query_list}):
        typed_queries = [query for query in query_list if str(query.get("query_type", "unknown")) == query_type]
        typed_ids = {str(query.get("query_id")) for query in typed_queries if query.get("query_id") is not None}
        typed_rows = [row for query_id, row in rows_by_id.items() if query_id in typed_ids]
        schema_valid_rows = [
            row for row in typed_rows if row.get("status") in {"evaluated", "unavailable"}
        ]
        evaluated_rows = [row for row in typed_rows if row.get("status") == "evaluated"]
        raw_per_type[query_type] = {
            "total_queries": len(typed_queries),
            "runnable_queries": sum(row.get("status") != "missing_result" for row in typed_rows),
            "gt_eligible_queries": sum(query_has_accuracy_gt(query) for query in typed_queries),
            "schema_valid_count": len(schema_valid_rows),
            "retrieval_success": aggregate_metric(evaluated_rows, "retrieval_success"),
            "expected_object_id_hit": aggregate_metric(evaluated_rows, "expected_object_id_hit"),
            "not_found_correctness": aggregate_metric(evaluated_rows, "not_found_correct"),
        }

    return {
        "total_queries": len(query_list),
        "runnable_queries": sum(row.get("status") != "missing_result" for row in per_query),
        "gt_eligible_queries": sum(query_has_accuracy_gt(query) for query in query_list),
        "schema_valid_count": sum(
            row.get("status") in {"evaluated", "unavailable"} for row in per_query
        ),
        "query_type_normalization": QUERY_TYPE_ALIASES,
        "per_query_type": per_type,
        "raw_per_query_type": raw_per_type,
    }


def evaluate(
    benchmark_path: Path,
    run_dir: Path,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    benchmark = load_json(benchmark_path)
    queries = benchmark.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Benchmark file must contain a queries list")
    all_query_by_id = {
        str(query["query_id"]): query
        for query in queries
        if isinstance(query, dict) and query.get("query_id")
    }

    run_config_path = run_dir / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.exists() else {}
    benchmark_to_scene_id: dict[str, str] = {}
    configured_scenes = run_config.get("scenes")
    if isinstance(configured_scenes, list):
        for scene in configured_scenes:
            if not isinstance(scene, dict) or scene.get("scene_id") is None:
                continue
            scene_id = str(scene["scene_id"])
            benchmark_scene_id = str(scene.get("benchmark_scene_id") or scene_id)
            benchmark_to_scene_id[benchmark_scene_id] = scene_id

    configured_benchmark_scene_ids = set(benchmark_to_scene_id)
    if not configured_benchmark_scene_ids:
        configured_benchmark_scene_ids = {
            str(scene_id)
            for scene_id in run_config.get("scene_ids", [])
            if scene_id is not None
        }

    query_by_id: dict[str, dict[str, Any]] = {}
    for query_id, query in all_query_by_id.items():
        benchmark_scene_id = str(query.get("scene_id"))
        if configured_benchmark_scene_ids and benchmark_scene_id not in configured_benchmark_scene_ids:
            continue
        effective_query = dict(query)
        scene_id = benchmark_to_scene_id.get(benchmark_scene_id)
        if scene_id is not None and scene_id != benchmark_scene_id:
            effective_query["benchmark_scene_id"] = benchmark_scene_id
            effective_query["scene_id"] = scene_id
        query_by_id[query_id] = effective_query
    excluded_query_count = len(all_query_by_id) - len(query_by_id)
    depth_reliable = run_depth_reliable(run_config)
    bbox_metrics_enabled = bbox_3d_metrics_enabled(run_config)
    official_bbox_gt = official_dataset_bbox_gt_enabled(run_config)
    results_dir = results_dir or run_dir / "query_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing query_results directory: {results_dir}")

    result_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    warnings: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read {path.name}: {exc}")
            continue
        query_id = str(data.get("query_id") or path.stem)
        if query_id in result_by_id:
            warnings.append(f"Duplicate result for {query_id}; kept {result_by_id[query_id][0].name}")
            continue
        result_by_id[query_id] = (path, data)

    unknown_ids = sorted(set(result_by_id) - set(query_by_id))
    if unknown_ids:
        warnings.append(f"Ignored results absent from benchmark: {', '.join(unknown_ids)}")

    per_query: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    for query_id, query in query_by_id.items():
        saved = result_by_id.get(query_id)
        if saved is None:
            missing_ids.append(query_id)
            per_query.append({
                "query_id": query_id,
                "scene_id": query.get("scene_id"),
                "source_dataset": query.get("source_dataset"),
                "query_type": query.get("query_type"),
                "status": "missing_result",
                "failure_categories": [],
                "schema_errors": [],
            })
            continue
        row = evaluate_one(
            query,
            saved[1],
            depth_reliable=depth_reliable and bbox_metrics_enabled,
            official_bbox_gt=official_bbox_gt,
        )
        try:
            result_file = saved[0].relative_to(run_dir)
        except ValueError:
            result_file = saved[0]
        row["result_file"] = str(result_file).replace("\\", "/")
        per_query.append(row)

    evaluated_rows = [row for row in per_query if row.get("status") == "evaluated"]
    unavailable_rows = [row for row in per_query if row.get("status") == "unavailable"]
    schema_valid_rows = evaluated_rows + unavailable_rows
    present_rows = [row for row in per_query if row.get("status") != "missing_result"]
    mode_counts = Counter(str(row.get("mode")) for row in schema_valid_rows)
    failure_counts = Counter(category for row in present_rows for category in row.get("failure_categories", []))
    all_failure_counts = {category: failure_counts.get(category, 0) for category in FAILURE_CATEGORIES}

    runtimes = [value for row in evaluated_rows if (value := numeric(row.get("runtime_seconds"))) is not None]
    checked = [value for row in evaluated_rows if (value := numeric(row.get("checked_view_count"))) is not None]
    visited = [value for row in evaluated_rows if (value := numeric(row.get("visited_node_count"))) is not None]
    ious = eligible_iou_values(evaluated_rows)
    missing_bbox_prediction_count = sum(
        row.get("bbox_3d_iou_eligible") and row.get("bbox_3d_iou") is None
        for row in evaluated_rows
    )

    if missing_ids:
        warnings.append(f"{len(missing_ids)} benchmark queries have no saved result")
    bbox_warning = bbox_evaluation_warning(run_config)
    if bbox_warning:
        warnings.append(bbox_warning)
    if not bbox_metrics_enabled:
        warnings.append("3D IoU is N/A because the run does not emit predicted 3D boxes")
    elif not depth_reliable:
        warnings.append("3D IoU is N/A because run_config does not confirm reliable depth")
    if unavailable_rows:
        warnings.append(f"{len(unavailable_rows)} query results are unavailable and excluded from measured metrics")
    unavailable_gt_count = sum(not row.get("ground_truth_available", False) for row in schema_valid_rows)
    if unavailable_gt_count:
        warnings.append(f"{unavailable_gt_count} query results lack GT and are excluded from accuracy metrics")

    runtime_status = "measured" if runtimes else "unavailable"
    checked_status = "measured" if checked else "unavailable"
    visited_status = "measured" if visited else "unavailable"

    summary = {
        "schema_version": METRICS_SCHEMA,
        "run_id": str(run_config.get("run_id") or run_dir.name),
        "benchmark_id": benchmark.get("benchmark_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "benchmark_total_query_count": len(all_query_by_id),
            "benchmark_query_count": len(query_by_id),
            "excluded_scene_query_count": excluded_query_count,
            "result_count": len(present_rows),
            "missing_result_count": len(missing_ids),
            "schema_valid_result_count": len(schema_valid_rows),
            "evaluated_result_count": len(evaluated_rows),
            "unavailable_result_count": len(unavailable_rows),
            "accuracy_eligible_result_count": sum(
                row.get("accuracy_metrics_status") == "eligible" for row in evaluated_rows
            ),
            "ground_truth_unavailable_result_count": unavailable_gt_count,
        },
        "mode_counts": dict(sorted(mode_counts.items())),
        "compact_metrics": compact_metrics(query_by_id.values(), per_query),
        "per_source_dataset": source_dataset_metrics(per_query),
        "metrics": {
            "retrieval_success": aggregate_metric(evaluated_rows, "retrieval_success"),
            "expected_view_hit": aggregate_metric(evaluated_rows, "expected_view_hit"),
            "expected_node_hit": aggregate_metric(evaluated_rows, "expected_node_hit"),
            "expected_zone_hit": aggregate_metric(evaluated_rows, "expected_zone_hit"),
            "expected_object_hit": aggregate_metric(evaluated_rows, "expected_object_hit"),
            "expected_object_id_hit": aggregate_metric(evaluated_rows, "expected_object_id_hit"),
            "not_found_correctness": aggregate_metric(evaluated_rows, "not_found_correct"),
            "runtime_seconds": {
                "mean": mean(runtimes),
                "total": round(sum(runtimes), 4) if runtimes else None,
                "denominator": len(runtimes),
                "status": runtime_status,
            },
            "token_usage": aggregate_tokens(evaluated_rows),
            "context_size_chars": aggregate_numeric(evaluated_rows, "context_size_chars"),
            "prompt_size_chars": aggregate_numeric(evaluated_rows, "prompt_size_chars"),
            "estimated_input_tokens": aggregate_numeric(evaluated_rows, "estimated_input_tokens"),
            "checked_view_count": {
                "mean": mean(checked),
                "total": int(sum(checked)) if checked else None,
                "denominator": len(checked),
                "status": checked_status,
            },
            "visited_node_count": {
                "mean": mean(visited),
                "total": int(sum(visited)) if visited else None,
                "denominator": len(visited),
                "status": visited_status,
            },
            "bbox_3d_iou": {
                "value": mean(ious),
                "status": "measured" if ious else "N/A",
                "denominator": len(ious),
                "missing_prediction_count": missing_bbox_prediction_count,
                "reason": None if ious else "Reliable depth and reliable GT 3D boxes were not both available",
            },
            "bbox_acc_at_0_1": aggregate_iou_threshold(ious, 0.1),
            "bbox_acc_at_0_25": aggregate_iou_threshold(ious, 0.25),
            "bbox_acc_at_0_5": aggregate_iou_threshold(ious, 0.5),
        },
        "failure_categories": all_failure_counts,
        "missing_query_ids": missing_ids,
        "per_query": per_query,
        "warnings": warnings,
    }
    return summary


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summary_markdown(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    compact = summary.get("compact_metrics", {})
    metrics = summary["metrics"]
    lines = [
        "# Metrics Summary",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Benchmark: `{summary.get('benchmark_id')}`",
        f"- Results: {coverage['result_count']}/{coverage['benchmark_query_count']}",
        f"- Schema-valid results: {coverage['schema_valid_result_count']}",
        f"- Evaluated results: {coverage.get('evaluated_result_count', 0)}",
        f"- Unavailable results: {coverage.get('unavailable_result_count', 0)}",
        f"- Accuracy-eligible results: {coverage.get('accuracy_eligible_result_count', 0)}",
        f"- Queries excluded by scene scope: {coverage.get('excluded_scene_query_count', 0)}",
        f"- Modes: `{json.dumps(summary.get('mode_counts', {}), sort_keys=True)}`",
        "",
        "## Gate Coverage",
        "",
        "| Gate metric | Status | Value |",
        "|---|---|---:|",
        f"| result output coverage | measured | {coverage['result_count']} / {coverage['benchmark_query_count']} |",
        f"| canonical schema validity | measured | {coverage['schema_valid_result_count']} / {coverage['result_count']} |",
        "",
        "## Query Type Coverage",
        "",
        "| Query type | Total | Runnable | GT-eligible | Schema-valid | Retrieval hit rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    per_query_type = compact.get("per_query_type", {}) if isinstance(compact, dict) else {}
    for query_type in QUERY_TYPE_STRATA:
        record = per_query_type.get(query_type, {}) if isinstance(per_query_type, dict) else {}
        hit_rate = record.get("retrieval_success", {}) if isinstance(record, dict) else {}
        lines.append(
            f"| {query_type} | {record.get('total_queries', 0)} | "
            f"{record.get('runnable_queries', 0)} | "
            f"{record.get('gt_eligible_queries', 0)} | "
            f"{record.get('schema_valid_count', 0)} | "
            f"{format_value(hit_rate.get('value'))} |"
        )
    raw_per_query_type = compact.get("raw_per_query_type", {}) if isinstance(compact, dict) else {}
    lines.extend([
        "",
        "## Raw Query Type Coverage",
        "",
        "| Query type | Total | Runnable | GT-eligible | Schema-valid | Retrieval hit rate | Object-ID hit rate | Not-found correctness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if isinstance(raw_per_query_type, dict) and raw_per_query_type:
        for query_type, record in raw_per_query_type.items():
            hit_rate = record.get("retrieval_success", {}) if isinstance(record, dict) else {}
            object_id_rate = record.get("expected_object_id_hit", {}) if isinstance(record, dict) else {}
            not_found_rate = record.get("not_found_correctness", {}) if isinstance(record, dict) else {}
            lines.append(
                f"| {query_type} | {record.get('total_queries', 0)} | "
                f"{record.get('runnable_queries', 0)} | "
                f"{record.get('gt_eligible_queries', 0)} | "
                f"{record.get('schema_valid_count', 0)} | "
                f"{format_value(hit_rate.get('value'))} | "
                f"{format_value(object_id_rate.get('value'))} | "
                f"{format_value(not_found_rate.get('value'))} |"
            )
    else:
        lines.append("| N/A | 0 | 0 | 0 | 0 | N/A | N/A | N/A |")
    lines.extend([
        "",
        "## Source Dataset Metrics",
        "",
        "Missing or invalid 3D-box predictions count as IoU 0 and remain in every Acc@k denominator.",
        "",
        "| Source | Queries | Object-ID hit | Acc@0.1 | Acc@0.25 | Acc@0.5 | Runtime mean (s) | Estimated input tokens mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    per_source = summary.get("per_source_dataset", {})
    if isinstance(per_source, dict) and per_source:
        for source, record in per_source.items():
            lines.append(
                f"| {source} | {record.get('query_count', 0)} | "
                f"{format_value(record.get('expected_object_id_hit', {}).get('value'))} | "
                f"{format_value(record.get('bbox_acc_at_0_1', {}).get('value'))} | "
                f"{format_value(record.get('bbox_acc_at_0_25', {}).get('value'))} | "
                f"{format_value(record.get('bbox_acc_at_0_5', {}).get('value'))} | "
                f"{format_value(record.get('runtime_seconds', {}).get('mean'))} | "
                f"{format_value(record.get('estimated_input_tokens', {}).get('mean'))} |"
            )
    else:
        lines.append("| N/A | 0 | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines.extend([
        "",
        "## Metrics",
        "",
        "`measured` means the denominator contains applicable records. `unavailable` means no executable or GT-eligible record exists. `N/A` is reserved for metrics whose required modality or GT is absent.",
        "",
        "| Metric | Status | Value | Numerator / denominator |",
        "|---|---|---:|---:|",
    ])
    for key in (
        "retrieval_success",
        "expected_view_hit",
        "expected_node_hit",
        "expected_zone_hit",
        "expected_object_hit",
        "expected_object_id_hit",
        "not_found_correctness",
    ):
        record = metrics[key]
        lines.append(
            f"| {key} | {record.get('status', 'unavailable')} | {format_value(record['value'])} | "
            f"{record['numerator']} / {record['denominator']} |"
        )
    lines.extend([
        f"| runtime_seconds mean | {metrics['runtime_seconds']['status']} | {format_value(metrics['runtime_seconds']['mean'])} | {metrics['runtime_seconds']['denominator']} records |",
        f"| checked_view_count mean | {metrics['checked_view_count']['status']} | {format_value(metrics['checked_view_count']['mean'])} | {metrics['checked_view_count']['denominator']} records |",
        f"| visited_node_count mean | {metrics['visited_node_count']['status']} | {format_value(metrics['visited_node_count']['mean'])} | {metrics['visited_node_count']['denominator']} records |",
        f"| context_size_chars mean | {metrics['context_size_chars']['status']} | {format_value(metrics['context_size_chars']['mean'])} | {metrics['context_size_chars']['denominator']} records |",
        f"| estimated_input_tokens mean | {metrics['estimated_input_tokens']['status']} | {format_value(metrics['estimated_input_tokens']['mean'])} | {metrics['estimated_input_tokens']['denominator']} records |",
        f"| bbox_3d_iou | {metrics['bbox_3d_iou']['status']} | {format_value(metrics['bbox_3d_iou']['value'])} | {metrics['bbox_3d_iou']['denominator']} records |",
        f"| Acc@0.1 | {metrics['bbox_acc_at_0_1']['status']} | {format_value(metrics['bbox_acc_at_0_1']['value'])} | {metrics['bbox_acc_at_0_1']['numerator']} / {metrics['bbox_acc_at_0_1']['denominator']} |",
        f"| Acc@0.25 | {metrics['bbox_acc_at_0_25']['status']} | {format_value(metrics['bbox_acc_at_0_25']['value'])} | {metrics['bbox_acc_at_0_25']['numerator']} / {metrics['bbox_acc_at_0_25']['denominator']} |",
        f"| Acc@0.5 | {metrics['bbox_acc_at_0_5']['status']} | {format_value(metrics['bbox_acc_at_0_5']['value'])} | {metrics['bbox_acc_at_0_5']['numerator']} / {metrics['bbox_acc_at_0_5']['denominator']} |",
        "",
        "## Measured Token Usage",
        "",
        "Token semantics are recorded below. `provider_billing_tokens: false` means local tokenizer/model tokens, not API usage or cost.",
        "",
        "```json",
        json.dumps(metrics["token_usage"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Estimated Context And Tokens",
        "",
        "```json",
        json.dumps(
            {
                "context_size_chars": metrics["context_size_chars"],
                "prompt_size_chars": metrics["prompt_size_chars"],
                "estimated_input_tokens": metrics["estimated_input_tokens"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        "## Warnings",
        "",
    ])
    lines.extend(f"- {warning}" for warning in summary.get("warnings", []))
    if not summary.get("warnings"):
        lines.append("- None")
    return "\n".join(lines) + "\n"


def failure_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Failure Modes",
        "",
        f"Run: `{summary['run_id']}`",
        "",
        "Missing result files are coverage gaps and are not assigned a semantic failure category.",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category in FAILURE_CATEGORIES:
        lines.append(f"| {category} | {summary['failure_categories'].get(category, 0)} |")
    lines.extend(["", "## Per-query failures", ""])
    failed = [row for row in summary.get("per_query", []) if row.get("failure_categories") or row.get("schema_errors")]
    if failed:
        for row in failed:
            categories = ", ".join(row.get("failure_categories", [])) or "schema error"
            lines.append(f"- `{row.get('query_id')}`: {categories}")
            for error in row.get("schema_errors", []):
                lines.append(f"  - {error}")
    else:
        lines.append("- No classified failures in available schema-valid results.")
    lines.extend(["", "## Coverage gaps", ""])
    missing = summary.get("missing_query_ids", [])
    lines.append(f"Missing results ({len(missing)}): {', '.join(missing) if missing else 'none'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate canonical SemanticSplat query results")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("docs/benchmarks/benchmark_queries_v1.json"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--results", type=Path, help="Explicit query_results directory")
    parser.add_argument("--out", type=Path, help="Output directory containing run_config.json")
    args = parser.parse_args()

    if args.run_dir is not None:
        output_dir = (args.out or args.run_dir).resolve()
        results_dir = args.results.resolve() if args.results is not None else None
    elif args.results is not None and args.out is not None:
        output_dir = args.out.resolve()
        results_dir = args.results.resolve()
    else:
        parser.error("use --run-dir, or provide both --results and --out")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate(args.benchmark.resolve(), output_dir, results_dir=results_dir)
    json_path = output_dir / "metrics_summary.json"
    markdown_path = output_dir / "metrics_summary.md"
    failures_path = output_dir / "failure_modes.md"
    write_json(json_path, summary)
    markdown_path.write_text(summary_markdown(summary), encoding="utf-8")
    failures_path.write_text(failure_markdown(summary), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {failures_path}")


if __name__ == "__main__":
    main()
