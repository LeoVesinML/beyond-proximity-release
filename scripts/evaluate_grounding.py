#!/usr/bin/env python3
"""BBQ-aligned grounding evaluation for canonical SemanticSplat results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.evaluate_results import bbox_iou_3d, box_bounds
except ModuleNotFoundError:  # Direct execution: python scripts/evaluate_grounding.py
    from evaluate_results import bbox_iou_3d, box_bounds


SCHEMA_VERSION = "semanticsplat.grounding_metrics.v2"
CANONICAL_RESULT_SCHEMA = "semanticsplat.query_result.v1"
THRESHOLDS = (0.1, 0.25, 0.5)
RECALL_K = (1, 3, 5)
FAILURE_TYPES = (
    "wrong object",
    "wrong relation",
    "false positive",
    "invalid/unavailable prediction",
    "bad bbox",
    "missing GT",
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    return round(sum(collected) / len(collected), 6) if collected else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _first(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def expected_boxes(query: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = query.get("expected_bboxes_3d")
    if isinstance(boxes, list):
        return [box for box in boxes if isinstance(box, dict)]
    box = query.get("expected_bbox_3d")
    return [box] if isinstance(box, dict) else []


def expected_object_ids(query: dict[str, Any]) -> set[str]:
    values = _strings(query.get("expected_object_ids"))
    if query.get("expected_object_id") is not None:
        values.append(str(query["expected_object_id"]))
    return {value for value in values if value}


def predicted_object_id(result: dict[str, Any]) -> str | None:
    value = _first(result, ("object_id", "predicted_object_id", "selected_object_id"))
    bbox = result.get("bbox_3d")
    if value is None and isinstance(bbox, dict):
        value = bbox.get("object_id")
    return str(value) if value is not None else None


def ranked_object_ids(result: dict[str, Any]) -> list[str]:
    values = result.get("ranked_object_ids")
    if isinstance(values, list):
        ranked = [str(value) for value in values if value is not None]
    else:
        candidates = result.get("ranked_candidates")
        ranked = [
            str(item["object_id"])
            for item in candidates
            if isinstance(item, dict) and item.get("object_id") is not None
        ] if isinstance(candidates, list) else []
    selected = predicted_object_id(result)
    if selected is not None and selected not in ranked:
        ranked.insert(0, selected)
    return list(dict.fromkeys(ranked))


def _rank_of_expected(ranked: list[str], expected: set[str]) -> int | None:
    for rank, object_id in enumerate(ranked, start=1):
        if object_id in expected:
            return rank
    return None


def _legacy_normalize(raw: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Expose inherited pre-v1 fields without presenting them as canonical."""
    if raw.get("schema_version") == CANONICAL_RESULT_SCHEMA:
        return raw, "canonical"
    normalized = dict(raw)
    if not isinstance(normalized.get("result"), dict):
        answer = raw.get("answer") if isinstance(raw.get("answer"), dict) else {}
        normalized["result"] = {
            "found": _first(raw, ("found", "success")),
            "bbox_3d": _first(raw, ("bbox_3d", "predicted_bbox_3d")),
            "object_id": _first(raw, ("object_id", "predicted_object_id")),
            "relation_correct": _first(raw, ("relation_correct", "relation_match")),
            **answer,
        }
    if not isinstance(normalized.get("metrics_log"), dict):
        metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
        normalized["metrics_log"] = metrics
    return normalized, "legacy_adapted"


def validate_result(
    data: dict[str, Any], query: dict[str, Any], input_shape: str
) -> list[str]:
    errors: list[str] = []
    for field in ("query_id", "scene_id"):
        if data.get(field) is None:
            errors.append(f"missing {field}")
    if data.get("query_id") is not None and str(data["query_id"]) != str(query.get("query_id")):
        errors.append("query_id does not match benchmark")
    if data.get("scene_id") is not None and str(data["scene_id"]) != str(query.get("scene_id")):
        errors.append("scene_id does not match benchmark")
    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result must be an object")
    else:
        if "found" not in result:
            errors.append("missing result.found")
        elif not isinstance(result["found"], bool):
            errors.append("result.found must be boolean")
        if "bbox_3d" not in result:
            errors.append("missing result.bbox_3d")
        if not any(
            key in result
            for key in ("object_id", "predicted_object_id", "selected_object_id", "selected_node_id")
        ) and not (
            isinstance(result.get("bbox_3d"), dict)
            and result["bbox_3d"].get("object_id") is not None
        ):
            errors.append("missing result object identifier")
    metrics = data.get("metrics_log")
    if not isinstance(metrics, dict):
        errors.append("metrics_log must be an object")
    if input_shape == "canonical":
        for field in ("runtime_seconds", "visited_node_count", "checked_view_count", "construction_cost"):
            if not isinstance(metrics, dict) or field not in metrics:
                errors.append(f"missing metrics_log.{field}")
    return errors


def _best_iou(prediction: Any, boxes: list[dict[str, Any]]) -> float | None:
    values = [
        value
        for box in boxes
        if (value := bbox_iou_3d(prediction, box)) is not None
    ]
    return max(values) if values else None


def _metric_value(metrics: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    return _number(_first(metrics, aliases))


def _search_variant(data: dict[str, Any]) -> str:
    result = data.get("result") if isinstance(data.get("result"), dict) else {}
    metrics = data.get("metrics_log") if isinstance(data.get("metrics_log"), dict) else {}
    value = (
        data.get("search_variant")
        or result.get("search_variant")
        or metrics.get("search_variant")
        or data.get("method")
    )
    return str(value or "unspecified")


def _relation_failure(query: dict[str, Any], result: dict[str, Any], object_hit: bool | None) -> bool:
    if str(query.get("query_type")) not in {"relational", "multi_hop"}:
        return False
    explicit = _first(result, ("relation_correct", "relation_match", "relation_satisfied"))
    if explicit is not None:
        return explicit is not True
    return object_hit is True and result.get("found") is not True


def evaluate_query(
    query: dict[str, Any],
    raw: dict[str, Any] | None,
    *,
    result_file: str | None = None,
) -> dict[str, Any]:
    boxes = expected_boxes(query)
    valid_gt_boxes = [box for box in boxes if box_bounds(box) is not None]
    gt_status = str(query.get("verification_status") or "")
    bbox_eligible = bool(valid_gt_boxes) and gt_status not in {"missing_gt", "ambiguous"}
    id_gt = expected_object_ids(query)
    recall_eligible = bool(id_gt) and gt_status not in {"missing_gt", "ambiguous"}
    negative_eligible = bool(
        query.get("expected_output_type") == "not_found"
        and "verified" in gt_status
        and ("absence" in gt_status or query.get("negative_gt_verified") is True)
    )
    failures: set[str] = set()

    if raw is None:
        data: dict[str, Any] = {}
        input_shape = "missing_result"
        schema_errors = ["missing result file"]
        result: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        result_valid = False
    else:
        data, input_shape = _legacy_normalize(raw)
        schema_errors = validate_result(data, query, input_shape)
        result = data.get("result") if isinstance(data.get("result"), dict) else {}
        metrics = data.get("metrics_log") if isinstance(data.get("metrics_log"), dict) else {}
        result_valid = (
            data.get("availability_status", "available") == "available"
            and not schema_errors
        )

    prediction = result.get("bbox_3d")
    prediction_valid = box_bounds(prediction) is not None
    measured_iou = _best_iou(prediction, valid_gt_boxes) if bbox_eligible and prediction_valid else None
    iou = float(measured_iou) if measured_iou is not None else (0.0 if bbox_eligible else None)
    actual_id = predicted_object_id(result)
    ranked_ids = ranked_object_ids(result)
    expected_rank = _rank_of_expected(ranked_ids, id_gt) if recall_eligible else None
    recall_hits = {
        k: (expected_rank is not None and expected_rank <= k) if recall_eligible else None
        for k in RECALL_K
    }
    reciprocal_rank = (
        (1.0 / expected_rank if expected_rank is not None else 0.0)
        if recall_eligible
        else None
    )
    negative_correct = (
        bool(result_valid and result.get("found") is False)
        if negative_eligible
        else None
    )
    prediction_available = bool(result_valid and result.get("found") is True)

    if not bbox_eligible and not recall_eligible and not negative_eligible:
        failures.add("missing GT")
    if raw is None or not result_valid:
        failures.add("invalid/unavailable prediction")
    if bbox_eligible and not prediction_valid:
        failures.add("bad bbox")
    if recall_eligible and recall_hits[1] is False and result_valid:
        failures.add("wrong object")
    if negative_eligible and negative_correct is False and result_valid:
        failures.add("false positive")
    if _relation_failure(query, result, recall_hits[1]):
        failures.add("wrong relation")

    return {
        "query_id": str(query.get("query_id")),
        "scene_id": str(query.get("scene_id") or "unspecified"),
        "source": str(query.get("source_dataset") or query.get("source") or "unspecified"),
        "query_type": str(query.get("query_type") or "unspecified"),
        "search_variant": _search_variant(data),
        "result_file": result_file,
        "input_shape": input_shape,
        "schema_valid": not schema_errors and input_shape == "canonical",
        "schema_errors": schema_errors,
        "prediction_available": prediction_available,
        "result_valid": result_valid,
        "bbox_eligible": bbox_eligible,
        "bbox_iou_3d": iou,
        "bbox_prediction_valid": prediction_valid,
        "recall_at_1_eligible": recall_eligible,
        "recall_at_1": recall_hits[1],
        "recall_at_3": recall_hits[3],
        "recall_at_5": recall_hits[5],
        "reciprocal_rank": reciprocal_rank,
        "expected_rank": expected_rank,
        "ranked_object_ids": ranked_ids,
        "negative_eligible": negative_eligible,
        "negative_correct": negative_correct,
        "expected_object_ids": sorted(id_gt),
        "predicted_object_id": actual_id,
        "failures": sorted(failures),
        "efficiency": {
            "runtime_seconds": _metric_value(metrics, ("runtime_seconds", "latency_seconds")),
            "checked_nodes": _metric_value(
                metrics, ("visited_node_count", "checked_node_count", "nodes_checked")
            ),
            "checked_views": _metric_value(
                metrics, ("checked_view_count", "views_checked")
            ),
            "checked_objects": _metric_value(
                metrics, ("checked_object_count", "objects_checked", "semantic_entries_scanned")
            ),
            "context_chars": _metric_value(
                metrics, ("context_size_chars", "context_chars", "prompt_size_chars")
            ),
            "estimated_tokens": _metric_value(
                metrics, ("estimated_input_tokens", "estimated_tokens")
            ),
        },
        "construction_cost": metrics.get("construction_cost"),
        "_segmentation": _segmentation_pair(query, result),
    }


def _extract_mask(container: dict[str, Any], names: tuple[str, ...]) -> Any:
    value = _first(container, names)
    if isinstance(value, dict):
        value = _first(value, ("mask", "labels", "array", "values"))
    return value


def _flatten_mask(value: Any) -> list[int] | None:
    flattened: list[int] = []

    def visit(item: Any) -> bool:
        if isinstance(item, list):
            return all(visit(child) for child in item)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        number = float(item)
        if not math.isfinite(number) or not number.is_integer():
            return False
        flattened.append(int(number))
        return True

    return flattened if isinstance(value, list) and visit(value) and flattened else None


def _segmentation_pair(
    query: dict[str, Any], result: dict[str, Any]
) -> tuple[list[int], list[int], set[int]] | None:
    gt = _flatten_mask(
        _extract_mask(
            query,
            ("segmentation_gt", "gt_segmentation", "ground_truth_mask", "expected_segmentation_mask"),
        )
    )
    pred = _flatten_mask(
        _extract_mask(
            result,
            ("segmentation_prediction", "predicted_segmentation", "segmentation_mask", "mask"),
        )
    )
    if gt is None or pred is None or len(gt) != len(pred):
        return None
    ignored = {
        int(value)
        for value in query.get("segmentation_ignore_labels", [])
        if _number(value) is not None
    }
    return pred, gt, ignored


def segmentation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: Counter[tuple[int, int]] = Counter()
    gt_counts: Counter[int] = Counter()
    pairs = 0
    for row in rows:
        pair = row.get("_segmentation")
        if pair is None:
            continue
        prediction, ground_truth, ignored = pair
        pairs += 1
        for pred, gt in zip(prediction, ground_truth):
            if gt in ignored:
                continue
            confusion[(gt, pred)] += 1
            gt_counts[gt] += 1
    if not pairs or not gt_counts:
        reason = (
            "Segmentation prediction and GT arrays/masks are not jointly available; "
            "object retrieval or oracle object IDs are not segmentation evidence."
        )
        return {
            key: {"status": "N/A", "value": None, "reason": reason, "query_count": 0}
            for key in ("mAcc", "mIoU", "fmIoU")
        }
    classes = set(gt_counts)
    classes.update(pred for (_, pred) in confusion)
    accuracies: list[float] = []
    ious: dict[int, float] = {}
    for label in sorted(classes):
        true_positive = confusion[(label, label)]
        gt_total = gt_counts[label]
        pred_total = sum(count for (gt, pred), count in confusion.items() if pred == label)
        union = gt_total + pred_total - true_positive
        if gt_total:
            accuracies.append(true_positive / gt_total)
        if union:
            ious[label] = true_positive / union
    total_gt = sum(gt_counts.values())
    measured = {
        "mAcc": _mean(accuracies),
        "mIoU": _mean(ious.values()),
        "fmIoU": round(
            sum((gt_counts[label] / total_gt) * ious.get(label, 0.0) for label in gt_counts),
            6,
        ),
    }
    return {
        key: {
            "status": "measured",
            "value": value,
            "reason": None,
            "query_count": pairs,
        }
        for key, value in measured.items()
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    iou_rows = [row for row in rows if row["bbox_eligible"]]
    recall_rows = [row for row in rows if row["recall_at_1_eligible"]]
    negative_rows = [row for row in rows if row["negative_eligible"]]
    ious = [float(row["bbox_iou_3d"]) for row in iou_rows]
    efficiency: dict[str, Any] = {}
    for field in (
        "runtime_seconds",
        "checked_nodes",
        "checked_views",
        "checked_objects",
        "context_chars",
        "estimated_tokens",
    ):
        values = [
            value
            for row in rows
            if (value := _number(row["efficiency"].get(field))) is not None
        ]
        efficiency[field] = {
            "mean": _mean(values),
            "total": round(sum(values), 6) if values else None,
            "denominator": len(values),
            "status": "measured" if values else "N/A",
        }
    result: dict[str, Any] = {
        "query_count": len(rows),
        "bbox_eligible_count": len(iou_rows),
        "bbox_iou_3d": {
            "value": _mean(ious),
            "denominator": len(ious),
            "status": "measured" if ious else "N/A",
        },
        "mrr_exact_object_id": {
            "value": _mean(float(row["reciprocal_rank"]) for row in recall_rows),
            "denominator": len(recall_rows),
            "status": "measured" if recall_rows else "N/A",
        },
        "negative_accuracy": {
            "value": _rate(
                sum(row["negative_correct"] is True for row in negative_rows),
                len(negative_rows),
            ),
            "numerator": sum(
                row["negative_correct"] is True for row in negative_rows
            ),
            "denominator": len(negative_rows),
            "status": "measured" if negative_rows else "N/A",
        },
        "efficiency": efficiency,
    }
    for k in RECALL_K:
        numerator = sum(row[f"recall_at_{k}"] is True for row in recall_rows)
        result[f"recall_at_{k}_exact_object_id"] = {
            "value": _rate(numerator, len(recall_rows)),
            "numerator": numerator,
            "denominator": len(recall_rows),
            "status": "measured" if recall_rows else "N/A",
        }
    for threshold in THRESHOLDS:
        numerator = sum(iou >= threshold for iou in ious)
        suffix = str(threshold).replace(".", "_")
        record = {
            "value": _rate(numerator, len(ious)),
            "numerator": numerator,
            "denominator": len(ious),
            "threshold": threshold,
            "status": "measured" if ious else "N/A",
        }
        result[f"acc_at_{suffix}"] = record
        result[f"bbox_acc_at_{suffix}"] = record
    return result


def _strata(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unspecified")].append(row)
    return {name: aggregate_rows(group) for name, group in sorted(groups.items())}


def _construction_cost(rows: list[dict[str, Any]], run_config: dict[str, Any]) -> dict[str, Any]:
    records = [row["construction_cost"] for row in rows if row.get("construction_cost") is not None]
    run_cost = run_config.get("construction_cost")
    return {
        "status": "reported" if records or run_cost is not None else "N/A",
        "scope": "one-time construction; excluded from per-query efficiency",
        "run": run_cost,
        "per_query_records": records,
        "record_count": len(records),
    }


def evaluate(
    benchmark_path: Path,
    results_dir: Path,
    *,
    run_config_path: Path | None = None,
) -> dict[str, Any]:
    benchmark = _load_json(benchmark_path)
    queries = benchmark.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Benchmark must contain a queries list")
    run_config = (
        _load_json(run_config_path)
        if run_config_path is not None and run_config_path.exists()
        else {}
    )
    files: dict[str, tuple[Path, dict[str, Any]]] = {}
    read_warnings: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            raw = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            read_warnings.append(f"{path.name}: {exc}")
            continue
        query_id = str(raw.get("query_id") or path.stem)
        if query_id in files:
            read_warnings.append(f"duplicate {query_id}: ignored {path.name}")
        else:
            files[query_id] = (path, raw)

    rows: list[dict[str, Any]] = []
    for query in queries:
        if not isinstance(query, dict) or query.get("query_id") is None:
            continue
        query_id = str(query["query_id"])
        saved = files.get(query_id)
        rows.append(
            evaluate_query(
                query,
                saved[1] if saved else None,
                result_file=saved[0].name if saved else None,
            )
        )
    failure_counts = Counter(failure for row in rows for failure in row["failures"])
    segmentation = segmentation_metrics(rows)
    public_rows = [{key: value for key, value in row.items() if key != "_segmentation"} for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark.get("benchmark_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "benchmark_queries": len(rows),
            "result_files": sum(row["input_shape"] != "missing_result" for row in rows),
            "canonical_results": sum(row["input_shape"] == "canonical" for row in rows),
            "legacy_adapted_results": sum(row["input_shape"] == "legacy_adapted" for row in rows),
            "schema_valid_results": sum(row["schema_valid"] for row in rows),
        },
        "metrics": {**aggregate_rows(rows), "segmentation": segmentation},
        "construction_cost": _construction_cost(rows, run_config),
        "strata": {
            "scene": _strata(rows, "scene_id"),
            "source": _strata(rows, "source"),
            "query_type": _strata(rows, "query_type"),
            "search_variant": _strata(rows, "search_variant"),
        },
        "failure_taxonomy": {
            category: failure_counts.get(category, 0) for category in FAILURE_TYPES
        },
        "per_query": public_rows,
        "failures": [row for row in public_rows if row["failures"] or row["schema_errors"]],
        "warnings": read_warnings,
    }


def _fmt(value: Any) -> str:
    return "N/A" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def summary_markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# BBQ-aligned grounding summary",
        "",
        f"- Benchmark: `{summary.get('benchmark_id')}`",
        f"- Queries: {summary['coverage']['benchmark_queries']}",
        f"- Canonical / legacy-adapted results: {summary['coverage']['canonical_results']} / {summary['coverage']['legacy_adapted_results']}",
        "",
        "Missing, unavailable, and invalid predictions remain in GT-eligible 3D-box denominators as IoU 0.",
        "",
        "| Metric | Value | Numerator / denominator |",
        "|---|---:|---:|",
        f"| 3D bbox IoU | {_fmt(metrics['bbox_iou_3d']['value'])} | {metrics['bbox_iou_3d']['denominator']} eligible |",
    ]
    for threshold in THRESHOLDS:
        record = metrics[f"acc_at_{str(threshold).replace('.', '_')}"]
        lines.append(
            f"| Acc@{threshold:g} | {_fmt(record['value'])} | {record['numerator']} / {record['denominator']} |"
        )
    for k in RECALL_K:
        recall = metrics[f"recall_at_{k}_exact_object_id"]
        lines.append(
            f"| Recall@{k} exact object ID | {_fmt(recall['value'])} | "
            f"{recall['numerator']} / {recall['denominator']} |"
        )
    mrr = metrics["mrr_exact_object_id"]
    lines.append(
        f"| MRR exact object ID | {_fmt(mrr['value'])} | {mrr['denominator']} eligible |"
    )
    negative = metrics["negative_accuracy"]
    lines.append(
        f"| Verified-negative accuracy | {_fmt(negative['value'])} | "
        f"{negative['numerator']} / {negative['denominator']} |"
    )
    lines.extend(["", "## Segmentation", "", "| Metric | Status | Value |", "|---|---|---:|"])
    for name, record in metrics["segmentation"].items():
        lines.append(f"| {name} | {record['status']} | {_fmt(record['value'])} |")
    lines.extend(["", "## Failure taxonomy", "", "| Failure | Count |", "|---|---:|"])
    for category in FAILURE_TYPES:
        lines.append(f"| {category} | {summary['failure_taxonomy'][category]} |")
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| Measure | Mean | Total | Records |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, record in metrics["efficiency"].items():
        lines.append(
            f"| {name} | {_fmt(record['mean'])} | {_fmt(record['total'])} | {record['denominator']} |"
        )
    lines.extend(
        [
            "",
            "Construction cost is reported separately because it is a one-time indexing/map cost.",
            "",
            f"- Construction cost status: `{summary['construction_cost']['status']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "query_id",
        "scene_id",
        "source",
        "query_type",
        "search_variant",
        "input_shape",
        "schema_valid",
        "prediction_available",
        "bbox_eligible",
        "bbox_iou_3d",
        "recall_at_1_eligible",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "reciprocal_rank",
        "expected_rank",
        "ranked_object_ids",
        "negative_eligible",
        "negative_correct",
        "predicted_object_id",
        "failures",
        "runtime_seconds",
        "checked_nodes",
        "checked_views",
        "checked_objects",
        "context_chars",
        "estimated_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["per_query"]:
            flat = {key: row.get(key) for key in fields}
            flat["failures"] = "; ".join(row["failures"])
            flat.update(row["efficiency"])
            writer.writerow(flat)


def write_outputs(summary: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "grounding_summary.json",
        "csv": out_dir / "grounding_summary.csv",
        "markdown": out_dir / "grounding_summary.md",
        "failures": out_dir / "grounding_failures.json",
    }
    paths["json"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(paths["csv"], summary)
    paths["markdown"].write_text(summary_markdown(summary), encoding="utf-8")
    paths["failures"].write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "failure_taxonomy": summary["failure_taxonomy"],
                "per_query_failures": summary["failures"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BBQ-aligned 3D grounding results")
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-config", type=Path)
    args = parser.parse_args()
    if not args.results.is_dir():
        parser.error(f"results directory does not exist: {args.results}")
    summary = evaluate(
        args.benchmark.resolve(),
        args.results.resolve(),
        run_config_path=args.run_config.resolve() if args.run_config else None,
    )
    for path in write_outputs(summary, args.out.resolve()).values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
