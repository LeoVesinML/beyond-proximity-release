"""Strict helpers for converting real baseline exports to canonical results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


QUERY_RESULT_SCHEMA = "semanticsplat.query_result.v1"
RUN_CONFIG_SCHEMA = "semanticsplat.run_config.v1"
VALID_MODES = {"live", "cached_live"}


class BaselineAdapterError(ValueError):
    """Raised when native baseline evidence is incomplete or inconsistent."""


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineAdapterError(f"Missing native input: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineAdapterError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BaselineAdapterError(f"Expected JSON object: {path}")
    return data


def require_nonempty_string(data: dict[str, Any], field: str, context: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BaselineAdapterError(f"{context} requires non-empty {field}")
    return value.strip()


def validate_native_export(
    native_path: Path,
    native: dict[str, Any],
    *,
    baseline: str,
    scene_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    provenance = native.get("provenance")
    predictions = native.get("predictions")
    if not isinstance(provenance, dict):
        raise BaselineAdapterError("Native export requires provenance object")
    if provenance.get("native_execution") is not True:
        raise BaselineAdapterError("provenance.native_execution must be true")
    if str(provenance.get("baseline", "")).strip().lower() != baseline:
        raise BaselineAdapterError(f"provenance.baseline must be {baseline}")
    if str(provenance.get("scene_id", "")).strip() != scene_id:
        raise BaselineAdapterError("provenance.scene_id does not match --scene-id")
    for field in ("repository", "revision", "command"):
        require_nonempty_string(provenance, field, "provenance")

    checkpoints = provenance.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints or not all(
        isinstance(value, str) and value.strip() for value in checkpoints
    ):
        raise BaselineAdapterError("provenance.checkpoints requires at least one real checkpoint identifier")
    artifacts = provenance.get("native_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BaselineAdapterError("provenance.native_artifacts requires at least one source artifact")
    for value in artifacts:
        if not isinstance(value, str) or not value.strip():
            raise BaselineAdapterError("provenance.native_artifacts entries must be paths")
        artifact = Path(value)
        if not artifact.is_absolute():
            artifact = native_path.parent / artifact
        if not artifact.is_file():
            raise BaselineAdapterError(f"Native artifact does not exist: {artifact}")

    if not isinstance(predictions, list) or not predictions:
        raise BaselineAdapterError("Native export requires at least one prediction")
    if not all(isinstance(item, dict) for item in predictions):
        raise BaselineAdapterError("Every native prediction must be an object")
    return provenance, predictions


def canonical_metrics(record: dict[str, Any], visited: list[str], selected_views: list[str]) -> dict[str, Any]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise BaselineAdapterError("Every native prediction requires metrics")
    runtime = metrics.get("runtime_seconds")
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or runtime < 0:
        raise BaselineAdapterError("metrics.runtime_seconds must be a non-negative number")
    model_calls = metrics.get("model_call_count")
    if isinstance(model_calls, bool) or not isinstance(model_calls, int) or model_calls < 0:
        raise BaselineAdapterError("metrics.model_call_count must be a non-negative integer")
    cache_hits = metrics.get("cache_hits", 0)
    if isinstance(cache_hits, bool) or not isinstance(cache_hits, int) or cache_hits < 0:
        raise BaselineAdapterError("metrics.cache_hits must be a non-negative integer")
    return {
        "runtime_seconds": float(runtime),
        "stage_runtime_seconds": metrics.get("stage_runtime_seconds", {"baseline_query": float(runtime)}),
        "token_usage": metrics.get("token_usage"),
        "model_call_count": model_calls,
        "cache_hits": cache_hits,
        "retry_count": int(metrics.get("retry_count", 0)),
        "failure_count": int(metrics.get("failure_count", 0)),
        "visited_node_count": len(visited),
        "checked_view_count": int(metrics.get("checked_view_count", len(selected_views))),
        "construction_cost": metrics.get("construction_cost"),
    }


def write_adapter_run(
    *,
    native_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    run_id: str,
    method: str,
    mode: str,
    scene_id: str,
    benchmark_scene_id: str,
    adapt_prediction: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> Path:
    if mode not in VALID_MODES:
        raise BaselineAdapterError("Baseline mode must be live or cached_live")
    native = load_object(native_path)
    provenance, predictions = validate_native_export(
        native_path, native, baseline=method, scene_id=scene_id
    )
    benchmark = load_object(benchmark_path)
    queries = benchmark.get("queries")
    if not isinstance(queries, list):
        raise BaselineAdapterError("Benchmark requires queries list")
    query_by_id = {
        str(query["query_id"]): query
        for query in queries
        if isinstance(query, dict)
        and query.get("query_id")
        and str(query.get("scene_id")) == benchmark_scene_id
    }

    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for prediction in predictions:
        query_id = require_nonempty_string(prediction, "query_id", "prediction")
        if query_id in seen:
            raise BaselineAdapterError(f"Duplicate prediction query_id: {query_id}")
        seen.add(query_id)
        query = query_by_id.get(query_id)
        if query is None:
            raise BaselineAdapterError(
                f"Prediction {query_id} is absent from benchmark scene {benchmark_scene_id}"
            )
        result = adapt_prediction(prediction, query)
        result.update({
            "schema_version": QUERY_RESULT_SCHEMA,
            "run_id": run_id,
            "method": method,
            "mode": mode,
            "availability_status": "available",
            "availability_reason": None,
            "scene_id": scene_id,
            "query_id": query_id,
            "query": query.get("query"),
            "query_type": query.get("query_type"),
        })
        canonical.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    query_dir = output_dir / "query_results"
    query_dir.mkdir(exist_ok=True)
    for result in canonical:
        (query_dir / f"{result['query_id']}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    now = datetime.now(timezone.utc).isoformat()
    run_config = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "method": method,
        "mode": mode,
        "status": "complete",
        "scene_ids": [scene_id],
        "scenes": [{
            "scene_id": scene_id,
            "benchmark_scene_id": benchmark_scene_id,
        }],
        "benchmark_path": str(benchmark_path),
        "native_input": str(native_path),
        "provenance": provenance,
        "result_count": len(canonical),
        "finished_at": now,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logs = {
        "run_id": run_id,
        "mode": mode,
        "finished_at": now,
        "events": [{"stage": "native_output_adaptation", "status": "complete", "count": len(canonical)}],
        "warnings": ["Adapter output is valid only with the preserved native artifacts and provenance."],
    }
    (output_dir / "logs.json").write_text(
        json.dumps(logs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "baseline_summary.md").write_text(
        f"# {method} Baseline Run\n\n"
        f"- Mode: `{mode}`\n"
        f"- Scene: `{scene_id}`\n"
        f"- Canonical results: {len(canonical)}\n"
        f"- Native input: `{native_path}`\n",
        encoding="utf-8",
    )
    return output_dir
