#!/usr/bin/env python3
"""Run a reproducible SemanticSplat experiment without Cursor or UI."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.model_client import (  # noqa: E402
    ModelClient,
    ModelClientError,
    create_model_client,
)
from backend.geometry.unprojector import bbox_2d_to_3d  # noqa: E402
from scripts.evaluate_results import (  # noqa: E402
    evaluate,
    failure_markdown,
    summary_markdown,
    write_json,
)


QUERY_RESULT_SCHEMA = "semanticsplat.query_result.v1"
RUN_CONFIG_SCHEMA = "semanticsplat.run_config.v1"
SMALL_BBOX_LABEL_TERMS = {
    "sign", "clock", "lamp", "vase", "marker", "label", "camera", "monitor",
    "screen", "tv", "panel", "graphic", "photograph", "picture", "artwork",
    "stand", "bin", "trash", "railing", "handrail",
}
LARGE_BBOX_LABEL_TERMS = {
    "wall", "facade", "curtain", "floor", "ground", "pavement", "water",
    "skyline", "shoreline", "island", "building", "stairs", "staircase",
    "corridor", "walkway", "door", "doors", "window", "windows", "seats",
    "tables", "chairs", "sofas",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(
                f"{path} must use JSON-compatible YAML unless PyYAML is installed: {json_error}"
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be an object: {path}")
    return data


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def configured_scene_entries(
    config: dict[str, Any],
    scene_root: Path,
    benchmark_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    configured = config.get("scenes")
    if configured is None:
        scene_ids = [str(item) for item in config.get("scene_ids", [])]
        if not scene_ids:
            scene_ids = sorted({str(query.get("scene_id")) for query in benchmark_queries if query.get("scene_id")})
        return [
            {
                "scene_id": scene_id,
                "benchmark_scene_id": scene_id,
                "path": scene_root / scene_id,
                "semantic_eval_allowed": None,
                "geometry_eval_allowed": None,
                "validation_report": None,
            }
            for scene_id in scene_ids
        ]

    if not isinstance(configured, list) or not configured:
        raise ValueError("Config scenes must be a non-empty list")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in configured:
        if not isinstance(item, dict):
            raise ValueError("Each config scenes entry must be an object")
        scene_id = str(item.get("scene_id") or "").strip()
        if not scene_id:
            raise ValueError("Each config scenes entry requires scene_id")
        if scene_id in seen:
            raise ValueError(f"Duplicate scene_id in config scenes: {scene_id}")
        seen.add(scene_id)

        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Config scene {scene_id} requires path")
        scene_path = resolve_path(raw_path)
        if not scene_path.is_dir():
            raise FileNotFoundError(f"Configured scene path not found for {scene_id}: {scene_path}")

        raw_report = item.get("validation_report")
        if not isinstance(raw_report, str) or not raw_report.strip():
            raise ValueError(f"Config scene {scene_id} requires validation_report")
        validation_report = resolve_path(raw_report)
        if not validation_report.is_file():
            raise FileNotFoundError(f"Validation report not found for {scene_id}: {validation_report}")
        validation = load_json(validation_report)
        validation_flags = validation.get("semantic_validation")
        if not isinstance(validation_flags, dict):
            validation_flags = validation
        if str(validation.get("scene_id") or validation_flags.get("scene_id")) != scene_id:
            raise ValueError(f"Validation report scene_id mismatch for {scene_id}: {validation_report}")

        semantic_allowed = item.get("semantic_eval_allowed")
        geometry_allowed = item.get("geometry_eval_allowed")
        if not isinstance(semantic_allowed, bool) or not isinstance(geometry_allowed, bool):
            raise ValueError(f"Config scene {scene_id} requires boolean evaluation flags")
        reported_semantic = validation_flags.get(
            "semantic_eval_allowed", validation_flags.get("semantic_evaluation_allowed")
        )
        reported_geometry = validation_flags.get(
            "geometry_eval_allowed", validation_flags.get("three_d_localization_allowed")
        )
        if reported_semantic is not semantic_allowed:
            raise ValueError(f"Config semantic_eval_allowed disagrees with validation report for {scene_id}")
        if reported_geometry is not geometry_allowed:
            raise ValueError(f"Config geometry_eval_allowed disagrees with validation report for {scene_id}")

        benchmark_scene_id = str(item.get("benchmark_scene_id") or scene_id).strip()
        if not benchmark_scene_id:
            raise ValueError(f"Config scene {scene_id} has an empty benchmark_scene_id")

        entries.append({
            "scene_id": scene_id,
            "benchmark_scene_id": benchmark_scene_id,
            "path": scene_path,
            "semantic_eval_allowed": semantic_allowed,
            "geometry_eval_allowed": geometry_allowed,
            "validation_report": validation_report,
        })

    legacy_ids = [str(item) for item in config.get("scene_ids", [])]
    entry_ids = [entry["scene_id"] for entry in entries]
    if legacy_ids and legacy_ids != entry_ids:
        raise ValueError("Config scene_ids must match scenes entries in the same order")
    benchmark_ids = [entry["benchmark_scene_id"] for entry in entries]
    if len(set(benchmark_ids)) != len(benchmark_ids):
        raise ValueError("Config benchmark_scene_id values must be unique")
    return entries


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def load_scene(scene_dir: Path) -> dict[str, Any]:
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    transforms_path = scene_dir / "transforms.json"
    transforms = load_json(transforms_path) if transforms_path.exists() else {"frames": []}

    frame_by_view: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(transforms.get("frames", []), 1):
        if not isinstance(frame, dict):
            continue
        view_id = str(frame.get("view_id") or f"v{index:03d}")
        frame_by_view[view_id] = frame

    views: dict[str, dict[str, Any]] = {}
    for path in sorted((scene_dir / "views").glob("*.json")):
        data = load_json(path)
        view_id = str(data.get("view_id") or path.stem)
        views[view_id] = data

    tree: dict[str, dict[str, Any]] = {}
    for path in sorted((scene_dir / "tree").glob("node_*.json")):
        data = load_json(path)
        node_id = str(data.get("node_id") or path.stem.removeprefix("node_"))
        tree[node_id] = data

    manifest_path = scene_dir / "tree" / "manifest.json"
    tree_manifest = load_json(manifest_path) if manifest_path.is_file() else None
    captured_view_count = sum(
        data.get("schema_version") == "semanticsplat.captured_view_json.v1"
        for data in views.values()
    )
    semantic_index_format = (
        "captured_viewjson_v1" if views and captured_view_count == len(views) else "legacy_or_unknown"
    )

    return {
        "scene_dir": scene_dir,
        "transforms": transforms,
        "frame_by_view": frame_by_view,
        "existing_views": views,
        "existing_tree": tree,
        "tree_manifest": tree_manifest,
        "semantic_index_format": semantic_index_format,
        "semantic_index_complete": bool(
            tree_manifest.get("complete") if isinstance(tree_manifest, dict) else tree
        ),
    }


def validate_depth(scene_dir: Path, sample_limit: int = 3) -> dict[str, Any]:
    depth_files = sorted((scene_dir / "depths").glob("*.npy"))
    if not depth_files:
        return {
            "status": "missing",
            "checked_frame_count": 0,
            "constant_frame_count": 0,
            "reason": "No .npy depth files are available.",
        }
    try:
        import numpy as np
    except ImportError:
        return {
            "status": "unchecked",
            "checked_frame_count": 0,
            "constant_frame_count": 0,
            "reason": "NumPy is unavailable for depth validation.",
        }

    checked = 0
    constant = 0
    invalid = 0
    for path in depth_files[: max(1, sample_limit)]:
        array = np.load(path)
        checked += 1
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            invalid += 1
            continue
        if float(np.nanmax(finite)) == float(np.nanmin(finite)):
            constant += 1
    reliable = checked > 0 and constant == 0 and invalid == 0
    reason = "Sampled depth maps are finite and non-constant." if reliable else (
        f"Sampled {checked} maps: {constant} constant and {invalid} without finite values."
    )
    return {
        "status": "reliable" if reliable else "unreliable",
        "checked_frame_count": checked,
        "constant_frame_count": constant,
        "invalid_frame_count": invalid,
        "reason": reason,
    }


def image_path_for(scene: dict[str, Any], view_id: str) -> Path:
    frame = scene["frame_by_view"].get(view_id, {})
    file_path = frame.get("file_path") if isinstance(frame, dict) else None
    if file_path:
        return scene["scene_dir"] / str(file_path)
    return scene["scene_dir"] / "images" / f"{view_id}.png"


def numeric_bbox_2d(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if any(item < 0.0 or item > 1.0 for item in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def clamp_bbox_value(value: float) -> float:
    return min(1.0, max(0.0, value))


def normalized_bbox_2d(values: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = values
    x1 = clamp_bbox_value(x1)
    y1 = clamp_bbox_value(y1)
    x2 = clamp_bbox_value(x2)
    y2 = clamp_bbox_value(y2)
    if x2 - x1 < 0.04:
        mid = (x1 + x2) / 2
        x1 = clamp_bbox_value(mid - 0.02)
        x2 = clamp_bbox_value(mid + 0.02)
    if y2 - y1 < 0.04:
        mid = (y1 + y2) / 2
        y1 = clamp_bbox_value(mid - 0.02)
        y2 = clamp_bbox_value(mid + 0.02)
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def shrink_bbox_2d(box: list[float], scale_x: float, scale_y: float) -> list[float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = (x2 - x1) * scale_x
    height = (y2 - y1) * scale_y
    return normalized_bbox_2d((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))


def approximate_bbox_2d_from_object(obj: dict[str, Any]) -> list[float] | None:
    label = str(obj.get("label") or "")
    approx_location = str(obj.get("approx_location") or "")
    attributes = obj.get("attributes")
    if isinstance(attributes, list):
        attribute_text = " ".join(str(item) for item in attributes if item is not None)
    elif isinstance(attributes, dict):
        attribute_text = " ".join(str(item) for item in attributes.values() if item is not None)
    else:
        attribute_text = ""
    text = " ".join([label, approx_location, attribute_text]).strip().lower()
    if not text:
        return None

    label_terms = set(re.findall(r"[a-z0-9]+", label.lower()))
    if "left" in text and "right" in text:
        x1, x2 = 0.05, 0.95
    elif "center-left" in text or "centre-left" in text:
        x1, x2 = 0.18, 0.58
    elif "center-right" in text or "centre-right" in text:
        x1, x2 = 0.42, 0.82
    elif "left" in text:
        x1, x2 = 0.04, 0.48
    elif "right" in text:
        x1, x2 = 0.52, 0.96
    elif "center" in text or "centre" in text or "central" in text or "middle" in text:
        x1, x2 = 0.27, 0.73
    else:
        x1, x2 = 0.18, 0.82

    if any(word in text for word in ("ceiling", "upper", "above", "top")):
        y1, y2 = 0.02, 0.38
    elif any(word in text for word in ("floor", "ground", "pavement", "water", "roadway", "walkway")):
        y1, y2 = 0.48, 0.98
    elif any(word in text for word in ("foreground", "near", "low", "bottom")):
        y1, y2 = 0.42, 0.96
    elif any(word in text for word in ("background", "distant", "far")):
        y1, y2 = 0.06, 0.62
    elif any(word in text for word in ("wall", "facade", "panel", "curtain", "window", "door")):
        y1, y2 = 0.08, 0.90
    else:
        y1, y2 = 0.18, 0.86

    box = normalized_bbox_2d((x1, y1, x2, y2))
    if label_terms & SMALL_BBOX_LABEL_TERMS:
        box = shrink_bbox_2d(box, 0.48, 0.44)
    if label_terms & LARGE_BBOX_LABEL_TERMS:
        box = normalized_bbox_2d((box[0] - 0.05, box[1] - 0.04, box[2] + 0.05, box[3] + 0.04))
    return box


def frame_depth_path(scene: dict[str, Any], view: dict[str, Any], view_id: str) -> Path | None:
    frame = scene["frame_by_view"].get(view_id, {})
    depth_value = frame.get("depth_file_path") if isinstance(frame, dict) else None
    if depth_value is None:
        depth_value = view.get("depth_path")
    if not isinstance(depth_value, str) or not depth_value.strip():
        return None
    return scene["scene_dir"] / depth_value


def view_object_list(view: dict[str, Any]) -> list[dict[str, Any]]:
    value = view.get("visible_objects")
    if not isinstance(value, list):
        value = view.get("objects")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def enrich_view_bboxes(scene: dict[str, Any], views: dict[str, dict[str, Any]]) -> dict[str, Any]:
    transforms = scene.get("transforms", {})
    required = ("fl_x", "fl_y", "cx", "cy", "w", "h")
    if not all(isinstance(transforms.get(key), (int, float)) for key in required):
        return {
            "status": "skipped_missing_intrinsics",
            "objects_seen": sum(len(view_object_list(view)) for view in views.values()),
            "bbox_2d_inferred": 0,
            "bbox_3d_generated": 0,
            "bbox_3d_existing": 0,
        }

    try:
        import numpy as np
    except ImportError:
        return {
            "status": "skipped_missing_numpy",
            "objects_seen": sum(len(view_object_list(view)) for view in views.values()),
            "bbox_2d_inferred": 0,
            "bbox_3d_generated": 0,
            "bbox_3d_existing": 0,
        }

    stats = {
        "status": "complete",
        "objects_seen": 0,
        "bbox_2d_inferred": 0,
        "bbox_3d_generated": 0,
        "bbox_3d_existing": 0,
        "bbox_3d_failed": 0,
    }
    depth_cache: dict[str, Any] = {}

    for view_id, view in views.items():
        frame = scene["frame_by_view"].get(view_id, {})
        transform_matrix = frame.get("transform_matrix") if isinstance(frame, dict) else None
        if not isinstance(transform_matrix, list):
            continue
        depth_path = frame_depth_path(scene, view, view_id)
        for obj in view_object_list(view):
            stats["objects_seen"] += 1
            if isinstance(obj.get("bbox_3d"), dict):
                stats["bbox_3d_existing"] += 1
                continue

            bbox_2d = numeric_bbox_2d(obj.get("bbox_2d"))
            if bbox_2d is None:
                bbox_2d = approximate_bbox_2d_from_object(obj)
                if bbox_2d is not None:
                    obj["bbox_2d"] = bbox_2d
                    obj["bbox_2d_source"] = "approx_location_fallback_v1"
                    stats["bbox_2d_inferred"] += 1
            if bbox_2d is None or depth_path is None or not depth_path.is_file():
                stats["bbox_3d_failed"] += 1
                continue

            depth_key = str(depth_path)
            try:
                if depth_key not in depth_cache:
                    depth_cache[depth_key] = np.load(depth_path).astype(np.float32)
                bbox_3d = bbox_2d_to_3d(
                    bbox_norm=tuple(bbox_2d),
                    depth_map=depth_cache[depth_key],
                    fl_x=float(transforms["fl_x"]),
                    fl_y=float(transforms["fl_y"]),
                    cx=float(transforms["cx"]),
                    cy=float(transforms["cy"]),
                    transform_matrix=transform_matrix,
                    width=int(transforms["w"]),
                    height=int(transforms["h"]),
                )
            except (OSError, ValueError, TypeError):
                bbox_3d = None
            if bbox_3d is None:
                stats["bbox_3d_failed"] += 1
                continue
            bbox_3d.label = str(obj.get("label") or "")
            obj["bbox_3d"] = bbox_3d.model_dump()
            obj["bbox_3d_source"] = (
                "bbox_2d_depth_unprojection_v1"
                if obj.get("bbox_2d_source") != "approx_location_fallback_v1"
                else "approx_location_depth_unprojection_v1"
            )
            stats["bbox_3d_generated"] += 1

    return stats


def select_views(scene: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    strategy = str(settings.get("strategy", "existing"))
    if strategy != "existing":
        raise ValueError(f"Unsupported view selection strategy in Phase 2: {strategy}")
    available = sorted(set(scene["existing_views"]) | set(scene["frame_by_view"]))
    max_views = settings.get("max_views")
    if isinstance(max_views, int) and max_views > 0:
        available = available[:max_views]
    return available


def stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    counters = ("model_call_count", "cache_hits", "retry_count", "failure_count")
    result = {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in counters}
    token_usage: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        previous = before.get(key)
        current = after.get(key)
        if isinstance(current, int):
            token_usage[key] = current - (previous if isinstance(previous, int) else 0)
    result["token_usage"] = token_usage or None
    return result


def compact_tree_for_context(tree: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": str(node.get("node_id") or key),
            "node_type": node.get("node_type") or node.get("type"),
            "name": node.get("name"),
            "summary": node.get("summary"),
            "view_ids": [str(item) for item in node.get("view_ids", []) if item is not None]
            if isinstance(node.get("view_ids"), list)
            else [],
            "children_ids": [str(item) for item in node.get("children_ids", []) if item is not None]
            if isinstance(node.get("children_ids"), list)
            else [],
        }
        for key, node in sorted(tree.items())
        if isinstance(node, dict)
    ]


def compact_views_for_context(views: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields = (
        "summary",
        "scene_summary",
        "room_type",
        "visible_regions",
        "visible_objects",
        "objects",
        "landmarks",
        "free_text_notes",
    )
    return {
        view_id: {field: view[field] for field in fields if field in view}
        for view_id, view in sorted(views.items())
        if isinstance(view, dict)
    }


def estimate_query_context_metrics(
    query: dict[str, Any],
    tree: dict[str, dict[str, Any]],
    views: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    context = {
        "query": query,
        "tree": compact_tree_for_context(tree),
        "views": compact_views_for_context(views),
    }
    serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    prompt = (
        "Answer the scene query using only the supplied semantic index. "
        "Return one JSON object with structured_plan, visited_nodes, selected_views, "
        "checked_view_ids, result, and warnings.\n\n"
        + serialized
    )
    prompt_size_chars = len(prompt)
    return {
        "context_size_chars": len(serialized),
        "prompt_size_chars": prompt_size_chars,
        "estimated_input_tokens": max(1, (prompt_size_chars + 3) // 4),
        "estimated_token_method": "ceil(prompt_size_chars / 4)",
    }


def log_event(logs: dict[str, Any], stage: str, status: str, **details: Any) -> None:
    logs["events"].append({"at": utc_now(), "stage": stage, "status": status, **details})


def timed_stage(
    logs: dict[str, Any],
    stage: str,
    function: Callable[[], Any],
) -> tuple[Any, float]:
    started = time.perf_counter()
    try:
        value = function()
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 6)
        log_event(logs, stage, "error", runtime_seconds=elapsed, error=str(exc))
        raise
    elapsed = round(time.perf_counter() - started, 6)
    log_event(logs, stage, "complete", runtime_seconds=elapsed)
    return value, elapsed


def string_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def bbox_3d_quality(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, "bbox_3d_missing"
    if not isinstance(value, dict):
        return None, "bbox_3d_not_an_object"
    center = value.get("center")
    dimensions = value.get("size", value.get("dimensions"))
    if not isinstance(center, (list, tuple)) or len(center) != 3:
        return None, "bbox_3d_center_invalid"
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3:
        return None, "bbox_3d_dimensions_invalid"
    try:
        center_values = [float(item) for item in center]
        dimension_values = [float(item) for item in dimensions]
    except (TypeError, ValueError):
        return None, "bbox_3d_values_non_numeric"
    if not all(math.isfinite(item) for item in [*center_values, *dimension_values]):
        return None, "bbox_3d_values_non_finite"
    if any(item <= 0.0 for item in dimension_values):
        return None, "bbox_3d_extents_non_positive"
    normalized = dict(value)
    normalized["center"] = center_values
    if "size" in value:
        normalized["size"] = dimension_values
    else:
        normalized["dimensions"] = dimension_values
    return normalized, None


def selected_object_id(result: dict[str, Any]) -> str | None:
    value = result.get("selected_object_id")
    if value is not None:
        text = str(value).strip()
        if text:
            return text
    for key in ("bbox_3d", "bbox_2d"):
        bbox = result.get(key)
        if isinstance(bbox, dict):
            for id_key in ("object_id", "instance_id", "id"):
                value = bbox.get(id_key)
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text
    return None


def canonical_result(
    *,
    run_id: str,
    method: str,
    mode: str,
    query: dict[str, Any],
    answer: dict[str, Any],
    runtime_seconds: float,
    usage: dict[str, Any],
    context_metrics: dict[str, Any],
    depth_validation: dict[str, Any],
    geometry_eval_allowed: bool | None = None,
) -> dict[str, Any]:
    warnings = [str(item) for item in answer.get("warnings", [])]
    if depth_validation.get("status") != "reliable":
        warnings.append(f"Unreliable depth: {depth_validation.get('reason', 'validation failed')}")
    visited_nodes = string_id_list(answer.get("visited_nodes"))
    traversal_path = string_id_list(answer.get("traversal_path")) or visited_nodes
    selected_views = string_id_list(answer.get("selected_views"))
    checked_node_ids = string_id_list(answer.get("checked_node_ids")) or visited_nodes
    checked_view_ids = string_id_list(answer.get("checked_view_ids")) or selected_views
    checked_object_ids = string_id_list(answer.get("checked_object_ids"))
    raw_checked_object_count = answer.get("checked_object_count")
    checked_object_count = (
        raw_checked_object_count
        if isinstance(raw_checked_object_count, int) and not isinstance(raw_checked_object_count, bool)
        and raw_checked_object_count >= len(checked_object_ids)
        else len(checked_object_ids)
    )
    result = answer.get("result") if isinstance(answer.get("result"), dict) else {}
    found = bool(result.get("found", False))
    negative_query = (
        query.get("query_type") == "negative"
        or query.get("expected_output_type") == "not_found"
    )
    geometry_query_eligible = (
        geometry_eval_allowed is True
        and (
            query.get("geometry_eval_allowed") is True
            or query.get("expected_output_type") in {"object", "object_3d_bbox"}
            or isinstance(query.get("expected_bbox_3d"), dict)
            or bool(query.get("expected_bboxes_3d"))
        )
    )
    raw_bbox_3d = result.get("bbox_3d")
    dataset_gt_bbox = bool(
        isinstance(raw_bbox_3d, dict)
        and raw_bbox_3d.get("object_id") is not None
        and str(raw_bbox_3d.get("coordinate_frame", "")).startswith(("replica_", "scannet_"))
    )
    bbox_3d, bbox_quality_reason = bbox_3d_quality(raw_bbox_3d)
    if bbox_3d is not None and depth_validation.get("status") != "reliable" and not dataset_gt_bbox:
        bbox_3d = None
        bbox_quality_reason = "depth_validation_not_reliable"
    if negative_query:
        bbox_3d = None
        bbox_availability_status = "not_applicable"
        bbox_unavailable_reason = "negative_query"
    elif not found:
        bbox_3d = None
        bbox_availability_status = "not_applicable"
        bbox_unavailable_reason = "object_not_found"
    elif not geometry_query_eligible:
        bbox_3d = None
        bbox_availability_status = "not_eligible"
        bbox_unavailable_reason = "geometry_evaluation_not_eligible"
    elif bbox_3d is None:
        bbox_availability_status = "unavailable"
        bbox_unavailable_reason = bbox_quality_reason or "bbox_3d_unavailable"
    else:
        bbox_availability_status = "available"
        bbox_unavailable_reason = None
    result_unavailable = bool(
        geometry_query_eligible
        and not negative_query
        and (not found or bbox_3d is None)
    )
    canonical_payload = {
        "found": found,
        "matched_object": result.get("matched_object"),
        "selected_object_id": selected_object_id(result),
        "ranked_object_ids": string_id_list(result.get("ranked_object_ids")),
        "ranked_candidates": (
            result.get("ranked_candidates")
            if isinstance(result.get("ranked_candidates"), list)
            else []
        ),
        "selected_node_id": result.get("selected_node_id"),
        "selected_view_id": result.get("selected_view_id"),
        "bbox_2d": result.get("bbox_2d"),
        "bbox_3d": bbox_3d,
        "bbox_availability_status": bbox_availability_status,
        "bbox_unavailable_reason": bbox_unavailable_reason,
        "camera_pose": result.get("camera_pose"),
        "confidence": float(result.get("confidence", 0.0)),
        "explanation": str(result.get("explanation", "")),
        "relation_satisfied": result.get("relation_satisfied"),
        "relation_audit": (
            result.get("relation_audit")
            if isinstance(result.get("relation_audit"), dict)
            else {}
        ),
        "search_score": result.get("search_score"),
        "score_details": result.get("score_details", {}),
    }
    return {
        "schema_version": QUERY_RESULT_SCHEMA,
        "run_id": run_id,
        "method": method,
        "mode": mode,
        "availability_status": "unavailable" if result_unavailable else "available",
        "availability_reason": bbox_unavailable_reason if result_unavailable else None,
        "scene_id": query["scene_id"],
        "query_id": query["query_id"],
        "query": query["query"],
        "query_type": query["query_type"],
        "structured_plan": answer.get("structured_plan", {}),
        "visited_nodes": visited_nodes,
        "traversal_path": traversal_path,
        "selected_views": selected_views,
        "checked_node_ids": checked_node_ids,
        "checked_view_ids": checked_view_ids,
        "checked_object_ids": checked_object_ids,
        "search_variant": str(answer.get("search_variant") or "unspecified"),
        "fallback_used": bool(answer.get("fallback_used", False)),
        "fallback_reason": answer.get("fallback_reason"),
        "result": canonical_payload,
        "metrics_log": {
            "runtime_seconds": runtime_seconds,
            "stage_runtime_seconds": {"answer_query": runtime_seconds},
            "token_usage": usage.get("token_usage"),
            "model_call_count": usage.get("model_call_count", 0),
            "cache_hits": usage.get("cache_hits", 0),
            "retry_count": usage.get("retry_count", 0),
            "failure_count": usage.get("failure_count", 0),
            "visited_node_count": len(visited_nodes),
            "checked_node_count": len(checked_node_ids),
            "checked_view_count": len(checked_view_ids),
            "checked_object_count": checked_object_count,
            "context_size_chars": context_metrics.get("context_size_chars"),
            "prompt_size_chars": context_metrics.get("prompt_size_chars"),
            "estimated_input_tokens": context_metrics.get("estimated_input_tokens"),
            "estimated_token_method": context_metrics.get("estimated_token_method"),
            "construction_cost": None,
        },
        "warnings": sorted(set(warnings)),
    }


def canonical_unavailable_result(
    *,
    run_id: str,
    method: str,
    mode: str,
    query: dict[str, Any],
    reason: str,
    depth_validation: dict[str, Any],
) -> dict[str, Any]:
    result = canonical_result(
        run_id=run_id,
        method=method,
        mode=mode,
        query=query,
        answer={
            "structured_plan": {"status": "unavailable"},
            "visited_nodes": [],
            "selected_views": [],
            "checked_view_ids": [],
            "result": {
                "found": False,
                "matched_object": None,
                "selected_node_id": None,
                "selected_view_id": None,
                "bbox_2d": None,
                "bbox_3d": None,
                "camera_pose": None,
                "confidence": 0.0,
                "explanation": reason,
            },
            "warnings": [reason, "No semantic prediction was produced."],
        },
        runtime_seconds=0.0,
        usage={
            "token_usage": None,
            "model_call_count": 0,
            "cache_hits": 0,
            "retry_count": 0,
            "failure_count": 0,
        },
        context_metrics={},
        depth_validation=depth_validation,
        geometry_eval_allowed=False,
    )
    result["availability_status"] = "unavailable"
    result["availability_reason"] = reason
    return result


def write_logs(run_dir: Path, logs: dict[str, Any]) -> None:
    write_json(run_dir / "logs.json", logs)


def write_evaluation(benchmark_path: Path, run_dir: Path) -> dict[str, Any]:
    summary = evaluate(benchmark_path, run_dir)
    write_json(run_dir / "metrics_summary.json", summary)
    (run_dir / "metrics_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    (run_dir / "failure_modes.md").write_text(failure_markdown(summary), encoding="utf-8")
    return summary


def run_experiment(
    config_path: str | Path,
    *,
    mode_override: str | None = None,
    query_limit_override: int | None = None,
    output_override: str | Path | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    run_id = str(config.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Config requires run_id")
    mode = str(mode_override or config.get("mode", "stub"))
    method = str(config.get("method", "semantic_splat"))
    temperature = float(config.get("model", {}).get("temperature", 0.0))
    benchmark_path = resolve_path(
        config.get("benchmark_path", "docs/benchmarks/benchmark_queries_v1.json")
    )
    output_root = resolve_path(config.get("output_root", "outputs/week2"))
    scene_root = resolve_path(config.get("scene_root", "backend/data/scenes"))
    run_dir = resolve_path(output_override) if output_override is not None else output_root / run_id
    if output_override is not None:
        run_id = run_dir.name
        if not run_id:
            raise ValueError("Output override must identify a run directory")
    query_limit = query_limit_override if query_limit_override is not None else config.get("query_limit")
    if query_limit is not None and (isinstance(query_limit, bool) or not isinstance(query_limit, int) or query_limit < 1):
        raise ValueError("query limit must be a positive integer")
    resume = bool(config.get("resume", True))

    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run directory already exists and resume=false: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "query_results").mkdir(exist_ok=True)
    (run_dir / "model_cache").mkdir(exist_ok=True)

    logs: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "started_at": utc_now(),
        "finished_at": None,
        "events": [],
        "warnings": [],
    }
    run_config: dict[str, Any] = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "week": config.get("week"),
        "method": method,
        "mode": mode,
        "status": "running",
        "started_at": logs["started_at"],
        "finished_at": None,
        "git_revision": git_revision(),
        "config_source": str(config_path),
        "benchmark_path": str(benchmark_path),
        "query_limit": query_limit,
        "output_dir": str(run_dir),
        "scene_ids": config.get("scene_ids", []),
        "scenes": config.get("scenes", []),
        "scene_status": {},
        "model": {
            "provider": None,
            "model": None,
            "version": None,
            "temperature": temperature,
        },
        "depth_validation": {"status": "unchecked", "reason": "Scene not loaded yet."},
        "warnings": [],
    }
    write_json(run_dir / "run_config.json", run_config)
    write_logs(run_dir, logs)

    try:
        model_settings = config.get("model", {})
        cache_setting = model_settings.get("cache_dir") if isinstance(model_settings, dict) else None
        cache_dir = resolve_path(cache_setting) if cache_setting else run_dir / "model_cache"
        run_config["model_cache_dir"] = str(cache_dir)
        client = create_model_client(mode, cache_dir=cache_dir, temperature=temperature)
        run_config["model"] = client.provenance()
        log_event(logs, "model_client", "complete", **client.provenance())

        benchmark = load_json(benchmark_path)
        benchmark_queries = benchmark.get("queries")
        if not isinstance(benchmark_queries, list):
            raise ValueError("Benchmark must contain a queries list")
        scene_entries = configured_scene_entries(config, scene_root, benchmark_queries)
        run_config["scene_ids"] = [entry["scene_id"] for entry in scene_entries]
        run_config["scenes"] = [
            {
                "scene_id": entry["scene_id"],
                "benchmark_scene_id": entry["benchmark_scene_id"],
                "path": str(entry["path"]),
                "semantic_eval_allowed": entry["semantic_eval_allowed"],
                "geometry_eval_allowed": entry["geometry_eval_allowed"],
                "validation_report": str(entry["validation_report"]) if entry["validation_report"] else None,
            }
            for entry in scene_entries
        ]
        benchmark_to_scene_id = {
            entry["benchmark_scene_id"]: entry["scene_id"] for entry in scene_entries
        }

        active_scene_ids: set[str] | None = None
        if isinstance(query_limit, int):
            active_scene_ids = set()
            scoped_count = 0
            for query in benchmark_queries:
                if not isinstance(query, dict):
                    continue
                scene_id = benchmark_to_scene_id.get(str(query.get("scene_id")))
                if scene_id is None:
                    continue
                active_scene_ids.add(scene_id)
                scoped_count += 1
                if scoped_count >= query_limit:
                    break

        scene_bundles: dict[str, dict[str, Any]] = {}
        unavailable_scenes: dict[str, str] = {}
        all_depth: dict[str, dict[str, Any]] = {}
        description_runtime = 0.0
        tree_runtime = 0.0
        for scene_entry in scene_entries:
            scene_id = scene_entry["scene_id"]
            if active_scene_ids is not None and scene_id not in active_scene_ids:
                run_config["scene_status"][scene_id] = {
                    "status": "skipped_query_limit",
                    "reason": "No query for this scene is inside the active query limit.",
                }
                continue
            scene_path = scene_entry["path"]
            scene, _ = timed_stage(logs, f"scene_loading:{scene_id}", lambda path=scene_path: load_scene(path))
            depth, _ = timed_stage(
                logs,
                f"depth_validation:{scene_id}",
                lambda s=scene: validate_depth(s["scene_dir"], int(config.get("depth_sample_limit", 3))),
            )
            all_depth[scene_id] = depth

            selected, _ = timed_stage(
                logs,
                f"view_selection:{scene_id}",
                lambda s=scene: select_views(s, config.get("view_selection", {})),
            )
            if scene_entry["semantic_eval_allowed"] is False:
                reason = "Skipped because semantic_eval_allowed=false in the validated scene config."
                run_config["scene_status"][scene_id] = {
                    "status": "skipped_ineligible",
                    "reason": reason,
                    "candidate_view_count": len(selected),
                    "semantic_index_format": scene["semantic_index_format"],
                    "semantic_index_complete": scene["semantic_index_complete"],
                }
                unavailable_scenes[scene_id] = reason
                run_config["warnings"].append(f"Scene {scene_id}: {reason}")
                log_event(logs, f"scene_batch:{scene_id}", "skipped", reason=reason)
                continue
            if (
                scene["semantic_index_format"] == "captured_viewjson_v1"
                and not scene["semantic_index_complete"]
            ):
                raise ValueError(
                    f"Scene {scene_id} is marked eligible but its captured semantic index is incomplete"
                )
            if not selected:
                raise ValueError(f"No views selected for scene {scene_id}")
            descriptions: dict[str, dict[str, Any]] = {}

            def describe_selected() -> dict[str, dict[str, Any]]:
                for view_id in selected:
                    descriptions[view_id] = client.describe_view(
                        image_path_for(scene, view_id),
                        {"view_id": view_id, "existing_analysis": scene["existing_views"].get(view_id)},
                    )
                return descriptions

            descriptions, elapsed = timed_stage(logs, f"view_descriptions:{scene_id}", describe_selected)
            description_runtime += elapsed
            bbox_enrichment, _ = timed_stage(
                logs,
                f"bbox_enrichment:{scene_id}",
                lambda s=scene, d=descriptions: enrich_view_bboxes(s, d),
            )
            tree, elapsed = timed_stage(
                logs,
                f"tree_loading_or_construction:{scene_id}",
                lambda s=scene, d=descriptions: client.build_tree(
                    d,
                    {
                        "scene_id": scene_id,
                        "existing_tree": s["existing_tree"],
                        "view_selection_strategy": config.get("view_selection", {}).get("strategy", "existing"),
                    },
                ),
            )
            tree_runtime += elapsed
            scene_bundles[scene_id] = {
                "scene": scene,
                "selected_view_ids": selected,
                "views": descriptions,
                "tree": tree,
                "tree_manifest": scene.get("tree_manifest"),
                "bbox_enrichment": bbox_enrichment,
                "geometry_eval_allowed": scene_entry["geometry_eval_allowed"],
            }
            run_config["scene_status"][scene_id] = {
                "status": "ready",
                "candidate_view_count": len(selected),
                "semantic_index_format": scene["semantic_index_format"],
                "bbox_enrichment": bbox_enrichment,
            }

        if len(all_depth) == 1:
            run_config["depth_validation"] = next(iter(all_depth.values()))
        else:
            statuses = {value.get("status") for value in all_depth.values()}
            run_config["depth_validation"] = {
                "status": "reliable" if statuses == {"reliable"} else "unreliable",
                "scenes": all_depth,
                "reason": "All scenes must be reliable for aggregate 3D metrics.",
            }
        official_gt_bbox_count = sum(
            int(bundle.get("bbox_enrichment", {}).get("bbox_3d_existing", 0))
            for bundle in scene_bundles.values()
            if isinstance(bundle.get("tree_manifest"), dict)
            and bundle["tree_manifest"].get("semantic_index_mode") == "official_gt"
        )
        if official_gt_bbox_count:
            run_config["bbox_3d_evaluation"] = {
                "status": "official_dataset_gt_enabled",
                "bbox_3d_count": official_gt_bbox_count,
                "gt_source": "official dataset semantic/instance object boxes",
                "note": "Official GT boxes are scored in dataset coordinates and do not depend on captured depth reliability.",
            }
        if mode == "stub":
            run_config["warnings"].append("Stub mode uses deterministic semantic-index fallback, not model reasoning.")
        if run_config["depth_validation"].get("status") != "reliable":
            if official_gt_bbox_count:
                run_config["warnings"].append(
                    "Depth is not reliable for RGB-D unprojection; official dataset bbox_3d outputs remain enabled."
                )
            else:
                run_config["warnings"].append("Depth is not reliable; canonical bbox_3d outputs and 3D IoU are disabled.")
        if any(
            bundle.get("bbox_enrichment", {}).get("bbox_3d_generated", 0) > 0
            for bundle in scene_bundles.values()
        ):
            run_config["warnings"].append(
                "Some predicted bbox_3d values were generated from semantic-index bbox_2d or approx_location plus captured depth; these are coarse predictions, not GT."
            )
        run_config["construction_cost"] = {
            "runtime_seconds": round(description_runtime + tree_runtime, 6),
            "view_descriptions_runtime_seconds": round(description_runtime, 6),
            "tree_runtime_seconds": round(tree_runtime, 6),
            "token_usage": client.stats_snapshot().get("total_tokens"),
        }
        write_json(run_dir / "run_config.json", run_config)

        query_count = 0
        for query in benchmark_queries:
            if not isinstance(query, dict):
                continue
            benchmark_scene_id = str(query.get("scene_id"))
            scene_id = benchmark_to_scene_id.get(benchmark_scene_id, benchmark_scene_id)
            if scene_id not in scene_bundles and scene_id not in unavailable_scenes:
                continue
            if isinstance(query_limit, int) and query_count >= query_limit:
                break
            query_count += 1
            query_id = str(query["query_id"])
            effective_query = dict(query)
            if scene_id != benchmark_scene_id:
                effective_query["benchmark_scene_id"] = benchmark_scene_id
                effective_query["scene_id"] = scene_id
            output_path = run_dir / "query_results" / f"{query_id}.json"
            if resume and output_path.exists():
                try:
                    existing = load_json(output_path)
                    if existing.get("schema_version") == QUERY_RESULT_SCHEMA and existing.get("mode") == mode:
                        log_event(logs, f"query:{query_id}", "resumed", result_file=str(output_path))
                        continue
                except (OSError, ValueError, json.JSONDecodeError):
                    pass

            if scene_id in unavailable_scenes:
                result = canonical_unavailable_result(
                    run_id=run_id,
                    method=method,
                    mode=mode,
                    query=effective_query,
                    reason=unavailable_scenes[scene_id],
                    depth_validation=all_depth[scene_id],
                )
                write_json(output_path, result)
                log_event(
                    logs,
                    f"query:{query_id}",
                    "unavailable",
                    reason=unavailable_scenes[scene_id],
                    result_file=str(output_path),
                )
                write_logs(run_dir, logs)
                continue

            bundle = scene_bundles[scene_id]
            context_metrics = estimate_query_context_metrics(effective_query, bundle["tree"], bundle["views"])
            before = client.stats_snapshot()
            started = time.perf_counter()
            answer = client.answer_query(effective_query, bundle["tree"], bundle["views"])
            runtime = round(time.perf_counter() - started, 6)
            usage = stats_delta(before, client.stats_snapshot())
            result = canonical_result(
                run_id=run_id,
                method=method,
                mode=mode,
                query=effective_query,
                answer=answer,
                runtime_seconds=runtime,
                usage=usage,
                context_metrics=context_metrics,
                depth_validation=all_depth[scene_id],
                geometry_eval_allowed=bundle["geometry_eval_allowed"],
            )
            write_json(output_path, result)
            log_event(
                logs,
                f"query:{query_id}",
                "complete",
                runtime_seconds=runtime,
                found=result["result"]["found"],
                result_file=str(output_path),
            )
            write_logs(run_dir, logs)

        summary, _ = timed_stage(logs, "metrics_evaluation", lambda: write_evaluation(benchmark_path, run_dir))
        logs["finished_at"] = utc_now()
        logs["warnings"] = list(run_config["warnings"])
        run_config["status"] = "complete"
        run_config["finished_at"] = logs["finished_at"]
        run_config["result_coverage"] = summary.get("coverage")
        write_json(run_dir / "run_config.json", run_config)
        write_logs(run_dir, logs)
        return run_dir
    except Exception as exc:
        logs["finished_at"] = utc_now()
        log_event(logs, "run", "error", error=str(exc), error_type=type(exc).__name__)
        run_config["status"] = "failed"
        run_config["finished_at"] = logs["finished_at"]
        run_config["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(run_dir / "run_config.json", run_config)
        write_logs(run_dir, logs)
        if isinstance(exc, ModelClientError):
            raise SystemExit(f"Model client error: {exc}") from exc
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a headless SemanticSplat experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("stub", "live", "cached_live"), help="Override config mode")
    parser.add_argument("--limit", type=int, help="Override the maximum number of scoped benchmark queries")
    parser.add_argument("--out", type=Path, help="Override the complete run output directory")
    args = parser.parse_args()
    run_dir = run_experiment(
        args.config,
        mode_override=args.mode,
        query_limit_override=args.limit,
        output_override=args.out,
    )
    print(f"Experiment complete: {run_dir}")


if __name__ == "__main__":
    main()
