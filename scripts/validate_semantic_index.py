#!/usr/bin/env python3
"""Validate captured ViewJSON/tree structure and semantic execution eligibility."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.io.captured_semantic_index import (  # noqa: E402
    QUERY_RUNNER_SCHEMA_COMPATIBLE,
    TREE_MANIFEST_SCHEMA_VERSION,
    TREE_NODE_SCHEMA_VERSION,
    inspect_view_index,
    load_json,
    semantic_counts,
)
from scripts.run_experiment import load_scene  # noqa: E402


VALID_SEMANTIC_NODE_TYPES = {"region", "object", "landmark", "sign", "facility"}
REPORT_SCHEMA_VERSION = "semanticsplat.semantic_index_validation.v1"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_tree(scene_dir: Path, expected_view_ids: set[str]) -> dict[str, Any]:
    tree_dir = scene_dir / "tree"
    manifest_path = tree_dir / "manifest.json"
    result: dict[str, Any] = {
        "manifest_exists": manifest_path.is_file(),
        "root_node_exists": False,
        "node_count": 0,
        "semantic_node_count": 0,
        "complete": False,
        "valid": False,
        "errors": [],
    }
    if not manifest_path.is_file():
        result["errors"].append("Missing tree/manifest.json")
        return result
    try:
        manifest = load_json(manifest_path)
        if manifest.get("schema_version") != TREE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"manifest schema_version must be {TREE_MANIFEST_SCHEMA_VERSION!r}")
        if manifest.get("scene_id") != scene_dir.name:
            raise ValueError("manifest scene_id does not match scene directory")
        root_id = manifest.get("root_node_id")
        if not isinstance(root_id, str) or not root_id:
            raise ValueError("manifest root_node_id is missing")
        node_ids = manifest.get("node_ids")
        if not isinstance(node_ids, list) or not node_ids or not all(isinstance(item, str) for item in node_ids):
            raise ValueError("manifest node_ids must be a non-empty string array")
        if manifest.get("node_count") != len(node_ids):
            raise ValueError("manifest node_count does not match node_ids")

        nodes: dict[str, dict[str, Any]] = {}
        for node_id in node_ids:
            path = tree_dir / f"node_{node_id}.json"
            if not path.is_file():
                raise ValueError(f"missing tree node file {path.name}")
            node = load_json(path)
            if node.get("schema_version") != TREE_NODE_SCHEMA_VERSION:
                raise ValueError(f"{path.name} has unsupported schema_version")
            if node.get("scene_id") != scene_dir.name or node.get("node_id") != node_id:
                raise ValueError(f"{path.name} scene_id/node_id mismatch")
            node_type = node.get("node_type")
            if node_type not in {"root", "zone", "region", "object", "landmark", "sign", "facility", "unknown"}:
                raise ValueError(f"{path.name} has unsupported node_type={node_type!r}")
            children = node.get("children_ids")
            views = node.get("view_ids")
            if not isinstance(children, list) or not all(isinstance(item, str) for item in children):
                raise ValueError(f"{path.name} children_ids must be a string array")
            if not isinstance(views, list) or not all(isinstance(item, str) for item in views):
                raise ValueError(f"{path.name} view_ids must be a string array")
            nodes[node_id] = node

        if root_id not in nodes or nodes[root_id].get("node_type") != "root":
            raise ValueError("root node is missing or is not node_type=root")
        result["root_node_exists"] = True
        dangling = sorted(
            {child for node in nodes.values() for child in node["children_ids"]} - set(nodes)
        )
        if dangling:
            raise ValueError("tree contains dangling children: " + ", ".join(dangling))
        root_views = set(nodes[root_id]["view_ids"])
        if not expected_view_ids.issubset(root_views):
            raise ValueError("root node does not cover every captured ViewJSON")

        semantic_nodes = [node for node in nodes.values() if node["node_type"] in VALID_SEMANTIC_NODE_TYPES]
        result["node_count"] = len(nodes)
        result["semantic_node_count"] = len(semantic_nodes)
        result["complete"] = bool(manifest.get("complete"))
        result["manifest"] = manifest
        result["valid"] = bool(result["complete"] and semantic_nodes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["errors"].append(str(exc))
    return result


def validate_semantic_index(scene_dir: Path) -> dict[str, Any]:
    scene_dir = scene_dir.resolve()
    inspection = inspect_view_index(scene_dir, scene_dir / "views")
    views = inspection["valid_views"]
    counts = semantic_counts(views)
    expected_ids = set(inspection["expected_view_ids"])
    view_index_complete = bool(
        expected_ids
        and len(views) == len(expected_ids)
        and not inspection["invalid_files"]
        and not inspection["missing_view_ids"]
        and not inspection["unexpected_view_ids"]
    )
    modes = sorted({view["semantic_index_mode"] for view in views})
    mode = modes[0] if len(modes) == 1 else None
    tree = validate_tree(scene_dir, expected_ids)

    runner_loadable = False
    runner_error: str | None = None
    try:
        loaded = load_scene(scene_dir)
        runner_loadable = bool(
            set(loaded["existing_views"]) == expected_ids
            and "root" in loaded["existing_tree"]
        )
        if not runner_loadable:
            runner_error = "run_experiment.load_scene did not load all views and the root node"
    except Exception as exc:
        runner_error = str(exc)

    warnings: list[str] = []
    if counts["semantic_item_count"] == 0:
        warnings.append("No semantic object, region, landmark, sign, or facility is annotated.")
    if counts["annotated_view_count"] < len(expected_ids):
        warnings.append(
            f"{len(expected_ids) - counts['annotated_view_count']} ViewJSON files have empty summaries."
        )
    if not tree["complete"]:
        warnings.append("Semantic tree is missing or marked incomplete.")
    if not QUERY_RUNNER_SCHEMA_COMPATIBLE:
        warnings.append("Current query runner does not yet search captured ViewJSON v1 fields; Phase H is required.")
    if mode == "stub":
        warnings.append("Stub semantic index must not be reported as manual/live/cached-live output.")
    if counts["bbox_3d_count"] and mode != "official_gt":
        warnings.append("Annotated/predicted bbox_3d values are not independent ground truth.")
    if mode == "official_gt":
        warnings.append("Semantic content and bbox_3d values are official dataset ground truth.")
    warnings.append("Independent GT 3D boxes/masks and validated predicted boxes are unavailable.")

    semantic_eval_allowed = bool(
        view_index_complete
        and counts["semantic_item_count"] > 0
        and tree["valid"]
        and runner_loadable
        and QUERY_RUNNER_SCHEMA_COMPATIBLE
    )
    errors = [f"{item['file']}: {item['error']}" for item in inspection["invalid_files"]]
    errors.extend(tree["errors"])
    if runner_error:
        errors.append(f"query runner load error: {runner_error}")

    if mode == "official_gt" and counts["bbox_3d_count"]:
        warnings = [
            warning
            for warning in warnings
            if warning != "Independent GT 3D boxes/masks and validated predicted boxes are unavailable."
        ]

    geometry_eval_allowed = bool(mode == "official_gt" and counts["bbox_3d_count"] > 0 and tree["valid"])

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scene_id": scene_dir.name,
        "scene": str(scene_dir),
        "expected_view_count": len(expected_ids),
        "view_json_count": inspection["template_file_count"],
        "valid_view_json_count": len(views),
        "invalid_view_json_count": len(inspection["invalid_files"]),
        "missing_view_ids": inspection["missing_view_ids"],
        "unexpected_view_ids": inspection["unexpected_view_ids"],
        "semantic_index_mode": mode,
        **counts,
        "tree_manifest_exists": tree["manifest_exists"],
        "root_node_exists": tree["root_node_exists"],
        "node_count": tree["node_count"],
        "semantic_node_count": tree["semantic_node_count"],
        "tree_complete": tree["complete"],
        "tree_valid": tree["valid"],
        "query_runner_loadable": runner_loadable,
        "query_runner_schema_compatible": QUERY_RUNNER_SCHEMA_COMPATIBLE,
        "semantic_eval_allowed": semantic_eval_allowed,
        "geometry_eval_allowed": geometry_eval_allowed,
        "errors": errors,
        "warnings": sorted(set(warnings)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate captured-scene semantic indexes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", type=Path)
    group.add_argument("--scenes", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    scenes = [args.scene] if args.scene is not None else list(args.scenes)
    assert all(scene is not None for scene in scenes)
    batch = len(scenes) > 1 or args.scenes is not None
    if batch:
        args.out.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for raw_scene in scenes:
        assert raw_scene is not None
        try:
            report = validate_semantic_index(raw_scene)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Semantic index validation failed for {raw_scene}: {exc}") from exc
        output_path = args.out / f"{raw_scene.name}_semantic_index.json" if batch else args.out
        write_json(output_path, report)
        reports.append(report)
        print(
            f"Wrote {output_path} scene={report['scene_id']} views={report['valid_view_json_count']} "
            f"tree={str(report['tree_valid']).lower()} "
            f"semantic={str(report['semantic_eval_allowed']).lower()}"
        )
    print(
        f"summary scenes={len(reports)} "
        f"semantic_allowed={sum(bool(report['semantic_eval_allowed']) for report in reports)} "
        f"geometry_allowed={sum(bool(report['geometry_eval_allowed']) for report in reports)}"
    )


if __name__ == "__main__":
    main()
