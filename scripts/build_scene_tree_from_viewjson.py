#!/usr/bin/env python3
"""Build a deterministic captured-scene semantic tree from real ViewJSON annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ZONE_GT_PATH = ROOT / "docs" / "benchmarks" / "manual_zone_gt_v2.json"

from backend.io.captured_semantic_index import (  # noqa: E402
    QUERY_RUNNER_SCHEMA_COMPATIBLE,
    TREE_MANIFEST_SCHEMA_VERSION,
    TREE_NODE_SCHEMA_VERSION,
    inspect_view_index,
    semantic_counts,
)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_")
    return normalized[:48] or "item"


def region_node_id(label: str) -> str:
    digest = hashlib.sha1(label.strip().casefold().encode("utf-8")).hexdigest()[:8]
    return f"region_{slug(label)}_{digest}"


def base_node(
    *,
    scene_id: str,
    node_id: str,
    node_type: str,
    name: str,
    summary: str,
    mode: str,
    view_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": TREE_NODE_SCHEMA_VERSION,
        "scene_id": scene_id,
        "node_id": node_id,
        "node_type": node_type,
        "name": name,
        "summary": summary,
        "semantic_index_mode": mode,
        "parent_id": "root" if node_type != "root" else None,
        "children_ids": [],
        "view_ids": sorted(set(view_ids)),
        "depth": 0 if node_type == "root" else 1,
    }


def semantic_nodes(scene_id: str, views: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    regions: dict[str, dict[str, Any]] = {}
    nodes: list[dict[str, Any]] = []
    for view in views:
        view_id = view["view_id"]
        for region in view["visible_regions"]:
            key = region["label"].strip().casefold()
            if key not in regions:
                node_id = region_node_id(region["label"])
                regions[key] = {
                    **base_node(
                        scene_id=scene_id,
                        node_id=node_id,
                        node_type="region",
                        name=region["label"],
                        summary=f"Region indexed from {mode} view annotations.",
                        mode=mode,
                        view_ids=[],
                    ),
                    "observations": [],
                }
            node = regions[key]
            node["view_ids"].append(view_id)
            node["observations"].append(
                {
                    "view_id": view_id,
                    "approx_location": region["approx_location"],
                    "source": region["source"],
                }
            )

        for index, obj in enumerate(view["visible_objects"], 1):
            node_id = f"object_{view_id}_{index:03d}_{slug(obj['label'])}"
            nodes.append(
                {
                    **base_node(
                        scene_id=scene_id,
                        node_id=node_id,
                        node_type="object",
                        name=obj["label"],
                        summary=f"Object observation from {view_id}.",
                        mode=mode,
                        view_ids=[view_id],
                    ),
                    "attributes": obj["attributes"],
                    "approx_location": obj["approx_location"],
                    "confidence": obj["confidence"],
                    "bbox_2d": obj["bbox_2d"],
                    "bbox_3d": obj["bbox_3d"],
                    "source": obj["source"],
                }
            )

        for index, landmark in enumerate(view["landmarks"], 1):
            kind = landmark["kind"]
            node_id = f"{kind}_{view_id}_{index:03d}_{slug(landmark['label'])}"
            nodes.append(
                {
                    **base_node(
                        scene_id=scene_id,
                        node_id=node_id,
                        node_type=kind,
                        name=landmark["label"],
                        summary=f"{kind.title()} observation from {view_id}.",
                        mode=mode,
                        view_ids=[view_id],
                    ),
                    "approx_location": landmark["approx_location"],
                    "source": landmark["source"],
                }
            )

    for node in regions.values():
        node["view_ids"] = sorted(set(node["view_ids"]))
        node["observations"] = sorted(node["observations"], key=lambda item: item["view_id"])
    return sorted([*regions.values(), *nodes], key=lambda node: node["node_id"])


def load_manual_zones(scene_id: str) -> list[dict[str, Any]]:
    if not ZONE_GT_PATH.is_file():
        return []
    data = json.loads(ZONE_GT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    for scene in data.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        if scene.get("captured_scene_id") != scene_id:
            continue
        zones = scene.get("zones", [])
        return [zone for zone in zones if isinstance(zone, dict)]
    return []


def zone_nodes(scene_id: str, zones: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for zone in zones:
        view_ids = [str(view_id) for view_id in zone.get("view_ids", [])]
        node = base_node(
            scene_id=scene_id,
            node_id=str(zone["zone_id"]),
            node_type="zone",
            name=str(zone["zone_label"]),
            summary=str(zone.get("description") or "Manual zone reference from captured-scene review."),
            mode=mode,
            view_ids=view_ids,
        )
        node["depth"] = 1
        node["zone_source"] = "docs/benchmarks/manual_zone_gt_v2.json"
        out.append(node)
    return sorted(out, key=lambda node: node["node_id"])


def assign_nodes_to_zones(nodes: list[dict[str, Any]], zones: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    if not zones:
        return nodes, [node["node_id"] for node in nodes]

    zone_by_id = {str(zone.get("zone_id") or zone.get("node_id")): zone for zone in zones}
    zone_children: dict[str, list[str]] = {zone_id: [] for zone_id in zone_by_id}
    root_children: list[str] = []

    for node in nodes:
        node_views = {str(view_id) for view_id in node.get("view_ids", [])}
        matches: list[tuple[int, int, str]] = []
        all_zone_ids: list[str] = []
        all_zone_labels: list[str] = []
        for zone_id, zone in zone_by_id.items():
            zone_views = {str(view_id) for view_id in zone.get("view_ids", [])}
            overlap = len(node_views & zone_views)
            if overlap <= 0:
                continue
            all_zone_ids.append(zone_id)
            all_zone_labels.append(str(zone.get("zone_label") or zone.get("name") or zone_id))
            matches.append((overlap, -len(zone_views), zone_id))
        if not matches:
            node["parent_id"] = "root"
            node["depth"] = 1
            root_children.append(str(node["node_id"]))
            continue

        matches.sort(reverse=True)
        best_zone_id = matches[0][2]
        node["parent_id"] = best_zone_id
        node["depth"] = 2
        node["zone_ids"] = sorted(all_zone_ids)
        node["zone_labels"] = sorted(set(all_zone_labels))
        zone_children[best_zone_id].append(str(node["node_id"]))

    for zone in zones:
        zone_id = str(zone.get("zone_id") or zone.get("node_id"))
        zone["children_ids"] = sorted(zone_children[zone_id])

    return nodes, root_children


def _prepare_output(out_dir: Path) -> set[str]:
    existing_json = list(out_dir.glob("*.json")) if out_dir.exists() else []
    if not existing_json:
        return set()
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Refusing to replace unowned tree JSON without manifest: {out_dir}")
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict) or existing.get("schema_version") != TREE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Refusing to replace a tree with a different schema: {manifest_path}")
    node_ids = existing.get("node_ids", [])
    return {str(node_id) for node_id in node_ids} if isinstance(node_ids, list) else set()


def build_scene_tree(scene_dir: Path, views_dir: Path, out_dir: Path) -> dict[str, Any]:
    scene_dir = scene_dir.resolve()
    views_dir = views_dir.resolve()
    out_dir = out_dir.resolve()
    inspection = inspect_view_index(scene_dir, views_dir)
    problems: list[str] = []
    if inspection["invalid_files"]:
        problems.extend(
            f"{item['file']}: {item['error']}" for item in inspection["invalid_files"]
        )
    if inspection["missing_view_ids"]:
        problems.append("missing ViewJSON: " + ", ".join(inspection["missing_view_ids"]))
    if inspection["unexpected_view_ids"]:
        problems.append("unexpected ViewJSON: " + ", ".join(inspection["unexpected_view_ids"]))
    if problems:
        raise ValueError("ViewJSON validation failed: " + " | ".join(problems))

    views = inspection["valid_views"]
    modes = {view["semantic_index_mode"] for view in views}
    if len(modes) != 1:
        raise ValueError(f"All ViewJSON files must use one semantic_index_mode, found {sorted(modes)}")
    mode = next(iter(modes))
    counts = semantic_counts(views)
    nodes = semantic_nodes(scene_dir.name, views, mode)
    zones = zone_nodes(scene_dir.name, load_manual_zones(scene_dir.name), mode)
    nodes, root_unzoned_children = assign_nodes_to_zones(nodes, zones)
    for zone in zones:
        zone["children_ids"] = list(zone.get("children_ids", []))
    root = base_node(
        scene_id=scene_dir.name,
        node_id="root",
        node_type="root",
        name=scene_dir.name,
        summary="Root of the captured-scene semantic index.",
        mode=mode,
        view_ids=[view["view_id"] for view in views],
    )
    root["children_ids"] = [node["node_id"] for node in zones] + root_unzoned_children
    all_nodes = [root, *zones, *nodes]

    warnings: list[str] = []
    empty_summary_count = len(views) - counts["annotated_view_count"]
    if counts["semantic_item_count"] == 0:
        warnings.append("Tree is incomplete: ViewJSON files contain no semantic objects, regions, or landmarks.")
    if empty_summary_count:
        warnings.append(f"Tree is incomplete: {empty_summary_count} ViewJSON files have empty summaries.")
    if counts["bbox_3d_count"] and mode != "official_gt":
        warnings.append("ViewJSON bbox_3d values are predictions/annotations, not independent 3D ground truth.")
    if mode == "official_gt":
        warnings.append("Official dataset GT semantic content; not manual annotation, stub output, or model output.")
    if zones:
        warnings.append("Manual zone nodes are reference labels from docs/benchmarks/manual_zone_gt_v2.json, not independent dataset GT.")
    if mode == "stub":
        warnings.append("Stub semantic content must not be reported as manual or live model output.")
    if not QUERY_RUNNER_SCHEMA_COMPATIBLE:
        warnings.append("Captured ViewJSON is not query-runner schema-compatible until Phase H.")
    complete = bool(
        views
        and counts["semantic_item_count"] > 0
        and counts["annotated_view_count"] == len(views)
    )

    manifest = {
        "schema_version": TREE_MANIFEST_SCHEMA_VERSION,
        "scene_id": scene_dir.name,
        "root_node_id": "root",
        "root_id": "root",
        "semantic_index_mode": mode,
        "node_ids": [node["node_id"] for node in all_nodes],
        "node_count": len(all_nodes),
        "view_count": len(views),
        "annotated_view_count": counts["annotated_view_count"],
        "semantic_item_count": counts["semantic_item_count"],
        "manual_zone_node_count": len(zones),
        "tree_native_zone_nodes": bool(zones),
        "complete": complete,
        "query_runner_schema_compatible": QUERY_RUNNER_SCHEMA_COMPATIBLE,
        "warnings": warnings,
    }

    old_node_ids = _prepare_output(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for node in all_nodes:
        write_json(out_dir / f"node_{node['node_id']}.json", node)
    write_json(out_dir / "manifest.json", manifest)
    current_node_ids = set(manifest["node_ids"])
    for stale_id in sorted(old_node_ids - current_node_ids):
        stale_path = out_dir / f"node_{stale_id}.json"
        if stale_path.is_file():
            stale_path.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a semantic tree from captured-scene ViewJSON")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_scene_tree(args.scene, args.views, args.out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Semantic tree build failed: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
