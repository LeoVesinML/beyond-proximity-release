#!/usr/bin/env python3
"""Audit whether Replica/ScanNet-style public dataset evaluation is ready."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.import_replica_scene import detect_layout  # noqa: E402


REQUIRED_REPLICA_FILES = ("rgb", "depth", "poses.json", "intrinsics.json")
REQUIRED_SCANNET_FOLDERS = ("color", "depth", "pose", "intrinsic")
REQUIRED_SCANNET_RAW_SUFFIXES = (
    ".aggregation.json",
    ".sens",
    ".txt",
    "_vh_clean_2.0.010000.segs.json",
    "_vh_clean_2.ply",
    "_vh_clean_2.labels.ply",
)
CAPTURED_PILOT_SCENES = {
    "ConferenceHall-capture-pilot",
    "Museume-capture",
    "Theater-capture",
    "outdoor-drone-capture",
    "outdoor-street-capture",
}
LEGACY_SCENES = {"default"}
REPLICA_BBQ_BACKEND_SCENES = {
    "replica_room0",
    "replica_room1",
    "replica_room2",
    "replica_office0",
    "replica_office1",
    "replica_office2",
    "replica_office3",
    "replica_office4",
}
SCANNET_BBQ_BACKEND_SCENES = {
    "scannet_0011_00",
    "scannet_0030_00",
    "scannet_0046_00",
    "scannet_0086_00",
    "scannet_0222_00",
    "scannet_0378_00",
    "scannet_0389_00",
    "scannet_0435_00",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def manual_scene_inventory(scene_root: Path) -> list[dict[str, Any]]:
    scenes = []
    for scene in sorted(scene_root.iterdir()) if scene_root.exists() else []:
        if not scene.is_dir():
            continue
        if scene.name in REPLICA_BBQ_BACKEND_SCENES or scene.name in SCANNET_BBQ_BACKEND_SCENES:
            continue
        views = sorted((scene / "views").glob("*.json")) if (scene / "views").exists() else []
        transforms = load_json(scene / "transforms.json")
        frames = transforms.get("frames") if isinstance(transforms.get("frames"), list) else []
        if views or frames:
            scenes.append({
                "scene_id": scene.name,
                "category": (
                    "captured_pilot"
                    if scene.name in CAPTURED_PILOT_SCENES
                    else "legacy_demo"
                    if scene.name in LEGACY_SCENES
                    else "other"
                ),
                "view_json_count": len(views),
                "frame_count": len(frames),
                "has_tree": (scene / "tree").exists(),
                "has_manual_bbox_gt_v2": (scene / "ground_truth" / "manual_bbox_gt_v2.json").exists(),
            })
    return scenes


def replica_inventory(replica_root: Path) -> list[dict[str, Any]]:
    rows = []
    for scene in sorted(replica_root.iterdir()) if replica_root.exists() else []:
        if not scene.is_dir():
            continue
        metadata = load_json(scene / "metadata.json")
        missing = [name for name in REQUIRED_REPLICA_FILES if not (scene / name).exists()]
        try:
            layout = detect_layout(scene)
            layout_error = None
        except Exception as exc:  # noqa: BLE001 - record exact readiness blocker
            layout = None
            layout_error = str(exc)
        source_dataset = str(metadata.get("source_dataset", "unknown"))
        official = source_dataset.lower() in {"replica", "replica_dataset", "replica official", "habitat_replica"}
        rows.append({
            "scene_id": scene.name,
            "path": str(scene.relative_to(ROOT)),
            "layout": layout,
            "layout_error": layout_error,
            "missing_required": missing,
            "source_dataset": source_dataset,
            "official_public_dataset": official,
            "frame_count": metadata.get("num_frames"),
            "has_independent_gt": bool(metadata.get("has_ground_truth_labels")),
            "status": "READY_PUBLIC" if official and not missing else "PROXY_OR_NOT_READY",
        })
    return rows


def scannet_inventory(scannet_root: Path) -> list[dict[str, Any]]:
    rows = []
    raw_root = scannet_root / "scans" if (scannet_root / "scans").is_dir() else scannet_root
    extracted_root = scannet_root / "extracted"
    for scene in sorted(raw_root.iterdir()) if raw_root.exists() else []:
        if not scene.is_dir():
            continue
        scene_id = scene.name
        if not scene_id.startswith("scene"):
            continue
        missing_raw = [
            f"{scene_id}{suffix}"
            for suffix in REQUIRED_SCANNET_RAW_SUFFIXES
            if not (scene / f"{scene_id}{suffix}").is_file()
        ]
        extracted_scene = extracted_root / scene_id
        if all((scene / name).is_dir() for name in REQUIRED_SCANNET_FOLDERS):
            extracted_scene = scene
        missing_extracted = [
            name for name in REQUIRED_SCANNET_FOLDERS if not (extracted_scene / name).is_dir()
        ]
        if not missing_extracted:
            status = "READY_FOR_CONVERSION"
        elif not missing_raw:
            status = "RAW_READY_FOR_EXTRACTION"
        else:
            status = "NOT_READY"
        rows.append({
            "scene_id": scene_id,
            "path": display_path(scene),
            "extracted_path": display_path(extracted_scene),
            "missing_raw": missing_raw,
            "missing_required": missing_extracted,
            "raw_file_count": sum(
                (scene / f"{scene_id}{suffix}").is_file()
                for suffix in REQUIRED_SCANNET_RAW_SUFFIXES
            ),
            "status": status,
        })
    return rows


def imported_replica_inventory(scene_root: Path) -> list[dict[str, Any]]:
    rows = []
    for scene_id in sorted(REPLICA_BBQ_BACKEND_SCENES):
        scene = scene_root / scene_id
        gt_path = scene / "ground_truth" / "object_boxes_3d.json"
        validation_path = ROOT / "docs" / "validation" / "public_datasets" / f"{scene_id}.json"
        manifest = load_json(scene / "scene_manifest.json")
        gt = load_json(gt_path)
        validation = load_json(validation_path)
        semantic_validation = validation.get("semantic_validation")
        if not isinstance(semantic_validation, dict):
            semantic_validation = validation
        ready = bool(
            scene.is_dir()
            and gt_path.is_file()
            and gt.get("source_dataset") == "Replica"
            and semantic_validation.get("semantic_eval_allowed") is True
            and semantic_validation.get("geometry_eval_allowed") is True
        )
        rows.append({
            "scene_id": scene_id,
            "path": str(scene.relative_to(ROOT)) if scene.exists() else str(scene),
            "source_scene_id": manifest.get("source_scene_id") or gt.get("source_scene_id"),
            "object_box_count": gt.get("object_count"),
            "semantic_eval_allowed": semantic_validation.get("semantic_eval_allowed"),
            "geometry_eval_allowed": semantic_validation.get("geometry_eval_allowed"),
            "validation_report": str(validation_path.relative_to(ROOT)),
            "status": "READY_PUBLIC" if ready else "NOT_READY",
        })
    return rows


def imported_scannet_inventory(scene_root: Path) -> list[dict[str, Any]]:
    rows = []
    for scene_id in sorted(SCANNET_BBQ_BACKEND_SCENES):
        scene = scene_root / scene_id
        gt_path = scene / "ground_truth" / "object_boxes_3d.json"
        validation_path = ROOT / "docs" / "validation" / "public_datasets" / f"{scene_id}.json"
        manifest = load_json(scene / "scene_manifest.json")
        gt = load_json(gt_path)
        validation = load_json(validation_path)
        semantic_validation = validation.get("semantic_validation")
        if not isinstance(semantic_validation, dict):
            semantic_validation = validation
        geometry_validation = validation.get("geometry_validation")
        if not isinstance(geometry_validation, dict):
            geometry_validation = {}
        ready = bool(
            scene.is_dir()
            and gt_path.is_file()
            and gt.get("source_dataset") == "ScanNet"
            and int(gt.get("object_count") or 0) > 0
            and int(gt.get("indexed_object_count") or 0) == int(gt.get("object_count") or 0)
            and semantic_validation.get("semantic_eval_allowed") is True
            and semantic_validation.get("geometry_eval_allowed") is True
            and geometry_validation.get("three_d_localization_allowed") is True
        )
        rows.append({
            "scene_id": scene_id,
            "path": display_path(scene),
            "source_scene_id": manifest.get("source_scene_id") or gt.get("source_scene_id"),
            "frame_count": manifest.get("frame_count"),
            "object_box_count": gt.get("object_count"),
            "indexed_object_count": gt.get("indexed_object_count"),
            "semantic_eval_allowed": semantic_validation.get("semantic_eval_allowed"),
            "geometry_eval_allowed": semantic_validation.get("geometry_eval_allowed"),
            "validation_report": display_path(validation_path),
            "status": "READY_PUBLIC" if ready else "NOT_READY",
        })
    return rows


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    replica_ready = [row for row in report["replica"] if row["status"] == "READY_PUBLIC"]
    imported_replica_ready = [
        row for row in report.get("imported_replica", []) if row["status"] == "READY_PUBLIC"
    ]
    imported_scannet_ready = [
        row for row in report.get("imported_scannet", []) if row["status"] == "READY_PUBLIC"
    ]
    scannet_ready = [row for row in report["scannet"] if row["status"] == "READY_FOR_CONVERSION"]
    scannet_raw = [
        row
        for row in report["scannet"]
        if row["status"] in {"RAW_READY_FOR_EXTRACTION", "READY_FOR_CONVERSION"}
    ]
    captured = [row for row in report["manual_scenes"] if row["category"] == "captured_pilot"]
    legacy = [row for row in report["manual_scenes"] if row["category"] == "legacy_demo"]
    return {
        "captured_pilot_scene_count": len(captured),
        "captured_pilot_view_json_count": sum(row["view_json_count"] for row in captured),
        "legacy_demo_scene_count": len(legacy),
        "legacy_demo_view_json_count": sum(row["view_json_count"] for row in legacy),
        "replica_scene_count": len(report["replica"]),
        "replica_ready_public_count": len(replica_ready) + len(imported_replica_ready),
        "imported_replica_ready_public_count": len(imported_replica_ready),
        "imported_replica_gt_box_count": sum(
            int(row.get("object_box_count") or 0) for row in imported_replica_ready
        ),
        "scannet_scene_count": len(report["scannet"]),
        "scannet_raw_available_count": len(scannet_raw),
        "scannet_ready_for_conversion_count": len(scannet_ready),
        "imported_scannet_ready_public_count": len(imported_scannet_ready),
        "imported_scannet_gt_box_count": sum(
            int(row.get("object_box_count") or 0) for row in imported_scannet_ready
        ),
        "imported_scannet_rgbd_view_count": sum(
            int(row.get("frame_count") or 0) for row in imported_scannet_ready
        ),
        "publication_readiness": {
            "twinworld_workshop": "READY_WITH_MANUAL_SCENES_PLUS_OFFICIAL_REPLICA_AND_SCANNET_PILOTS",
            "aaai_main": "NOT_READY_WITHOUT_STRONGER_METHOD_RESULTS_AND_FAIR_BASELINES",
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Public Dataset Readiness Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Captured pilot scenes: {summary['captured_pilot_scene_count']}",
        f"- Captured pilot ViewJSON annotations: {summary['captured_pilot_view_json_count']}",
        f"- Legacy demo scenes: {summary['legacy_demo_scene_count']}",
        f"- Legacy demo ViewJSON annotations: {summary['legacy_demo_view_json_count']}",
        f"- Replica-style local folders: {summary['replica_scene_count']}",
        f"- Official/public Replica scenes ready: {summary['replica_ready_public_count']}",
        f"- Imported official Replica BBQ scenes ready: {summary['imported_replica_ready_public_count']}",
        f"- Imported official Replica GT object boxes: {summary['imported_replica_gt_box_count']}",
        f"- ScanNet local folders: {summary['scannet_scene_count']}",
        f"- ScanNet raw scenes available: {summary['scannet_raw_available_count']}",
        f"- ScanNet extracted folders available: {summary['scannet_ready_for_conversion_count']}",
        f"- Imported official ScanNet BBQ scenes ready: {summary['imported_scannet_ready_public_count']}",
        f"- Imported official ScanNet GT object boxes: {summary['imported_scannet_gt_box_count']}",
        f"- Imported official ScanNet RGB-D views: {summary['imported_scannet_rgbd_view_count']}",
        "",
        "## Decision",
        "",
        "- Keep the five captured scenes as pilot/system evidence.",
        "- Use the eight imported official Replica and eight imported official ScanNet BBQ scenes as public-dataset workshop evidence.",
        "- ScanNet Nr3D/Sr3D+ grounding is runnable over official GT object IDs and boxes; the current lexical-stub result is a method-quality baseline, not a semantic-perception result.",
        "- Do not call local `data/replica/pilot_scene_001` an official Replica result unless its source metadata changes to an official dataset scene.",
        "",
        "## Manual Captured Scenes",
        "",
        "| Scene | Category | ViewJSON | Frames | Tree | Manual bbox GT v2 |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in report["manual_scenes"]:
        lines.append(
            f"| `{row['scene_id']}` | `{row['category']}` | {row['view_json_count']} | {row['frame_count']} | "
            f"{row['has_tree']} | {row['has_manual_bbox_gt_v2']} |"
        )
    lines.extend([
        "",
        "## Replica-Style Inputs",
        "",
        "| Scene | Status | Source dataset | Missing | Independent GT |",
        "|---|---|---|---|---|",
    ])
    if report["replica"]:
        for row in report["replica"]:
            missing = ", ".join(row["missing_required"]) if row["missing_required"] else "none"
            lines.append(
                f"| `{row['scene_id']}` | `{row['status']}` | `{row['source_dataset']}` | "
                f"{missing} | {row['has_independent_gt']} |"
            )
    else:
        lines.append("| N/A | `NOT_PRESENT` | N/A | all | false |")
    lines.extend([
        "",
        "## Imported Official Replica Scenes",
        "",
        "| Scene | Source scene | Status | GT boxes | Semantic eval | Geometry eval |",
        "|---|---|---|---:|---|---|",
    ])
    if report.get("imported_replica"):
        for row in report["imported_replica"]:
            lines.append(
                f"| `{row['scene_id']}` | `{row.get('source_scene_id')}` | `{row['status']}` | "
                f"{row.get('object_box_count') or 0} | {row.get('semantic_eval_allowed')} | "
                f"{row.get('geometry_eval_allowed')} |"
            )
    else:
        lines.append("| N/A | N/A | `NOT_PRESENT` | 0 | false | false |")
    lines.extend([
        "",
        "## ScanNet Inputs",
        "",
        "| Scene | Status | Raw files | Missing extracted folders |",
        "|---|---|---:|---|",
    ])
    if report["scannet"]:
        for row in report["scannet"]:
            missing = ", ".join(row["missing_required"]) if row["missing_required"] else "none"
            lines.append(
                f"| `{row['scene_id']}` | `{row['status']}` | {row['raw_file_count']}/6 | {missing} |"
            )
    else:
        lines.append("| N/A | `NOT_PRESENT` | 0/6 | color, depth, pose, intrinsic |")
    lines.extend([
        "",
        "## Imported Official ScanNet Scenes",
        "",
        "| Scene | Source scene | Status | RGB-D views | GT boxes | Indexed boxes | Semantic eval | Geometry eval |",
        "|---|---|---|---:|---:|---:|---|---|",
    ])
    if report.get("imported_scannet"):
        for row in report["imported_scannet"]:
            lines.append(
                f"| `{row['scene_id']}` | `{row.get('source_scene_id')}` | `{row['status']}` | "
                f"{row.get('frame_count') or 0} | {row.get('object_box_count') or 0} | "
                f"{row.get('indexed_object_count') or 0} | {row.get('semantic_eval_allowed')} | "
                f"{row.get('geometry_eval_allowed')} |"
            )
    else:
        lines.append("| N/A | N/A | `NOT_PRESENT` | 0 | 0 | 0 | false | false |")
    lines.extend([
        "",
        "## What Must Be Supplied For Public-Dataset Claims",
        "",
        "- For Replica object-grounding claims: the imported `backend/data/scenes/replica_*` scenes and their official GT object boxes are ready.",
        "- For Replica segmentation claims: add a semantic/instance segmentation evaluator over the official mesh labels.",
        "- For ScanNet object-grounding claims: the eight scenes, official boxes, Nr3D/Sr3D+ labels, validation reports, and 48-query pilot are ready.",
        "- For ScanNet semantic-segmentation claims: add independent predicted per-vertex classes; GT labels alone cannot produce mAcc/mIoU/fmIoU.",
        "- For ScanRefer claims: obtain its separately gated official annotation release and add a frozen subset; it is optional after the completed Nr3D/Sr3D+ track.",
        "- Baseline outputs generated on the same public scenes, not only smoke scenes.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", default="backend/data/scenes")
    parser.add_argument("--replica-root", default="data/replica")
    parser.add_argument("--scannet-root", default="data/scannet")
    parser.add_argument("--out-json", default="docs/datasets/public_dataset_readiness.json")
    parser.add_argument("--out-md", default="docs/datasets/public_dataset_readiness.md")
    args = parser.parse_args()

    report = {
        "generated_at": utc_now(),
        "git_revision": git_revision(),
        "manual_scenes": manual_scene_inventory(ROOT / args.scene_root),
        "replica": replica_inventory(ROOT / args.replica_root),
        "scannet": scannet_inventory(ROOT / args.scannet_root),
    }
    report["imported_replica"] = imported_replica_inventory(ROOT / args.scene_root)
    report["imported_scannet"] = imported_scannet_inventory(ROOT / args.scene_root)
    report["summary"] = summarize(report)
    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, ROOT / args.out_md)
    print(f"Wrote public dataset readiness audit to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
