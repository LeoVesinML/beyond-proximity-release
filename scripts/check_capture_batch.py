#!/usr/bin/env python3
"""Validate a batch of manually captured PLY scenes without hiding failures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_capture_pilot import check_capture


def unavailable_report(scene_dir: Path, error: Exception) -> dict[str, Any]:
    return {
        "scene": str(scene_dir.resolve()),
        "rgb_count": 0,
        "depth_count": 0,
        "pose_count": 0,
        "intrinsics_status": "invalid_or_missing",
        "metadata_status": "incomplete_or_missing",
        "rgb_resolution": None,
        "depth_resolution": None,
        "depth_min": None,
        "depth_max": None,
        "depth_mean": None,
        "depth_is_constant": None,
        "zero_depth_ratio": None,
        "pose_validity": {
            "valid_count": 0,
            "pose_count": 0,
            "all_valid": False,
        },
        "intrinsics_match_canvas": False,
        "semantic_index": {
            "view_json_count": 0,
            "invalid_view_json_count": 0,
            "missing_view_ids": [],
            "tree_node_count": 0,
            "tree_valid": False,
            "tree_covers_all_views": False,
            "complete": False,
        },
        "semantic_eval_allowed": False,
        "geometry_eval_allowed": False,
        "scaling_allowed": False,
        "warnings": [f"Capture validation unavailable: {error}"],
    }


def check_capture_batch(scene_dirs: Iterable[Path], out_dir: Path) -> list[dict[str, Any]]:
    scenes = list(scene_dirs)
    scene_ids = [scene.name for scene in scenes]
    duplicates = sorted({scene_id for scene_id in scene_ids if scene_ids.count(scene_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate scene folder names: {', '.join(duplicates)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for scene_dir in scenes:
        try:
            report = check_capture(scene_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report = unavailable_report(scene_dir, exc)

        output_path = out_dir / f"{scene_dir.name}_capture_check.json"
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports.append(report)
        print(
            f"Wrote {output_path} "
            f"scaling={str(report['scaling_allowed']).lower()} "
            f"semantic={str(report['semantic_eval_allowed']).lower()} "
            f"geometry={str(report['geometry_eval_allowed']).lower()}"
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Check multiple manual PLY capture directories")
    parser.add_argument("--scenes", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        reports = check_capture_batch(args.scenes, args.out)
    except ValueError as exc:
        raise SystemExit(f"Capture batch check failed: {exc}") from exc

    print(
        f"summary scenes={len(reports)} "
        f"scaling_allowed={sum(bool(report['scaling_allowed']) for report in reports)} "
        f"semantic_allowed={sum(bool(report['semantic_eval_allowed']) for report in reports)} "
        f"geometry_allowed={sum(bool(report['geometry_eval_allowed']) for report in reports)}"
    )


if __name__ == "__main__":
    main()
