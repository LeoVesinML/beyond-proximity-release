#!/usr/bin/env python3
"""Validate raw RGB-D hierarchy construction and comparison outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "raw_rgbd_hierarchy" / "five_scene_clip_v1",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    errors: list[str] = []
    required = (
        "raw_hierarchy.json",
        "metrics_summary.json",
        "per_query_results.json",
        "per_query_metrics.csv",
        "run_config.json",
        "metrics_summary.md",
    )
    for name in required:
        if not (output / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1

    hierarchy = json.loads((output / "raw_hierarchy.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "per_query_results.json").read_text(encoding="utf-8"))
    with (output / "per_query_metrics.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    policy = hierarchy.get("source_policy", {})
    if policy.get("manual_viewjson_files_read") != 0:
        errors.append("manual ViewJSON was read during construction")
    if policy.get("manual_zone_files_read") != 0:
        errors.append("manual zones were read during construction")
    if hierarchy.get("totals", {}).get("rgb_image_count") != 97:
        errors.append("raw hierarchy does not contain 97 RGB images")
    if hierarchy.get("totals", {}).get("depth_map_count") != 97:
        errors.append("raw hierarchy does not contain 97 depth maps")
    if hierarchy.get("totals", {}).get("model_call_count", 0) < 1:
        errors.append("construction model calls were not recorded")
    if hierarchy.get("totals", {}).get("text_tokens", 0) < 1:
        errors.append("construction semantic-label tokens were not recorded")
    if len(hierarchy.get("scenes", [])) != 5:
        errors.append("raw hierarchy does not contain five scenes")
    assigned = 0
    for scene in hierarchy.get("scenes", []):
        view_ids = {
            view_id
            for cluster in scene.get("clusters", [])
            for view_id in cluster.get("view_ids", [])
        }
        if len(view_ids) != scene.get("view_count"):
            errors.append(f"{scene.get('scene_id')}: incomplete cluster assignment")
        assigned += len(view_ids)
    if assigned != 97:
        errors.append("cluster assignments do not cover 97 unique scene views")
    if len(rows) != 150:
        errors.append(f"expected 150 query rows, found {len(rows)}")
    methods = tuple(summary.get("methods", {}))
    if len(methods) != 3:
        errors.append("expected three hierarchy comparison methods")
    if len(csv_rows) != 150 * 3:
        errors.append("CSV does not contain 450 method rows")
    if summary.get("quality_query_count") != 125:
        errors.append("quality denominator is not 125")
    if not summary.get("structure", {}).get("reference_loaded_after_construction"):
        errors.append("manual reference ordering is not recorded")
    if summary.get("query_encoder", {}).get("model_call_count", 0) < 1:
        errors.append("query encoder calls were not recorded")
    if summary.get("query_encoder", {}).get("input_tokens", 0) < 1:
        errors.append("query encoder token usage was not recorded")

    report = {
        "schema_version": "semanticsplat.raw_rgbd_hierarchy_validation.v1",
        "status": "passed" if not errors else "failed",
        "scene_count": len(hierarchy.get("scenes", [])),
        "raw_view_count": assigned,
        "query_count": len(rows),
        "csv_row_count": len(csv_rows),
        "errors": errors,
    }
    write_path = output / "validation_report.json"
    write_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
