#!/usr/bin/env python3
"""Freeze the exact Person 1 room0 plus scene0011_00 20-query pilot."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_scannet_official_queries import build_artifacts  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_pilot(
    *,
    replica_benchmark_path: Path,
    nr3d_path: Path,
    sr3d_path: Path,
    scene_root: Path,
) -> dict[str, Any]:
    replica = read_json(replica_benchmark_path)
    replica_queries = [
        query
        for query in replica.get("queries", [])
        if isinstance(query, dict) and query.get("scene_id") == "replica_room0"
    ]
    if len(replica_queries) != 7:
        raise ValueError(f"Expected 7 frozen Replica room0 queries, found {len(replica_queries)}")
    scannet, _ = build_artifacts(
        nr3d_path=nr3d_path,
        sr3d_path=sr3d_path,
        scene_root=scene_root,
        scenes=[("scannet_0011_00", "scene0011_00")],
        per_dataset_per_scene=7,
        expected_nr3d_count=None,
        expected_sr3d_count=None,
    )
    scannet_queries = list(scannet["queries"][:13])
    if len(scannet_queries) != 13:
        raise ValueError(f"Expected 13 ScanNet pilot queries, found {len(scannet_queries)}")
    queries = replica_queries + scannet_queries
    source_counts = Counter(str(query.get("source_dataset")) for query in queries)
    return {
        "schema_version": "semanticsplat.benchmark_queries.v1",
        "benchmark_id": "replica_room0_scannet_0011_person1_pilot_v1",
        "status": "ready_official_gt",
        "status_date": datetime.now(timezone.utc).date().isoformat(),
        "scene_scope": ["replica_room0", "scannet_0011_00"],
        "query_count": len(queries),
        "source_query_counts": dict(sorted(source_counts.items())),
        "selection_protocol": (
            "All seven frozen Replica room0 queries plus the first thirteen queries from "
            "the deterministic seven-per-source Nr3D/Sr3D+ scene0011_00 stratification. "
            "Selection does not inspect model results."
        ),
        "ground_truth_scope": (
            "Replica targets use official Replica semantic object IDs and boxes. ScanNet "
            "targets use official ReferIt3D IDs mapped to official ScanNet aggregation boxes."
        ),
        "queries": queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replica-benchmark",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "replica_scannet_pilot_queries.json",
    )
    parser.add_argument(
        "--nr3d",
        type=Path,
        default=ROOT / "data" / "scannet" / "annotations" / "nr3d.csv",
    )
    parser.add_argument(
        "--sr3d-plus",
        type=Path,
        default=ROOT / "data" / "scannet" / "annotations" / "sr3d" / "sr3d+.csv",
    )
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend" / "data" / "scenes")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "replica_scannet_room0_scene0011_pilot_v1.json",
    )
    args = parser.parse_args()
    pilot = build_pilot(
        replica_benchmark_path=args.replica_benchmark,
        nr3d_path=args.nr3d,
        sr3d_path=args.sr3d_plus,
        scene_root=args.scene_root,
    )
    write_json(args.out, pilot)
    print(f"Wrote exact {pilot['query_count']}-query Person 1 pilot to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
