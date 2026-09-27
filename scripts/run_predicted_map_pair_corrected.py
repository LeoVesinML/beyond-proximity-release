#!/usr/bin/env python3
"""Corrected predicted-map flat/hierarchical BGE evaluation (IEEE revision 2).

Differences from scripts/run_predicted_map_pair.py (which is left unchanged):
  * quality denominators contain positives only (queries with >=1 GT box);
    negatives are reported separately and never scored as IoU 0;
  * Replica GT boxes are the world AABB of the official oriented boxes
    (R * abb + t from info_semantic.json), written by
    scripts/audit_predicted_map_frames.py to
    outputs/ieee_revision2/predicted_maps/replica_queries_corrected_gt_boxes.json;
    the original importer used ``oriented_bbox.abb`` (object-local frame) as a
    world box.
Requires the native ConceptGraphs maps
(<root>/native_data/<scene>/exps/*mapping*/pcd_*.pkl.gz).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.agent_protocol import DEFAULT_LOCAL_MODEL, FastEmbedSemanticEncoder  # noqa: E402
from scripts.evaluate_results import box_bounds, expected_3d_boxes  # noqa: E402
from scripts.run_predicted_map_pair import _load_map, _map_path, _result_row, _split_spatial, _summarize  # noqa: E402


def run(sources: dict[str, tuple[Path, Path]], out: Path) -> dict[str, Any]:
    encoder = FastEmbedSemanticEncoder(cache_dir=ROOT / ".cache/fastembed")
    summaries, all_rows, missing = [], [], []
    for dataset, (map_root, bench_path) in sources.items():
        queries = json.loads(bench_path.read_text())["queries"]
        scene_ids = json.loads((map_root / "run_config.json").read_text())["scene_ids"]
        positives = [q for q in queries if q["scene_id"] in scene_ids
                     and any(box_bounds(b) is not None for b in expected_3d_boxes(q))]
        rows = {"flat_bge_predicted_map": [], "graph_bge_predicted_map": []}
        for scene_id in scene_ids:
            try:
                objects = _load_map(_map_path(map_root, scene_id))
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            centers = np.vstack([o.center for o in objects])
            branches = _split_spatial(list(range(len(objects))), centers)
            vecs = encoder.encode([o.text for o in objects])
            sq = [q for q in positives if q["scene_id"] == scene_id]
            for q, v in zip(sq, encoder.encode([str(q["query"]) for q in sq])):
                for method in rows:
                    rows[method].append(_result_row(dataset=dataset, scene_id=scene_id, query=q, objects=objects,
                                                    similarities=vecs @ v, branches=branches, method=method, encoder=encoder))
        if rows["flat_bge_predicted_map"]:
            summaries += [dict(_summarize(rows["flat_bge_predicted_map"]), dataset=dataset),
                          dict(_summarize(rows["graph_bge_predicted_map"], rows["flat_bge_predicted_map"]), dataset=dataset)]
            all_rows += rows["flat_bge_predicted_map"] + rows["graph_bge_predicted_map"]
    payload = {"schema_version": "beyond_proximity.predicted_map_pair.corrected_v2", "model_name": DEFAULT_LOCAL_MODEL,
               "status": "complete" if not missing else "incomplete_missing_maps", "missing": missing,
               "denominator": "positives only", "summaries": summaries,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.mkdir(parents=True, exist_ok=True)
    (out / "predicted_map_pair_corrected_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "predicted_map_pair_corrected_per_query.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scannet-root", type=Path, default=ROOT / "outputs/baselines/conceptgraphs_scannet_full_v1")
    ap.add_argument("--replica-root", type=Path, default=ROOT / "outputs/baselines/conceptgraphs_replica_full_v1")
    ap.add_argument("--scannet-benchmark", type=Path,
                    default=Path("/Users/leo/beyond-proximity-week6-person2/docs/benchmarks/scannet_bbq_grounding_pilot_v1.json"))
    ap.add_argument("--replica-benchmark", type=Path,
                    default=ROOT / "outputs/ieee_revision2/predicted_maps/replica_queries_corrected_gt_boxes.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision2/predicted_maps/predicted_map_pair_corrected")
    args = ap.parse_args()
    result = run({"ScanNet": (args.scannet_root, args.scannet_benchmark),
                  "Replica": (args.replica_root, args.replica_benchmark)}, args.out)
    print(json.dumps({k: result[k] for k in ("status", "missing", "summaries")}, indent=2))


if __name__ == "__main__":
    main()
