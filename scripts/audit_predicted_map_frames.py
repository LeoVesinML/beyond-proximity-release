#!/usr/bin/env python3
"""Revision-2 audit of the predicted-map (ConceptGraphs) object-grounding evaluation.

B1  Recompute saved predicted-map metrics over positives only (flat BGE,
    hierarchical BGE, native ConceptGraphs).
B2  GT-as-prediction sanity test through the same IoU code, then frame and
    box-convention checks between native ConceptGraphs boxes and GT boxes,
    and a corrected evaluation where a real bug is found.

Only saved artifacts are read; nothing here re-runs ConceptGraphs.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.recall_controlled import wilson_interval  # noqa: E402
from scripts.evaluate_results import bbox_iou_3d, best_bbox_iou_3d, box_bounds, expected_3d_boxes  # noqa: E402
from scripts.run_predicted_map_pair import PredictedObject, _result_row, _split_spatial  # noqa: E402

DEFAULT_SCANNET_BENCH = Path("/Users/leo/beyond-proximity-week6-person2/docs/benchmarks/scannet_bbq_grounding_pilot_v1.json")
DEFAULT_SCANNET_SCENES = Path("/Users/leo/beyond-proximity-week6-person2/backend/data/scenes")
DEFAULT_SCANNET_RAW = Path("/Users/leo/beyond-proximity-week6-person2/data/scannet/scans")
REPLICA_BENCH = ROOT / "docs/benchmarks/replica_scannet_pilot_queries.json"
REPLICA_SCENES = ROOT / "backend/data/scenes"
PAIR_ROWS = ROOT / "outputs/ieee_revision/predicted_map_pair_v1/predicted_map_pair_per_query.json"
CG_ROOT = {"ScanNet": ROOT / "outputs/baselines/conceptgraphs_scannet_full_v1",
           "Replica": ROOT / "outputs/baselines/conceptgraphs_replica_full_v1"}


def load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_positive(query: dict[str, Any]) -> bool:
    return any(box_bounds(box) is not None for box in expected_3d_boxes(query))


def corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def aabb(points: np.ndarray) -> dict[str, list[float]]:
    return {"min": points.min(0).tolist(), "max": points.max(0).tolist()}


def bounds_np(box: Any) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = box_bounds(box)
    return np.asarray(lo, float), np.asarray(hi, float)


def transform_box(box: Any, matrix: np.ndarray) -> dict[str, list[float]]:
    lo, hi = bounds_np(box)
    pts = corners(lo, hi)
    homog = np.hstack([pts, np.ones((8, 1))]) @ matrix.T
    return aabb(homog[:, :3])


def replica_obb_world_box(obj: dict[str, Any], mode: str) -> dict[str, list[float]]:
    """World AABB of an info_semantic oriented box.

    ``abb`` is the box in the object's local frame; world = R * p + t, with the
    quaternion stored as [x, y, z, w].  ``mode='as_used'`` reproduces the
    original importer, which dropped the orientation.
    """
    lo, hi = bounds_np(obj)
    if mode == "as_used":
        return {"min": lo.tolist(), "max": hi.tolist()}
    rot = Rotation.from_quat(obj["orientation"]["rotation"]).as_matrix()
    if mode == "R_transpose":
        rot = rot.T
    t = np.asarray(obj["orientation"]["translation"], float)
    return aabb(corners(lo, hi) @ rot.T + t)


def axis_alignment(raw_root: Path, scene_id: str) -> np.ndarray:
    folder = "scene" + scene_id.removeprefix("scannet_")
    for line in (raw_root / folder / f"{folder}.txt").read_text().splitlines():
        if line.strip().startswith("axisAlignment"):
            return np.asarray([float(v) for v in line.split("=")[1].split()], float).reshape(4, 4)
    raise ValueError(f"axisAlignment missing for {scene_id}")


def tokens(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", str(text).lower()))
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


def summarize(values: list[float], thresholds=(0.25, 0.5)) -> dict[str, Any]:
    n = len(values)
    out: dict[str, Any] = {"n": n, "mean_iou": round(float(np.mean(values)), 6) if n else None}
    for t in thresholds:
        k = int(sum(v >= t for v in values))
        lo, hi = wilson_interval(k, n)
        out[f"acc_at_{t}"] = {"value": round(k / n, 6) if n else None, "numerator": k, "denominator": n,
                              "wilson_95": [round(lo, 6), round(hi, 6)]}
    return out


def dist_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values)
    return {"n": len(values), "median_m": round(float(np.median(arr)), 4), "p25_m": round(float(np.percentile(arr, 25)), 4),
            "p75_m": round(float(np.percentile(arr, 75)), 4), "frac_within_0.5m": round(float((arr <= 0.5).mean()), 4),
            "frac_within_1m": round(float((arr <= 1.0).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scannet-benchmark", type=Path, default=DEFAULT_SCANNET_BENCH)
    ap.add_argument("--scannet-scenes", type=Path, default=DEFAULT_SCANNET_SCENES)
    ap.add_argument("--scannet-raw", type=Path, default=DEFAULT_SCANNET_RAW)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision2/predicted_maps")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    benches = {"ScanNet": load(args.scannet_benchmark)["queries"], "Replica": load(REPLICA_BENCH)["queries"]}
    scene_root = {"ScanNet": args.scannet_scenes, "Replica": REPLICA_SCENES}
    gt_objects: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for ds, queries in benches.items():
        gt_objects[ds] = {}
        for scene in sorted({q["scene_id"] for q in queries}):
            gt_objects[ds][scene] = load(scene_root[ds] / scene / "ground_truth/object_boxes_3d.json")["objects"]

    # ---------------- native ConceptGraphs predictions ----------------
    native: dict[str, dict[str, dict[str, Any]]] = {}
    for ds, root in CG_ROOT.items():
        native[ds] = {}
        for path in sorted(glob.glob(str(root / "native_results_*.json"))):
            for pred in load(Path(path))["predictions"]:
                native[ds][pred["query_id"]] = pred
    known_pred_boxes: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for ds in native:
        known_pred_boxes[ds] = {}
        qscene = {q["query_id"]: q["scene_id"] for q in benches[ds]}
        for qid, pred in native[ds].items():
            if pred.get("bbox_3d"):
                known_pred_boxes[ds].setdefault(qscene[qid], {})[pred["selected_object_id"]] = {
                    "box": pred["bbox_3d"], "label": pred.get("matched_object") or ""}

    report: dict[str, Any] = {"schema_version": "beyond_proximity.ieee_revision2.predicted_map_audit.v1"}

    # ---------------- B1: denominators ----------------
    rows = load(PAIR_ROWS)
    b1: dict[str, Any] = {}
    for ds, queries in benches.items():
        pos_ids = {q["query_id"] for q in queries if is_positive(q)}
        neg_ids = {q["query_id"] for q in queries if not is_positive(q)}
        entry: dict[str, Any] = {"benchmark_queries": len(queries), "positives": len(pos_ids), "negatives": len(neg_ids)}
        for method in ("flat_bge_predicted_map", "graph_bge_predicted_map"):
            mrows = [r for r in rows if r["dataset"] == ds and r["method"] == method]
            for label, subset in (("as_reported_all_queries", mrows),
                                  ("positives_only", [r for r in mrows if r["query_id"] in pos_ids])):
                s = summarize([r["top1_iou"] for r in subset])
                k3 = int(sum(r["recall_at_3_iou_0_25"] for r in subset))
                s["recall_at_3_iou_0_25"] = {"value": round(k3 / len(subset), 6), "numerator": k3, "denominator": len(subset)}
                s["mean_checked_objects"] = round(float(np.mean([r["checked_objects"] for r in subset])), 4)
                s["mean_serialized_top3_tokens"] = round(float(np.mean([r["serialized_top3_tokens"] for r in subset])), 4)
                entry[f"{method}:{label}"] = s
            neg_rows = [r for r in mrows if r["query_id"] in neg_ids]
            entry[f"{method}:negatives_in_denominator"] = {
                "count": len(neg_rows), "top1_iou_values": sorted({r["top1_iou"] for r in neg_rows}),
                "note": "best_bbox_iou_3d returns None for an empty expected box list; `or 0.0` scores it as a miss"}
        nat = [best_bbox_iou_3d(native[ds][q["query_id"]].get("bbox_3d"), q) or 0.0 for q in queries if q["query_id"] in pos_ids]
        entry["native_conceptgraphs:positives_only"] = summarize(nat)
        b1[ds] = entry
    report["B1_denominators"] = b1

    # ---------------- B2a: GT-as-prediction sanity ----------------
    class _Tok:
        def token_count(self, text: str) -> int:
            return len(str(text).split())

    sanity: dict[str, Any] = {}
    for ds, queries in benches.items():
        direct, flat_row, graph_row = [], [], []
        for q in [q for q in queries if is_positive(q)]:
            gt_box = expected_3d_boxes(q)[0]
            lo, hi = bounds_np(gt_box)
            direct.append(best_bbox_iou_3d({"min": lo.tolist(), "max": hi.tolist()}, q) or 0.0)
            objs = []
            for obj in gt_objects[ds][q["scene_id"]]:
                olo, ohi = bounds_np(obj)
                objs.append(PredictedObject(object_id=str(obj["object_id"]), text=str(obj["label"]),
                                            bbox={"min": olo.tolist(), "max": ohi.tolist()}, center=(olo + ohi) / 2))
            target_ids = {str(b.get("object_id")) for b in expected_3d_boxes(q)}
            sims = np.array([1.0 if o.object_id in target_ids else 0.0 for o in objs])
            branches = _split_spatial(list(range(len(objs))), np.vstack([o.center for o in objs]))
            for method, sink in (("flat_bge_predicted_map", flat_row), ("graph_bge_predicted_map", graph_row)):
                row = _result_row(dataset=ds, scene_id=q["scene_id"], query=q, objects=objs, similarities=sims,
                                  branches=branches, method=method, encoder=_Tok())
                sink.append(row["top1_iou"])
        sanity[ds] = {"direct_best_bbox_iou_3d": summarize(direct),
                      "through_run_predicted_map_pair._result_row_flat": summarize(flat_row),
                      "through_run_predicted_map_pair._result_row_graph": summarize(graph_row)}
    report["B2a_gt_as_prediction_sanity"] = sanity

    # ---------------- B2b: frame / convention checks ----------------
    def frame_metrics(ds: str, gt_box_fn, pred_tf) -> dict[str, Any]:
        """gt_box_fn(scene, gt_obj) -> box; pred_tf(scene, box) -> box."""
        best_any, center_any, center_same, pred_size, gt_size = [], [], [], [], []
        target_hit_any = []
        native_iou = []
        for scene, objs in gt_objects[ds].items():
            gboxes = {str(o["object_id"]): gt_box_fn(scene, o) for o in objs}
            glabels = {str(o["object_id"]): o.get("canonical_label") or o["label"] for o in objs}
            preds = {pid: {"box": pred_tf(scene, rec["box"]), "label": rec["label"]}
                     for pid, rec in known_pred_boxes[ds].get(scene, {}).items()}
            for pid, rec in preds.items():
                ious = [(bbox_iou_3d(rec["box"], g) or 0.0, gid) for gid, g in gboxes.items()]
                best, gid = max(ious)
                best_any.append(best)
                plo, phi = bounds_np(rec["box"])
                glo, ghi = bounds_np(gboxes[gid])
                pred_size.append(float(np.prod(phi - plo) ** (1 / 3)))
                gt_size.append(float(np.prod(ghi - glo) ** (1 / 3)))
            for q in [q for q in benches[ds] if q["scene_id"] == scene and is_positive(q)]:
                tids = [str(b.get("object_id")) for b in expected_3d_boxes(q)]
                tboxes = [gboxes[t] for t in tids if t in gboxes]
                tcent = [np.mean(bounds_np(b), axis=0) for b in tboxes]
                if preds:
                    pc = {pid: np.mean(bounds_np(r["box"]), axis=0) for pid, r in preds.items()}
                    center_any.append(min(np.linalg.norm(c - p) for c in tcent for p in pc.values()))
                    ttoks = set().union(*(tokens(glabels[t]) for t in tids if t in glabels))
                    same = [p for pid, p in pc.items() if tokens(preds[pid]["label"]) & ttoks]
                    if same:
                        center_same.append(min(np.linalg.norm(c - p) for c in tcent for p in same))
                    target_hit_any.append(max((bbox_iou_3d(r["box"], tb) or 0.0) for r in preds.values() for tb in tboxes))
                pred = native[ds][q["query_id"]].get("bbox_3d")
                native_iou.append(max(((bbox_iou_3d(pred_tf(scene, pred), tb) or 0.0) for tb in tboxes), default=0.0) if pred else 0.0)
        return {
            "known_predicted_boxes": len(best_any),
            "pred_box_best_iou_any_gt": {"median": round(float(np.median(best_any)), 4),
                                         "frac_ge_0.25": round(float(np.mean(np.array(best_any) >= 0.25)), 4),
                                         "frac_ge_0.5": round(float(np.mean(np.array(best_any) >= 0.5)), 4)},
            "gt_target_center_to_nearest_known_pred_center_any_label": dist_stats(center_any),
            "gt_target_center_to_nearest_known_pred_center_same_label": dist_stats(center_same),
            "frac_gt_targets_with_known_pred_iou_ge_0.25_lower_bound": round(float(np.mean(np.array(target_hit_any) >= 0.25)), 4),
            "frac_gt_targets_with_known_pred_iou_ge_0.5_lower_bound": round(float(np.mean(np.array(target_hit_any) >= 0.5)), 4),
            "median_cuberoot_volume_pred_m": round(float(np.median(pred_size)), 4),
            "median_cuberoot_volume_matched_gt_m": round(float(np.median(gt_size)), 4),
            "native_conceptgraphs_top1_positives": summarize(native_iou),
        }

    ident = lambda scene, box: box  # noqa: E731
    as_gt = lambda scene, obj: obj  # noqa: E731
    yz_swap = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)  # Y-up -> Z-up
    zy_swap = np.linalg.inv(yz_swap)
    align = {s: axis_alignment(args.scannet_raw, s) for s in gt_objects["ScanNet"]}
    frames: dict[str, Any] = {"ScanNet": {}, "Replica": {}}
    frames["ScanNet"]["identity (as evaluated)"] = frame_metrics("ScanNet", as_gt, ident)
    frames["ScanNet"]["pred <- axisAlignment @ pred (hyp: preds in raw .sens frame)"] = frame_metrics(
        "ScanNet", as_gt, lambda s, b: transform_box(b, align[s]))
    frames["ScanNet"]["pred <- inv(axisAlignment) @ pred (hyp: double alignment)"] = frame_metrics(
        "ScanNet", as_gt, lambda s, b: transform_box(b, np.linalg.inv(align[s])))
    frames["ScanNet"]["pred Y-up -> Z-up"] = frame_metrics("ScanNet", as_gt, lambda s, b: transform_box(b, yz_swap))
    gt_modes = {"as_used (abb, orientation dropped)": "as_used", "R*abb+t (orientation applied)": "R",
                "R^T*abb+t": "R_transpose"}
    for name, mode in gt_modes.items():
        fn = (lambda m: (lambda s, o: replica_obb_world_box(o, m)))(mode)
        frames["Replica"][f"GT {name}; pred identity"] = frame_metrics("Replica", fn, ident)
    frames["Replica"]["GT as_used; pred Y-up -> Z-up"] = frame_metrics(
        "Replica", lambda s, o: replica_obb_world_box(o, "as_used"), lambda s, b: transform_box(b, yz_swap))
    frames["Replica"]["GT as_used; pred Z-up -> Y-up"] = frame_metrics(
        "Replica", lambda s, o: replica_obb_world_box(o, "as_used"), lambda s, b: transform_box(b, zy_swap))
    report["B2b_frame_checks"] = frames
    report["B2b_notes"] = {
        "known_predicted_boxes": "The raw ConceptGraphs maps (native_data/*/pcd_*.pkl.gz) are not on this machine; only the "
                                 "boxes of objects that native ConceptGraphs returned as top-1 for some query are saved. Recall "
                                 "fractions over these boxes are therefore LOWER bounds on the recall ceiling of any retriever.",
        "frames_used": {"ScanNet_gt": "scannet_axis_aligned_mesh (prepare_scannet_scene.py applies axisAlignment to vertices and to poses)",
                        "ScanNet_pred": "ConceptGraphs mapping with relative_pose=False on the axis-aligned poses written by prepare_scannet_scene.py",
                        "Replica_gt": "info_semantic.json oriented_bbox.abb used as a world AABB (import_replica_semantic_mesh.py:boxes_from_info_semantic)",
                        "Replica_pred": "ConceptGraphs mapping on NICE-SLAM traj.txt camera-to-world poses (mesh frame, Z-up)"},
    }

    # ---------------- corrected evaluation (native ConceptGraphs; Replica GT orientation fix) -------------
    corrected_queries = []
    rep_obj = {s: {str(o["object_id"]): o for o in objs} for s, objs in gt_objects["Replica"].items()}
    for q in benches["Replica"]:
        q2 = dict(q)
        if is_positive(q):
            q2["expected_bboxes_3d"] = [
                {**replica_obb_world_box(rep_obj[q["scene_id"]][str(b["object_id"])], "R"), "object_id": str(b["object_id"]),
                 "label": b.get("label"), "coordinate_frame": "replica_mesh_world_obb_aabb"}
                for b in expected_3d_boxes(q)]
            q2["expected_bbox_3d"] = q2["expected_bboxes_3d"][0]
        corrected_queries.append(q2)
    (args.out / "replica_queries_corrected_gt_boxes.json").write_text(json.dumps(
        {"note": "Replica pilot queries with expected boxes replaced by the world AABB of the official oriented box "
                 "(R*abb+t). Local derived file; same queries/IDs as docs/benchmarks/replica_scannet_pilot_queries.json.",
         "queries": corrected_queries}, indent=1) + "\n")
    corrected = {}
    pos = [q for q in corrected_queries if is_positive(q)]
    corrected["native_conceptgraphs_replica_positives_corrected_gt"] = summarize(
        [best_bbox_iou_3d(native["Replica"][q["query_id"]].get("bbox_3d"), q) or 0.0 for q in pos])
    corrected["native_conceptgraphs_replica_positives_original_gt"] = b1["Replica"]["native_conceptgraphs:positives_only"]
    corrected["native_conceptgraphs_scannet_positives_unchanged"] = b1["ScanNet"]["native_conceptgraphs:positives_only"]
    # Partial BGE re-score: only rows whose top-1 object has a saved box.
    boxes_by_id = {pid: rec["box"] for s in known_pred_boxes["Replica"].values() for pid, rec in s.items()}
    by_qid = {q["query_id"]: q for q in pos}
    for method in ("flat_bge_predicted_map", "graph_bge_predicted_map"):
        mrows = [r for r in rows if r["dataset"] == "Replica" and r["method"] == method and r["query_id"] in by_qid]
        covered = [r for r in mrows if r["ranked_object_ids"][0] in boxes_by_id]
        corrected[f"{method}_replica_top1_box_available"] = {
            "positives": len(mrows), "top1_box_saved": len(covered),
            "status": "blocked: predicted maps absent, corrected full BGE rerun not possible on this machine",
            "covered_subset_original_gt": summarize([r["top1_iou"] for r in covered]),
            "covered_subset_corrected_gt": summarize([best_bbox_iou_3d(boxes_by_id[r["ranked_object_ids"][0]], by_qid[r["query_id"]]) or 0.0 for r in covered]),
        }
    report["corrected_evaluation"] = corrected
    (args.out / "predicted_map_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1)[:20000])


if __name__ == "__main__":
    main()
