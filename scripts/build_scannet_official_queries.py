#!/usr/bin/env python3
"""Build a BBQ-aligned ScanNet grounding pilot from official ReferIt3D GT."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent

SCENES = [
    ("scannet_0011_00", "scene0011_00"),
    ("scannet_0030_00", "scene0030_00"),
    ("scannet_0046_00", "scene0046_00"),
    ("scannet_0086_00", "scene0086_00"),
    ("scannet_0222_00", "scene0222_00"),
    ("scannet_0378_00", "scene0378_00"),
    ("scannet_0389_00", "scene0389_00"),
    ("scannet_0435_00", "scene0435_00"),
]

EXPECTED_NR3D_CORRECT_COUNT = 699
EXPECTED_SR3D_UNIQUE_TRIPLET_COUNT = 661
SR3D_RELATION_ORDER = [
    "closest",
    "farthest",
    "between",
    "left",
    "right",
    "front",
    "back",
    "above",
    "below",
    "supporting",
    "supported-by",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected true/false, got {value!r}")
    return normalized == "true"


def parse_int_list(value: Any, field: str) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected a list for {field}, got {type(parsed).__name__}")
    result: list[str] = []
    for item in parsed:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"Expected integer IDs in {field}, got {item!r}")
        result.append(str(item))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        value = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def annotation_file_record(
    path: Path,
    *,
    source_url: str,
    source_id: str,
    row_count: int,
) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "source_url": source_url,
        "source_id": source_id,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "row_count": row_count,
    }


def load_scene_gt(
    scene_id: str,
    scene_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    scene_dir = scene_root / scene_id
    boxes_path = scene_dir / "ground_truth" / "object_boxes_3d.json"
    assignments_path = scene_dir / "ground_truth" / "object_view_assignments.json"
    boxes_data = read_json(boxes_path)
    assignments_data = read_json(assignments_path)
    boxes = {
        str(item["object_id"]): item
        for item in boxes_data.get("objects", [])
        if isinstance(item, dict) and item.get("object_id") is not None
    }
    assignments = {
        str(object_id): str(view_id)
        for object_id, view_id in assignments_data.get("object_to_view", {}).items()
    }
    nodes: dict[str, str] = {}
    for path in sorted((scene_dir / "tree").glob("node_*.json")):
        node = read_json(path)
        box = node.get("bbox_3d")
        if not isinstance(box, dict) or box.get("object_id") is None:
            continue
        nodes[str(box["object_id"])] = str(node.get("node_id") or path.stem.removeprefix("node_"))
    if not boxes:
        raise ValueError(f"No official object boxes found for {scene_id}")
    missing_assignments = sorted(set(boxes) - set(assignments), key=int)
    missing_nodes = sorted(set(boxes) - set(nodes), key=int)
    if missing_assignments or missing_nodes:
        raise ValueError(
            f"Incomplete imported GT for {scene_id}: "
            f"missing view assignments={missing_assignments}, missing tree nodes={missing_nodes}"
        )
    return boxes, assignments, nodes


def bbox_for_query(box: dict[str, Any]) -> dict[str, Any]:
    return {
        "center": box["center"],
        "size": box["size"],
        "label": box.get("canonical_label") or box["label"],
        "object_id": str(box["object_id"]),
        "coordinate_frame": box.get("coordinate_frame", "scannet_axis_aligned_mesh"),
    }


def nr3d_category(row: dict[str, str]) -> str:
    spatial = parse_bool(row["uses_spatial_lang"])
    attribute = parse_bool(row["uses_color_lang"]) or parse_bool(row["uses_shape_lang"])
    if spatial and attribute:
        return "spatial_attribute"
    if spatial:
        return "spatial"
    if attribute:
        return "attribute"
    return "plain"


def choose_nr3d(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    def stable_key(row: dict[str, str]) -> tuple[int, str, int, str]:
        assignment = str(row.get("assignmentid", ""))
        assignment_number = int(assignment) if assignment.isdigit() else 2**31 - 1
        return assignment_number, str(row.get("stimulus_id", "")), int(row["target_id"]), row["utterance"]

    ordered = sorted(rows, key=stable_key)
    selected: list[dict[str, str]] = []
    selected_targets: set[str] = set()
    category_groups = [
        {"spatial_attribute"},
        {"spatial"},
        {"attribute", "plain"},
    ]
    for categories in category_groups:
        candidate = next(
            (
                row
                for row in ordered
                if row not in selected
                and row["target_id"] not in selected_targets
                and nr3d_category(row) in categories
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_targets.add(candidate["target_id"])
        if len(selected) == count:
            return selected
    for row in ordered:
        if row in selected:
            continue
        if row["target_id"] in selected_targets and any(
            candidate["target_id"] not in selected_targets for candidate in ordered if candidate not in selected
        ):
            continue
        selected.append(row)
        selected_targets.add(row["target_id"])
        if len(selected) == count:
            return selected
    raise ValueError(f"Could select only {len(selected)} of {count} requested Nr3D rows")


def sr3d_triplet_key(row: dict[str, str]) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        row["scan_id"],
        row["target_id"],
        row["reference_type"],
        tuple(parse_int_list(row["anchor_ids"], "anchor_ids")),
    )


def unique_sr3d_triplets(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str, str, tuple[str, ...]], dict[str, str]] = {}
    for row in rows:
        unique.setdefault(sr3d_triplet_key(row), row)
    return list(unique.values())


def choose_sr3d(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    def stable_key(row: dict[str, str]) -> tuple[int, int, tuple[int, ...], str]:
        relation = row["reference_type"]
        relation_rank = SR3D_RELATION_ORDER.index(relation) if relation in SR3D_RELATION_ORDER else len(SR3D_RELATION_ORDER)
        anchors = tuple(int(item) for item in parse_int_list(row["anchor_ids"], "anchor_ids"))
        return relation_rank, int(row["target_id"]), anchors, row["utterance"]

    ordered = sorted(rows, key=stable_key)
    selected: list[dict[str, str]] = []
    selected_targets: set[str] = set()
    selected_relations: set[str] = set()
    while len(selected) < count:
        best: dict[str, str] | None = None
        best_score: tuple[int, int] | None = None
        for row in ordered:
            if row in selected:
                continue
            score = (
                int(row["reference_type"] not in selected_relations),
                int(row["target_id"] not in selected_targets),
            )
            if best is None or score > best_score:
                best = row
                best_score = score
        if best is None:
            break
        selected.append(best)
        selected_targets.add(best["target_id"])
        selected_relations.add(best["reference_type"])
    if len(selected) != count:
        raise ValueError(f"Could select only {len(selected)} of {count} requested Sr3D+ rows")
    return selected


def query_record(
    *,
    backend_scene_id: str,
    scan_id: str,
    source_dataset: str,
    source_row: dict[str, str],
    box: dict[str, Any],
    view_id: str,
    node_id: str,
    anchor_ids: list[str],
    index: int,
) -> dict[str, Any]:
    target_id = str(box["object_id"])
    expected_box = bbox_for_query(box)
    source_name = "nr3d.csv" if source_dataset == "Nr3D" else "sr3d/sr3d+.csv"
    if source_dataset == "Nr3D":
        if parse_bool(source_row["uses_spatial_lang"]):
            query_type = "relational"
        elif parse_bool(source_row["uses_color_lang"]) or parse_bool(source_row["uses_shape_lang"]):
            query_type = "attribute"
        else:
            query_type = "simple_object"
        source_annotation_id = source_row["assignmentid"]
        relation = None
        language_features = {
            "mentions_target_class": parse_bool(source_row["mentions_target_class"]),
            "uses_object_lang": parse_bool(source_row["uses_object_lang"]),
            "uses_spatial_lang": parse_bool(source_row["uses_spatial_lang"]),
            "uses_color_lang": parse_bool(source_row["uses_color_lang"]),
            "uses_shape_lang": parse_bool(source_row["uses_shape_lang"]),
        }
    else:
        query_type = "relational"
        source_annotation_id = source_row["stimulus_id"]
        relation = source_row["reference_type"]
        language_features = {
            "mentions_target_class": parse_bool(source_row["mentions_target_class"]),
            "coarse_reference_type": source_row["coarse_reference_type"],
            "reference_type": relation,
        }
    prefix = "nr3d" if source_dataset == "Nr3D" else "sr3d_plus"
    return {
        "query_id": f"{backend_scene_id}_{prefix}_{index:03d}",
        "scene_id": backend_scene_id,
        "source_dataset": source_dataset,
        "dataset_scene_id": scan_id,
        "query": source_row["utterance"].strip(),
        "query_type": query_type,
        "expected_output_type": "object_3d_bbox",
        "expected_zone_ids": [],
        "expected_node_ids": [node_id],
        "expected_view_ids": [view_id],
        "expected_object_labels": [expected_box["label"]],
        "expected_object_id": target_id,
        "expected_object_ids": [target_id],
        "expected_anchor_object_ids": anchor_ids,
        "expected_relation": relation,
        "expected_bbox_3d": expected_box,
        "expected_bboxes_3d": [expected_box],
        "acceptance_rule": (
            "found=true, selected object_id equals the official ReferIt3D target_id, "
            "and predicted 3D bbox is scored against the official ScanNet instance AABB"
        ),
        "verification_status": "verified_official_scannet_referit3d_gt",
        "verification_source": [
            f"data/scannet/annotations/{source_name}",
            f"backend/data/scenes/{backend_scene_id}/ground_truth/object_boxes_3d.json",
            f"backend/data/scenes/{backend_scene_id}/ground_truth/object_view_assignments.json",
        ],
        "source_annotation_id": source_annotation_id,
        "source_instance_type": source_row["instance_type"],
        "language_features": language_features,
        "metrics_target": ["Recall@1", "ObjectIDHit", "Acc@0.1", "Acc@0.25", "Acc@0.5"],
        "failure_labels": [
            "wrong_object",
            "wrong_relation",
            "wrong_view",
            "bad_bbox",
            "unavailable_prediction",
        ],
        "gt_3d_reliable": True,
    }


def build_artifacts(
    *,
    nr3d_path: Path,
    sr3d_path: Path,
    scene_root: Path,
    scenes: list[tuple[str, str]] = SCENES,
    per_dataset_per_scene: int = 3,
    expected_nr3d_count: int | None = EXPECTED_NR3D_CORRECT_COUNT,
    expected_sr3d_count: int | None = EXPECTED_SR3D_UNIQUE_TRIPLET_COUNT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    nr3d_all = read_csv(nr3d_path)
    sr3d_all = read_csv(sr3d_path)
    scan_ids = {scan_id for _, scan_id in scenes}
    nr3d = [
        row
        for row in nr3d_all
        if row.get("scan_id") in scan_ids and parse_bool(row.get("correct_guess"))
    ]
    sr3d = unique_sr3d_triplets(row for row in sr3d_all if row.get("scan_id") in scan_ids)
    if expected_nr3d_count is not None and len(nr3d) != expected_nr3d_count:
        raise ValueError(f"Expected {expected_nr3d_count} BBQ-aligned Nr3D rows, found {len(nr3d)}")
    if expected_sr3d_count is not None and len(sr3d) != expected_sr3d_count:
        raise ValueError(f"Expected {expected_sr3d_count} BBQ-aligned Sr3D+ triplets, found {len(sr3d)}")

    nr_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    sr_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nr3d:
        nr_by_scene[row["scan_id"]].append(row)
    for row in sr3d:
        sr_by_scene[row["scan_id"]].append(row)

    queries: list[dict[str, Any]] = []
    missing_targets: list[dict[str, str]] = []
    missing_anchors: list[dict[str, str]] = []
    scene_audit: list[dict[str, Any]] = []
    for backend_scene_id, scan_id in scenes:
        boxes, assignments, nodes = load_scene_gt(backend_scene_id, scene_root)
        all_scene_rows = [("Nr3D", row) for row in nr_by_scene[scan_id]] + [
            ("Sr3D+", row) for row in sr_by_scene[scan_id]
        ]
        for source_dataset, row in all_scene_rows:
            target_id = row["target_id"]
            if target_id not in boxes:
                missing_targets.append({
                    "dataset": source_dataset,
                    "scan_id": scan_id,
                    "target_id": target_id,
                })
            if source_dataset == "Sr3D+":
                for anchor_id in parse_int_list(row["anchor_ids"], "anchor_ids"):
                    if anchor_id not in boxes:
                        missing_anchors.append({
                            "dataset": source_dataset,
                            "scan_id": scan_id,
                            "anchor_id": anchor_id,
                        })
        if missing_targets or missing_anchors:
            continue

        chosen_nr = choose_nr3d(nr_by_scene[scan_id], per_dataset_per_scene)
        chosen_sr = choose_sr3d(sr_by_scene[scan_id], per_dataset_per_scene)
        for index, row in enumerate(chosen_nr, 1):
            target_id = row["target_id"]
            queries.append(query_record(
                backend_scene_id=backend_scene_id,
                scan_id=scan_id,
                source_dataset="Nr3D",
                source_row=row,
                box=boxes[target_id],
                view_id=assignments[target_id],
                node_id=nodes[target_id],
                anchor_ids=[],
                index=index,
            ))
        for index, row in enumerate(chosen_sr, 1):
            target_id = row["target_id"]
            queries.append(query_record(
                backend_scene_id=backend_scene_id,
                scan_id=scan_id,
                source_dataset="Sr3D+",
                source_row=row,
                box=boxes[target_id],
                view_id=assignments[target_id],
                node_id=nodes[target_id],
                anchor_ids=parse_int_list(row["anchor_ids"], "anchor_ids"),
                index=index,
            ))
        scene_audit.append({
            "backend_scene_id": backend_scene_id,
            "scan_id": scan_id,
            "official_object_count": len(boxes),
            "nr3d_correct_guess_count": len(nr_by_scene[scan_id]),
            "sr3d_plus_unique_triplet_count": len(sr_by_scene[scan_id]),
            "selected_nr3d_count": len(chosen_nr),
            "selected_sr3d_plus_count": len(chosen_sr),
        })

    if missing_targets or missing_anchors:
        raise ValueError(
            f"Official annotation IDs failed GT mapping: "
            f"missing targets={len(missing_targets)}, missing anchors={len(missing_anchors)}"
        )
    expected_query_count = len(scenes) * per_dataset_per_scene * 2
    if len(queries) != expected_query_count:
        raise ValueError(f"Expected {expected_query_count} selected pilot queries, built {len(queries)}")
    query_type_counts = Counter(query["query_type"] for query in queries)
    source_counts = Counter(query["source_dataset"] for query in queries)
    generated_at = utc_now()
    benchmark = {
        "schema_version": "semanticsplat.benchmark_queries.v1",
        "benchmark_id": "scannet_bbq_aligned_referit3d_official_gt_pilot_v1",
        "status": "ready_official_scannet_referit3d_gt",
        "status_date": generated_at[:10],
        "scene_scope": [backend_scene_id for backend_scene_id, _ in scenes],
        "selection_protocol": (
            "Deterministic stratification, three Nr3D and three unique Sr3D+ queries per scene. "
            "Nr3D covers spatial+attribute, spatial, and non-spatial language where available; "
            "Sr3D+ greedily diversifies official relation types. No result metric is used for selection."
        ),
        "source_alignment": {
            "bbq_scannet_scene_subset": [scan_id for _, scan_id in scenes],
            "nr3d_correct_guess_count_over_full_subset": len(nr3d),
            "sr3d_plus_unique_triplet_count_over_full_subset": len(sr3d),
            "bbq_expected_nr3d_count": EXPECTED_NR3D_CORRECT_COUNT,
            "bbq_expected_sr3d_plus_unique_triplet_count": EXPECTED_SR3D_UNIQUE_TRIPLET_COUNT,
        },
        "query_count": len(queries),
        "source_query_counts": dict(sorted(source_counts.items())),
        "query_type_counts": dict(sorted(query_type_counts.items())),
        "ground_truth_scope": (
            "Target and anchor IDs come from official ReferIt3D Nr3D/Sr3D+ annotations. "
            "Expected boxes come from official ScanNet aggregation segments over the axis-aligned mesh. "
            "The semantic map also consumes these GT instances, so this is retrieval/grounding over an "
            "oracle GT object map, not semantic perception or predicted-box localization accuracy."
        ),
        "failure_label_contract": {
            "wrong_object": "selected object_id differs from the verified target_id",
            "wrong_relation": "selected target fails the annotated target-anchor relation",
            "wrong_view": "selected representative view differs from the imported assignment",
            "wrong_room_or_zone": "selected room or zone differs from applicable GT",
            "missing_gt": "target, anchor, label, or box cannot be mapped",
            "bad_bbox": "prediction box is missing, invalid, or below the declared IoU threshold",
            "unavailable_prediction": "no executable canonical result was produced",
        },
        "queries": queries,
    }
    manifest = {
        "schema_version": "semanticsplat.scannet_grounding_annotations_manifest.v1",
        "generated_at": generated_at,
        "dataset_license_note": (
            "ScanNet scenes remain subject to ScanNet Terms of Use. ReferIt3D annotations are obtained "
            "from the official ReferIt3D project links; raw data files are intentionally gitignored."
        ),
        "acquisition_command": (
            "python -B scripts/download_scannet_grounding_annotations.py "
            "--install-gdown --acknowledge-source-terms"
        ),
        "local_download_manifest": "data/scannet/annotations/download_manifest.json",
        "sources": {
            "nr3d": annotation_file_record(
                nr3d_path,
                source_url="https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view",
                source_id="1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC",
                row_count=len(nr3d_all),
            ),
            "sr3d_plus": annotation_file_record(
                sr3d_path,
                source_url="https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV",
                source_id="1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV/sr3d+.csv",
                row_count=len(sr3d_all),
            ),
        },
        "bbq_alignment_validation": {
            "nr3d_correct_guess_count": len(nr3d),
            "nr3d_expected_count": EXPECTED_NR3D_CORRECT_COUNT,
            "nr3d_count_matches": len(nr3d) == EXPECTED_NR3D_CORRECT_COUNT,
            "sr3d_plus_unique_triplet_count": len(sr3d),
            "sr3d_plus_expected_unique_triplet_count": EXPECTED_SR3D_UNIQUE_TRIPLET_COUNT,
            "sr3d_plus_count_matches": len(sr3d) == EXPECTED_SR3D_UNIQUE_TRIPLET_COUNT,
            "missing_target_id_count": len(missing_targets),
            "missing_anchor_id_count": len(missing_anchors),
            "all_annotation_ids_mapped_to_official_scannet_gt": not missing_targets and not missing_anchors,
        },
        "pilot": {
            "benchmark_path": "docs/benchmarks/scannet_bbq_grounding_pilot_v1.json",
            "benchmark_file_tracked": False,
            "distribution_note": (
                "The generated benchmark contains licensed ScanNet-derived per-query GT and "
                "remains local; builders and aggregate evidence are tracked."
            ),
            "selection_protocol": benchmark["selection_protocol"],
            "query_count": len(queries),
            "source_query_counts": benchmark["source_query_counts"],
        },
        "scenes": scene_audit,
    }
    return benchmark, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr3d", type=Path, default=ROOT / "data" / "scannet" / "annotations" / "nr3d.csv")
    parser.add_argument(
        "--sr3d-plus",
        type=Path,
        default=ROOT / "data" / "scannet" / "annotations" / "sr3d" / "sr3d+.csv",
    )
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend" / "data" / "scenes")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "scannet_bbq_grounding_pilot_v1.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "docs" / "datasets" / "scannet_grounding_annotations_manifest.json",
    )
    args = parser.parse_args()
    benchmark, manifest = build_artifacts(
        nr3d_path=args.nr3d,
        sr3d_path=args.sr3d_plus,
        scene_root=args.scene_root,
    )
    write_json(args.out, benchmark)
    write_json(args.manifest, manifest)
    print(
        f"Wrote {benchmark['query_count']} queries "
        f"({benchmark['source_query_counts']}) to {args.out}"
    )
    print(f"Wrote annotation provenance and GT audit to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
