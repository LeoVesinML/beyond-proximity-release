import csv
import json
from pathlib import Path

import pytest

import scripts.audit_public_dataset_readiness as readiness
from scripts.build_scannet_official_queries import build_artifacts, parse_int_list
from scripts.download_scannet_grounding_annotations import REQUIRED_FILES, validate_download


NR3D_FIELDS = [
    "assignmentid",
    "stimulus_id",
    "utterance",
    "correct_guess",
    "speaker_id",
    "listener_id",
    "scan_id",
    "instance_type",
    "target_id",
    "tokens",
    "dataset",
    "mentions_target_class",
    "uses_object_lang",
    "uses_spatial_lang",
    "uses_color_lang",
    "uses_shape_lang",
]

SR3D_FIELDS = [
    "scan_id",
    "target_id",
    "distractor_ids",
    "utterance",
    "stimulus_id",
    "coarse_reference_type",
    "reference_type",
    "instance_type",
    "anchors_types",
    "anchor_ids",
    "tokens",
    "dataset",
    "mentions_target_class",
    "correct_guess",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_scene(scene_root: Path) -> None:
    scene = scene_root / "scannet_fixture"
    (scene / "ground_truth").mkdir(parents=True)
    (scene / "tree").mkdir()
    objects = []
    assignments = {}
    for object_id, label in enumerate(("chair", "table", "cabinet", "door"), 1):
        objects.append(
            {
                "object_id": str(object_id),
                "label": label,
                "canonical_label": label,
                "center": [float(object_id), 0.0, 0.5],
                "size": [0.5, 0.5, 1.0],
                "coordinate_frame": "scannet_axis_aligned_mesh",
            }
        )
        view_id = f"v{object_id:03d}"
        assignments[str(object_id)] = view_id
        (scene / "tree" / f"node_object_{object_id}.json").write_text(
            json.dumps(
                {
                    "node_id": f"object_{object_id}",
                    "node_type": "object",
                    "bbox_3d": {
                        "object_id": str(object_id),
                        "center": [float(object_id), 0.0, 0.5],
                        "size": [0.5, 0.5, 1.0],
                    },
                }
            ),
            encoding="utf-8",
        )
    (scene / "ground_truth" / "object_boxes_3d.json").write_text(
        json.dumps({"objects": objects}), encoding="utf-8"
    )
    (scene / "ground_truth" / "object_view_assignments.json").write_text(
        json.dumps({"object_to_view": assignments}), encoding="utf-8"
    )


def _nr3d_row(
    assignment: int,
    target_id: int,
    *,
    spatial: bool,
    color: bool = False,
    shape: bool = False,
) -> dict[str, str]:
    label = ("chair", "table", "cabinet", "door")[target_id - 1]
    return {
        "assignmentid": str(assignment),
        "stimulus_id": f"scene_fixture-{label}-{target_id}",
        "utterance": f"find the {label}",
        "correct_guess": "True",
        "speaker_id": "1",
        "listener_id": "2",
        "scan_id": "scene_fixture",
        "instance_type": label,
        "target_id": str(target_id),
        "tokens": f"['find', 'the', '{label}']",
        "dataset": "nr3d",
        "mentions_target_class": "True",
        "uses_object_lang": "True",
        "uses_spatial_lang": str(spatial),
        "uses_color_lang": str(color),
        "uses_shape_lang": str(shape),
    }


def _sr3d_row(target_id: int, relation: str, anchor_id: int, variant: int) -> dict[str, str]:
    label = ("chair", "table", "cabinet", "door")[target_id - 1]
    return {
        "scan_id": "scene_fixture",
        "target_id": str(target_id),
        "distractor_ids": "[]",
        "utterance": f"find the {label} {relation} object {anchor_id} variant {variant}",
        "stimulus_id": f"scene_fixture-{label}-{target_id}-{relation}-{anchor_id}",
        "coarse_reference_type": "horizontal" if relation in {"closest", "farthest"} else "allocentric",
        "reference_type": relation,
        "instance_type": label,
        "anchors_types": "['object']",
        "anchor_ids": f"[{anchor_id}]",
        "tokens": "[]",
        "dataset": "sr3d",
        "mentions_target_class": "True",
        "correct_guess": "True",
    }


def test_builds_deduplicated_gt_mapped_six_query_fixture(tmp_path: Path):
    scene_root = tmp_path / "scenes"
    _write_scene(scene_root)
    nr3d_path = tmp_path / "annotations" / "nr3d.csv"
    sr3d_path = tmp_path / "annotations" / "sr3d+.csv"
    _write_csv(
        nr3d_path,
        NR3D_FIELDS,
        [
            _nr3d_row(1, 1, spatial=True, shape=True),
            _nr3d_row(2, 2, spatial=True),
            _nr3d_row(3, 3, spatial=False, color=True),
            _nr3d_row(4, 4, spatial=False),
        ],
    )
    sr3d_rows = []
    for target_id, relation, anchor_id in (
        (1, "closest", 2),
        (2, "farthest", 3),
        (3, "left", 4),
        (4, "right", 1),
    ):
        sr3d_rows.extend(
            _sr3d_row(target_id, relation, anchor_id, variant)
            for variant in (1, 2)
        )
    _write_csv(sr3d_path, SR3D_FIELDS, sr3d_rows)

    benchmark, manifest = build_artifacts(
        nr3d_path=nr3d_path,
        sr3d_path=sr3d_path,
        scene_root=scene_root,
        scenes=[("scannet_fixture", "scene_fixture")],
        per_dataset_per_scene=3,
        expected_nr3d_count=None,
        expected_sr3d_count=None,
    )

    assert benchmark["query_count"] == 6
    assert benchmark["source_query_counts"] == {"Nr3D": 3, "Sr3D+": 3}
    assert len({query["query_id"] for query in benchmark["queries"]}) == 6
    assert all(query["expected_object_id"] == query["expected_bbox_3d"]["object_id"] for query in benchmark["queries"])
    assert all(query["expected_view_ids"] and query["expected_node_ids"] for query in benchmark["queries"])
    assert manifest["bbq_alignment_validation"]["all_annotation_ids_mapped_to_official_scannet_gt"] is True
    assert manifest["scenes"][0]["sr3d_plus_unique_triplet_count"] == 4


def test_rejects_non_integer_annotation_id_lists():
    with pytest.raises(ValueError, match="integer IDs"):
        parse_int_list("[1, 'two']", "anchor_ids")


def test_grounding_annotation_download_validation_hashes_all_required_files(tmp_path: Path):
    for index, relative in enumerate(REQUIRED_FILES, 1):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode("ascii"))

    rows = validate_download(tmp_path)

    assert [row["path"] for row in rows] == [str(path).replace("\\", "/") for path in REQUIRED_FILES]
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_imported_scannet_readiness_requires_complete_official_gt(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(readiness, "SCANNET_BBQ_BACKEND_SCENES", {"scannet_fixture"})
    scene_root = tmp_path / "backend" / "data" / "scenes"
    scene = scene_root / "scannet_fixture"
    (scene / "ground_truth").mkdir(parents=True)
    (scene / "ground_truth" / "object_boxes_3d.json").write_text(
        json.dumps(
            {
                "source_dataset": "ScanNet",
                "source_scene_id": "scene_fixture",
                "object_count": 2,
                "indexed_object_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (scene / "scene_manifest.json").write_text(
        json.dumps({"source_scene_id": "scene_fixture", "frame_count": 3}),
        encoding="utf-8",
    )
    validation = tmp_path / "docs" / "validation" / "public_datasets" / "scannet_fixture.json"
    validation.parent.mkdir(parents=True)
    validation.write_text(
        json.dumps(
            {
                "semantic_validation": {
                    "semantic_eval_allowed": True,
                    "geometry_eval_allowed": True,
                },
                "geometry_validation": {"three_d_localization_allowed": True},
            }
        ),
        encoding="utf-8",
    )

    rows = readiness.imported_scannet_inventory(scene_root)

    assert len(rows) == 1
    row = rows[0]
    assert row["scene_id"] == "scannet_fixture"
    assert row["path"].replace("\\", "/") == "backend/data/scenes/scannet_fixture"
    assert row["validation_report"].replace("\\", "/") == "docs/validation/public_datasets/scannet_fixture.json"
    assert row["source_scene_id"] == "scene_fixture"
    assert row["frame_count"] == 3
    assert row["object_box_count"] == row["indexed_object_count"] == 2
    assert row["semantic_eval_allowed"] is row["geometry_eval_allowed"] is True
    assert row["status"] == "READY_PUBLIC"


def test_frozen_scannet_pilot_and_manifests_are_consistent():
    root = Path(__file__).resolve().parents[2]
    annotations = json.loads(
        (root / "docs" / "datasets" / "scannet_grounding_annotations_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    availability = json.loads(
        (root / "docs" / "datasets" / "replica_scannet_availability_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads((root / "configs" / "scannet_benchmark.yaml").read_text(encoding="utf-8"))
    person1_pilot = json.loads(
        (root / "docs" / "benchmarks" / "replica_scannet_person1_pilot_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )

    alignment = annotations["bbq_alignment_validation"]
    assert alignment["nr3d_correct_guess_count"] == 699
    assert alignment["sr3d_plus_unique_triplet_count"] == 661
    assert alignment["all_annotation_ids_mapped_to_official_scannet_gt"] is True
    assert len(availability["scannet"]["scenes"]) == 8
    assert all(scene["status"] == "ready" for scene in availability["scannet"]["scenes"])
    assert sum(scene["gt_object_box_count"] for scene in availability["scannet"]["scenes"]) == 392
    assert config["dataset_status"] == "ready"
    assert config["query_count_expected"] == annotations["pilot"]["query_count"] == 48
    assert person1_pilot["scene_scope"] == ["replica_room0", "scannet_0011_00"]
    assert person1_pilot["query_count"] == 20
    assert person1_pilot["source_query_counts"] == {"Nr3D": 7, "Replica": 7, "Sr3D+": 6}
