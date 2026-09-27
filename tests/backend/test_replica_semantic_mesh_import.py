import json
from pathlib import Path

from scripts.import_replica_semantic_mesh import import_replica_scene


def write_tiny_replica_scene(scene: Path) -> None:
    habitat = scene / "habitat"
    habitat.mkdir(parents=True)
    (habitat / "mesh_semantic.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property int object_id",
                "end_header",
                "0 0 0 1",
                "1 1 1 1",
                "2 2 2 2",
                "3 3 4 2",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    (habitat / "info_semantic.json").write_text(
        json.dumps(
            {
                "classes": [
                    {"id": 10, "name": "chair"},
                    {"id": 20, "name": "table"},
                ],
                "objects": [
                    {"id": 1, "class_id": 10},
                    {"id": 2, "class_id": 20},
                ],
            }
        ),
        encoding="utf-8",
    )


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_import_replica_semantic_mesh_creates_official_gt_scene(tmp_path):
    source = tmp_path / "room0"
    output = tmp_path / "replica_room0"
    write_tiny_replica_scene(source)

    result = import_replica_scene(
        source,
        output,
        scene_id="replica_room0",
        source_scene_id="room0",
        copy_mesh=False,
        overwrite=False,
        include_unlabeled=False,
    )

    boxes = load(output / "ground_truth" / "object_boxes_3d.json")
    view = load(output / "views" / "v001.json")
    tree_manifest = load(output / "tree" / "manifest.json")

    assert result["object_count"] == 2
    assert boxes["source_dataset"] == "Replica"
    assert boxes["object_count"] == 2
    assert {item["label"] for item in boxes["objects"]} == {"chair", "table"}
    assert view["semantic_index_mode"] == "official_gt"
    assert {obj["source"] for obj in view["visible_objects"]} == {"official_gt"}
    assert tree_manifest["complete"] is True
    assert result["semantic_validation"]["semantic_eval_allowed"] is True
    assert result["semantic_validation"]["geometry_eval_allowed"] is True
