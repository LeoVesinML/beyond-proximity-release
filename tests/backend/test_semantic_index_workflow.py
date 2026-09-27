import json
from pathlib import Path

import pytest

from scripts.build_scene_tree_from_viewjson import build_scene_tree
from scripts.create_manual_viewjson_templates import create_templates
from scripts.validate_semantic_index import validate_semantic_index


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def write_capture_scene(scene: Path, count: int = 2) -> None:
    frames = []
    for index in range(1, count + 1):
        view_id = f"v{index:03d}"
        image_path = scene / "images" / f"{view_id}.png"
        depth_path = scene / "depths" / f"{view_id}_depth.npy"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"captured-rgb")
        depth_path.write_bytes(b"captured-depth")
        frames.append(
            {
                "view_id": view_id,
                "file_path": f"images/{view_id}.png",
                "depth_file_path": f"depths/{view_id}_depth.npy",
                "transform_matrix": IDENTITY,
            }
        )
    scene.mkdir(parents=True, exist_ok=True)
    (scene / "transforms.json").write_text(
        json.dumps({"w": 2, "h": 2, "frames": frames}), encoding="utf-8"
    )
    create_templates(scene, scene / "views")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_empty_templates_build_incomplete_root_only_tree(tmp_path):
    scene = tmp_path / "captured-scene"
    write_capture_scene(scene)

    manifest = build_scene_tree(scene, scene / "views", scene / "tree")
    report = validate_semantic_index(scene)

    assert manifest["node_ids"] == ["root"]
    assert manifest["node_count"] == 1
    assert manifest["semantic_item_count"] == 0
    assert manifest["complete"] is False
    assert (scene / "tree" / "node_root.json").exists()
    assert report["valid_view_json_count"] == 2
    assert report["tree_manifest_exists"] is True
    assert report["root_node_exists"] is True
    assert report["tree_valid"] is False
    assert report["semantic_eval_allowed"] is False
    assert report["geometry_eval_allowed"] is False


def test_filled_manual_views_build_real_nodes_and_are_runner_compatible(tmp_path):
    scene = tmp_path / "captured-scene"
    write_capture_scene(scene)
    first_path = scene / "views" / "v001.json"
    second_path = scene / "views" / "v002.json"
    first = load(first_path)
    first.update(
        {
            "summary": "Conference table with a black chair and an exit sign.",
            "visible_regions": [
                {
                    "label": "conference table area",
                    "approx_location": "center",
                    "source": "manual",
                }
            ],
            "visible_objects": [
                {
                    "label": "chair",
                    "attributes": ["black"],
                    "approx_location": "left of the table",
                    "confidence": None,
                    "bbox_2d": None,
                    "bbox_3d": None,
                    "source": "manual",
                }
            ],
            "landmarks": [
                {
                    "label": "exit sign",
                    "kind": "sign",
                    "approx_location": "above the doorway",
                    "source": "manual",
                }
            ],
            "warnings": [],
        }
    )
    second = load(second_path)
    second.update({"summary": "Opposite view of the same room.", "warnings": []})
    save(first_path, first)
    save(second_path, second)

    manifest = build_scene_tree(scene, scene / "views", scene / "tree")
    report = validate_semantic_index(scene)
    node_types = {
        load(path)["node_type"] for path in (scene / "tree").glob("node_*.json")
    }

    assert manifest["complete"] is True
    assert manifest["semantic_item_count"] == 3
    assert {"root", "region", "object", "sign"}.issubset(node_types)
    assert report["visible_region_count"] == 1
    assert report["visible_object_count"] == 1
    assert report["sign_count"] == 1
    assert report["tree_valid"] is True
    assert report["query_runner_loadable"] is True
    assert report["query_runner_schema_compatible"] is True
    assert report["semantic_eval_allowed"] is True


def test_manual_confidence_is_rejected_instead_of_treated_as_real(tmp_path):
    scene = tmp_path / "captured-scene"
    write_capture_scene(scene, count=1)
    path = scene / "views" / "v001.json"
    data = load(path)
    data.update(
        {
            "summary": "A visible chair.",
            "visible_objects": [
                {
                    "label": "chair",
                    "attributes": [],
                    "approx_location": "center",
                    "confidence": 0.9,
                    "bbox_2d": None,
                    "bbox_3d": None,
                    "source": "manual",
                }
            ],
        }
    )
    save(path, data)

    with pytest.raises(ValueError, match="confidence must be null for manual annotations"):
        build_scene_tree(scene, scene / "views", scene / "tree")
