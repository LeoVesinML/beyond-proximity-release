import json
from pathlib import Path

import pytest

from scripts.create_manual_viewjson_templates import create_templates


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
    (scene / "transforms.json").write_text(
        json.dumps({"w": 2, "h": 2, "frames": frames}), encoding="utf-8"
    )


def test_create_templates_uses_real_capture_references_and_empty_semantics(tmp_path):
    scene = tmp_path / "captured-scene"
    write_capture_scene(scene)

    result = create_templates(scene, scene / "views")
    first = json.loads((scene / "views" / "v001.json").read_text(encoding="utf-8"))

    assert result["generated_count"] == 2
    assert result["semantic_content_complete"] is False
    assert first["scene_id"] == "captured-scene"
    assert first["image_path"] == "images/v001.png"
    assert first["depth_path"] == "depths/v001_depth.npy"
    assert first["pose_ref"] == "transforms.json#/frames/0/transform_matrix"
    assert first["semantic_index_mode"] == "manual"
    assert first["summary"] == ""
    assert first["visible_objects"] == []
    assert first["visible_regions"] == []
    assert first["landmarks"] == []


def test_create_templates_preserves_existing_annotations_by_default(tmp_path):
    scene = tmp_path / "captured-scene"
    write_capture_scene(scene, count=1)
    create_templates(scene, scene / "views")
    path = scene / "views" / "v001.json"
    annotated = json.loads(path.read_text(encoding="utf-8"))
    annotated["summary"] = "Manual description"
    path.write_text(json.dumps(annotated), encoding="utf-8")

    result = create_templates(scene, scene / "views")

    assert result["generated_count"] == 0
    assert result["skipped_existing_count"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["summary"] == "Manual description"


def test_create_templates_rejects_unindexed_rgb(tmp_path):
    scene = tmp_path / "captured-scene"
    write_capture_scene(scene, count=1)
    (scene / "images" / "orphan.png").write_bytes(b"orphan")

    with pytest.raises(ValueError, match="not represented in transforms.json"):
        create_templates(scene, scene / "views")
