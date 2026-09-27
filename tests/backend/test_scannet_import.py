import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_scannet_scene import convert_ground_truth, import_scene
from scripts.scannet_sensor_data import SensorDataIndex


IDENTITY = np.eye(4, dtype=np.float32)


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    return np.asarray(matrix, dtype="<f4").tobytes()


def _write_sens(path: Path) -> None:
    image = Image.new("RGB", (2, 2), color=(40, 80, 120))
    color_buffer = io.BytesIO()
    image.save(color_buffer, format="JPEG")
    color = color_buffer.getvalue()
    depth_values = np.asarray([[1000, 2000], [1500, 0]], dtype="<u2")
    depth = zlib.compress(depth_values.tobytes())
    name = b"fixture"
    payload = bytearray()
    payload += struct.pack("<I", 4)
    payload += struct.pack("<Q", len(name)) + name
    intrinsic = IDENTITY.copy()
    intrinsic[0, 0] = 1.0
    intrinsic[1, 1] = 1.0
    intrinsic[0, 2] = 0.5
    intrinsic[1, 2] = 0.5
    for matrix in (intrinsic, IDENTITY, intrinsic, IDENTITY):
        payload += _matrix_bytes(matrix)
    payload += struct.pack("<ii", 2, 1)
    payload += struct.pack("<IIII", 2, 2, 2, 2)
    payload += struct.pack("<f", 1000.0)
    payload += struct.pack("<Q", 1)
    payload += _matrix_bytes(IDENTITY)
    payload += struct.pack("<QQQQ", 10, 11, len(color), len(depth))
    payload += color + depth
    path.write_bytes(payload)


def _write_ply(path: Path, points: np.ndarray, labels: np.ndarray | None = None) -> None:
    properties = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar alpha",
    ]
    dtype = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("alpha", "u1"),
    ]
    if labels is not None:
        properties.append("property ushort label")
        dtype.append(("label", "<u2"))
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(points)}",
            *properties,
            "end_header",
            "",
        ]
    ).encode("ascii")
    records = np.zeros(len(points), dtype=np.dtype(dtype))
    records["x"], records["y"], records["z"] = points.T
    records["red"] = 50
    records["green"] = 100
    records["blue"] = 150
    records["alpha"] = 255
    if labels is not None:
        records["label"] = labels
    path.write_bytes(header + records.tobytes())


def _write_raw_scene(root: Path) -> tuple[Path, Path]:
    scene = root / "scene_fixture"
    scene.mkdir(parents=True)
    points = np.asarray(
        [
            [-0.2, -0.2, 2.0],
            [0.2, 0.2, 2.5],
            [-0.3, 0.1, 3.0],
            [0.3, 0.4, 3.5],
        ],
        dtype=np.float32,
    )
    _write_ply(scene / "scene_fixture_vh_clean_2.ply", points)
    _write_ply(
        scene / "scene_fixture_vh_clean_2.labels.ply",
        points,
        np.asarray([1, 1, 2, 2], dtype=np.uint16),
    )
    (scene / "scene_fixture_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": [10, 10, 20, 20]}), encoding="utf-8"
    )
    (scene / "scene_fixture.aggregation.json").write_text(
        json.dumps(
            {
                "segGroups": [
                    {"id": 0, "objectId": 0, "segments": [10], "label": "chair"},
                    {"id": 1, "objectId": 1, "segments": [20], "label": "table"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (scene / "scene_fixture.txt").write_text(
        "axisAlignment = " + " ".join(str(value) for value in IDENTITY.reshape(-1)) + "\n",
        encoding="utf-8",
    )
    _write_sens(scene / "scene_fixture.sens")
    label_map = root / "scannetv2-labels.combined.tsv"
    label_map.write_text(
        "id\traw_category\tcategory\tnyu40id\tnyu40class\n"
        "1\tchair\tchair\t5\tchair\n"
        "2\ttable\ttable\t7\ttable\n",
        encoding="utf-8",
    )
    return scene, label_map


def test_sensor_index_decodes_jpeg_and_zlib_depth(tmp_path: Path):
    sens = tmp_path / "fixture.sens"
    _write_sens(sens)

    index = SensorDataIndex.scan(sens)
    decoded = index.decode(0)

    assert index.header.version == 4
    assert index.header.frame_count == 1
    assert decoded.color.size == (2, 2)
    assert decoded.depth_meters.shape == (2, 2)
    assert decoded.depth_meters.tolist() == [[1.0, 2.0], [1.5, 0.0]]
    assert np.allclose(decoded.camera_to_world, IDENTITY)


def test_ground_truth_conversion_builds_instance_boxes_and_semantic_summary(tmp_path: Path):
    scene, label_map = _write_raw_scene(tmp_path)

    boxes, samples, metadata, alignment = convert_ground_truth(scene, label_map)

    assert np.allclose(alignment, IDENTITY)
    assert [box["object_id"] for box in boxes] == ["0", "1"]
    assert boxes[0]["label"] == "chair"
    assert boxes[0]["class_id"] == 5
    assert boxes[0]["center"] == [0.0, 0.0, 2.25]
    assert samples["1"].shape == (2, 3)
    assert metadata["semantic_segmentation"]["vertex_count"] == 4
    assert metadata["semantic_segmentation"]["class_count"] == 2


def test_full_scannet_import_produces_valid_rgbd_gt_scene(tmp_path: Path):
    raw_scene, label_map = _write_raw_scene(tmp_path / "raw")
    extracted = tmp_path / "extracted" / "scene_fixture"
    output = tmp_path / "backend" / "scannet_fixture"

    report = import_scene(
        raw_scene,
        extracted,
        output,
        label_map,
        backend_scene_id="scannet_fixture",
        max_views=1,
        candidate_stride=1,
        overwrite=False,
    )

    assert report["official_object_count"] == 2
    assert report["indexed_object_count"] == 2
    assert report["object_coverage_ratio"] == 1.0
    assert report["semantic_validation"]["semantic_eval_allowed"] is True
    assert report["semantic_validation"]["geometry_eval_allowed"] is True
    assert report["geometry_validation"]["geometry_inputs_valid"] is True
    assert report["geometry_validation"]["three_d_localization_allowed"] is True
    assert (output / "images" / "v001.jpg").is_file()
    assert (output / "depths" / "v001_depth.npy").is_file()
    assert (output / "ground_truth" / "object_boxes_3d.json").is_file()
