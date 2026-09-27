from pathlib import Path

import numpy as np
import pytest

from scripts.run_raw_rgbd_hierarchy_benchmark import (
    deterministic_kmeans,
    load_raw_scene,
    pairwise_zone_metrics,
    select_cluster_count,
)


ROOT = Path(__file__).resolve().parents[2]


def test_deterministic_kmeans_repeats() -> None:
    features = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [10.0, 0.0], [10.1, 0.0]]
    )
    first_labels, first_centers, _ = deterministic_kmeans(features, 3)
    second_labels, second_centers, _ = deterministic_kmeans(features, 3)
    assert np.array_equal(first_labels, second_labels)
    assert np.allclose(first_centers, second_centers)
    selected, candidates = select_cluster_count(features)
    assert 2 <= selected <= 3
    assert candidates


def test_raw_scene_loader_needs_no_viewjson(monkeypatch) -> None:
    scene = ROOT / "backend" / "data" / "scenes" / "ConferenceHall-capture-pilot"
    if len(list((scene / "images").glob("*.png"))) < 20 or len(
        list((scene / "depths").glob("*.npy"))
    ) < 20:
        pytest.skip("raw RGB-D scene assets are not distributed with the public code release")
    original = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        normalized = str(path).replace("\\", "/")
        assert "/views/" not in normalized
        assert "manual_zone_gt" not in normalized
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    records = load_raw_scene(scene)
    assert len(records) == 20
    assert all(record["image_path"].is_file() for record in records)
    assert all(record["depth_path"].is_file() for record in records)


def test_pairwise_zone_metrics_perfect_match() -> None:
    zones = [
        {"cluster_id": "a", "view_ids": ["v1", "v2"]},
        {"cluster_id": "b", "view_ids": ["v3", "v4"]},
    ]
    metrics = pairwise_zone_metrics(zones, zones, ["v1", "v2", "v3", "v4"])
    assert metrics["pairwise_f1"] == 1.0
    assert metrics["rand_index"] == 1.0
