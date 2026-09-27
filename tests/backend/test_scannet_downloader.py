from pathlib import Path

from scripts.download_scannet_official import (
    BBQ_SCENE_IDS,
    FULL_FILE_TYPES,
    build_jobs,
    download_once,
    file_url,
)
from scripts.audit_public_dataset_readiness import scannet_inventory


def test_bbq_download_plan_is_exact_and_includes_label_map(tmp_path: Path):
    jobs = build_jobs(tmp_path, BBQ_SCENE_IDS, FULL_FILE_TYPES)

    assert len(jobs) == len(BBQ_SCENE_IDS) * len(FULL_FILE_TYPES) + 1
    destinations = [destination for _, destination in jobs]
    for scene_id in BBQ_SCENE_IDS:
        assert tmp_path / "scans" / scene_id / f"{scene_id}.sens" in destinations
        assert tmp_path / "scans" / scene_id / f"{scene_id}_vh_clean_2.labels.ply" in destinations
    assert tmp_path / "tasks" / "scannetv2-labels.combined.tsv" in destinations


def test_sensor_stream_uses_v1_and_annotations_use_v2():
    scene_id = "scene0011_00"

    assert "/v1/scans/" in file_url(scene_id, ".sens")
    assert "/v2/scans/" in file_url(scene_id, ".aggregation.json")


def test_complete_partial_file_is_promoted_without_network(tmp_path: Path):
    destination = tmp_path / "scene0011_00.sens"
    partial = tmp_path / "scene0011_00.sens.part"
    partial.write_bytes(b"complete")

    download_once("https://invalid.example/not-requested", destination, len(b"complete"))

    assert destination.read_bytes() == b"complete"
    assert not partial.exists()


def test_readiness_audit_recognizes_complete_official_raw_layout(tmp_path: Path):
    scene_id = "scene0011_00"
    scene = tmp_path / "scans" / scene_id
    scene.mkdir(parents=True)
    for suffix in FULL_FILE_TYPES:
        (scene / f"{scene_id}{suffix}").write_bytes(b"fixture")

    rows = scannet_inventory(tmp_path)

    assert len(rows) == 1
    assert rows[0]["scene_id"] == scene_id
    assert rows[0]["raw_file_count"] == 6
    assert rows[0]["missing_raw"] == []
    assert rows[0]["status"] == "RAW_READY_FOR_EXTRACTION"


def test_readiness_audit_distinguishes_extracted_layout(tmp_path: Path):
    scene_id = "scene0011_00"
    scene = tmp_path / "scans" / scene_id
    scene.mkdir(parents=True)
    for suffix in FULL_FILE_TYPES:
        (scene / f"{scene_id}{suffix}").write_bytes(b"fixture")
    extracted = tmp_path / "extracted" / scene_id
    for folder in ("color", "depth", "pose", "intrinsic"):
        (extracted / folder).mkdir(parents=True, exist_ok=True)

    rows = scannet_inventory(tmp_path)

    assert rows[0]["missing_required"] == []
    assert rows[0]["status"] == "READY_FOR_CONVERSION"
