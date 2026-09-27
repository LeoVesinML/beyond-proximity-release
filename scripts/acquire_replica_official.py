#!/usr/bin/env python3
"""Download and extract the official Replica v1 dataset release."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RELEASE_API = "https://api.github.com/repos/facebookresearch/Replica-Dataset/releases/tags/v1.0"
ASSET_PREFIX = "replica_v1_0.tar.gz.part"
DEFAULT_DOWNLOAD_DIR = ROOT / "data" / "replica_official_downloads"
DEFAULT_EXTRACT_DIR = ROOT / "data" / "replica_official_raw"
DEFAULT_MANIFEST = ROOT / "docs" / "datasets" / "replica_official_acquisition_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_release_assets() -> list[dict[str, Any]]:
    with urllib.request.urlopen(RELEASE_API, timeout=60) as response:
        release = json.loads(response.read().decode("utf-8"))
    assets = [
        asset
        for asset in release.get("assets", [])
        if isinstance(asset, dict) and str(asset.get("name", "")).startswith(ASSET_PREFIX)
    ]
    assets.sort(key=lambda asset: str(asset["name"]))
    if not assets:
        raise RuntimeError("Replica release did not expose archive part assets")
    return assets


def run_curl(url: str, destination: Path) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl/curl.exe is required for resumable downloads")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        curl,
        "--silent",
        "--show-error",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--output",
        str(destination),
        url,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def part_status(asset: dict[str, Any], path: Path) -> dict[str, Any]:
    expected_size = int(asset.get("size") or 0)
    actual_size = path.stat().st_size if path.exists() else 0
    return {
        "name": asset["name"],
        "path": str(path.relative_to(ROOT)),
        "url": asset["browser_download_url"],
        "expected_size": expected_size,
        "actual_size": actual_size,
        "complete": bool(expected_size and actual_size == expected_size),
    }


def download_one(asset: dict[str, Any], download_dir: Path) -> dict[str, Any]:
    name = str(asset["name"])
    destination = download_dir / name
    status = part_status(asset, destination)
    if not status["complete"]:
        print(f"downloading {name}", flush=True)
        run_curl(str(asset["browser_download_url"]), destination)
    return part_status(asset, destination)


def download_parts(
    assets: list[dict[str, Any]],
    download_dir: Path,
    *,
    limit: int | None,
    workers: int,
) -> list[dict[str, Any]]:
    selected = assets[:limit] if limit else assets
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {
            executor.submit(download_one, asset, download_dir): str(asset["name"])
            for asset in selected
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                rows.append(future.result())
                print(f"complete {name}", flush=True)
            except Exception as exc:  # noqa: BLE001 - preserve every other completed part
                rows.append(
                    {
                        "name": name,
                        "path": str((download_dir / name).relative_to(ROOT)),
                        "url": next(
                            str(asset["browser_download_url"])
                            for asset in selected
                            if str(asset["name"]) == name
                        ),
                        "expected_size": next(
                            int(asset.get("size") or 0)
                            for asset in selected
                            if str(asset["name"]) == name
                        ),
                        "actual_size": (download_dir / name).stat().st_size
                        if (download_dir / name).exists()
                        else 0,
                        "complete": False,
                        "error": str(exc),
                    }
                )
                print(f"failed {name}: {exc}", flush=True)
    rows.sort(key=lambda row: str(row["name"]))
    return rows


def extract_archive(parts: list[Path], extract_dir: Path) -> None:
    tar = shutil.which("tar")
    if not tar:
        raise RuntimeError("tar is required to extract Replica archive parts")
    extract_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [tar, "-xzf", "-", "-C", str(extract_dir)],
        cwd=ROOT,
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for part in parts:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, process.stdin, length=8 * 1024 * 1024)
        process.stdin.close()
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
    if return_code != 0:
        raise RuntimeError(f"tar extraction failed with exit code {return_code}")


def scene_status(extract_dir: Path, scene_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for scene_id in scene_ids:
        raw_scene_id = scene_id
        if not (extract_dir / raw_scene_id).exists():
            match = __import__("re").fullmatch(r"([A-Za-z_]+?)([0-9]+)", scene_id)
            if match:
                raw_scene_id = f"{match.group(1)}_{match.group(2)}"
        scene_dir = extract_dir / raw_scene_id
        habitat_dir = scene_dir / "habitat"
        rows.append(
            {
                "scene_id": scene_id,
                "raw_scene_id": raw_scene_id,
                "path": str(scene_dir.relative_to(ROOT)) if scene_dir.exists() else str(scene_dir),
                "scene_dir_exists": scene_dir.is_dir(),
                "mesh_semantic_ply": (habitat_dir / "mesh_semantic.ply").is_file(),
                "info_semantic_json": (habitat_dir / "info_semantic.json").is_file(),
                "raw_mesh_ply": (scene_dir / "mesh.ply").is_file(),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--limit-parts", type=int, help="Debug only: download the first N archive parts")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene to check after extraction; can be passed more than once.",
    )
    args = parser.parse_args()

    if args.limit_parts is not None and args.limit_parts < 1:
        raise SystemExit("--limit-parts must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    assets = load_release_assets()
    expected_total = sum(int(asset.get("size") or 0) for asset in assets)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.extract_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_download:
        parts = [part_status(asset, args.download_dir / str(asset["name"])) for asset in assets]
    else:
        parts = download_parts(assets, args.download_dir, limit=args.limit_parts, workers=args.workers)

    complete_parts = [row for row in parts if row["complete"]]
    all_parts_complete = len(complete_parts) == len(assets)
    extracted = False
    extraction_error = None
    if not args.skip_extract and all_parts_complete:
        try:
            extract_archive([args.download_dir / str(asset["name"]) for asset in assets], args.extract_dir)
            extracted = True
        except Exception as exc:  # noqa: BLE001 - evidence manifest should capture exact blocker
            extraction_error = str(exc)
    elif not args.skip_extract:
        extraction_error = "Extraction skipped because not every archive part is complete."

    scene_ids = args.scene or [
        "room0",
        "room1",
        "room2",
        "office0",
        "office1",
        "office2",
        "office3",
        "office4",
    ]
    manifest = {
        "schema_version": "semanticsplat.replica_official_acquisition.v1",
        "generated_at": utc_now(),
        "source": {
            "release_api": RELEASE_API,
            "release_tag": "v1.0",
            "official_repository": "https://github.com/facebookresearch/Replica-Dataset",
            "license": "Replica Dataset Research Terms",
        },
        "download_dir": str(args.download_dir.relative_to(ROOT)),
        "extract_dir": str(args.extract_dir.relative_to(ROOT)),
        "expected_asset_count": len(assets),
        "expected_total_size_bytes": expected_total,
        "downloaded_asset_count": len(complete_parts),
        "download_complete": all_parts_complete,
        "extracted": extracted,
        "extraction_error": extraction_error,
        "parts": parts,
        "scene_status": scene_status(args.extract_dir, scene_ids),
    }
    write_json(args.manifest, manifest)
    print(f"Wrote {args.manifest}")
    if not all_parts_complete:
        return 2
    if extraction_error:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
