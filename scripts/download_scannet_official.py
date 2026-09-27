#!/usr/bin/env python3
"""Download the BBQ-aligned ScanNet subset from the official release server.

This is a Python 3, resumable replacement for the legacy ScanNet
``download-scannet.py`` helper. It intentionally has no "download everything"
mode: callers must select the fixed BBQ subset or explicit scene IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://kaldir.vc.in.tum.de/scannet"
TOS_URL = f"{BASE_URL}/ScanNet_TOS.pdf"
RELEASE_LIST_URL = f"{BASE_URL}/v2/scans.txt"
LABEL_MAP_URL = f"{BASE_URL}/v2/tasks/scannetv2-labels.combined.tsv"

BBQ_SCENE_IDS = (
    "scene0011_00",
    "scene0030_00",
    "scene0046_00",
    "scene0086_00",
    "scene0222_00",
    "scene0378_00",
    "scene0389_00",
    "scene0435_00",
)

GT_FILE_TYPES = (
    ".aggregation.json",
    ".txt",
    "_vh_clean_2.0.010000.segs.json",
    "_vh_clean_2.ply",
    "_vh_clean_2.labels.ply",
)
FULL_FILE_TYPES = GT_FILE_TYPES + (".sens",)
KNOWN_FILE_TYPES = (
    ".aggregation.json",
    ".sens",
    ".txt",
    "_vh_clean.ply",
    "_vh_clean_2.0.010000.segs.json",
    "_vh_clean_2.ply",
    "_vh_clean.segs.json",
    "_vh_clean.aggregation.json",
    "_vh_clean_2.labels.ply",
    "_2d-instance.zip",
    "_2d-instance-filt.zip",
    "_2d-label.zip",
    "_2d-label-filt.zip",
)

CHUNK_SIZE = 4 * 1024 * 1024
USER_AGENT = "SemanticSplat-ScanNet-Downloader/1.0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected official ScanNet v2 scenes without fetching the full 1.2 TB release."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/scannet"),
        help="Dataset root; files are written below <out-dir>/scans (default: data/scannet).",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--bbq-scenes",
        action="store_true",
        help="Download the eight ScanNet scenes used by BBQ.",
    )
    selection.add_argument(
        "--scene-id",
        action="append",
        dest="scene_ids",
        metavar="SCENE_ID",
        help="Download one official ID; repeat for multiple scenes.",
    )
    parser.add_argument(
        "--profile",
        choices=("gt", "full"),
        default="full",
        help="gt downloads mesh/semantic/instance GT; full also downloads RGB-D/pose .sens files.",
    )
    parser.add_argument(
        "--file-type",
        action="append",
        choices=KNOWN_FILE_TYPES,
        help="Override the profile with one or more explicit official file suffixes.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent downloads (default: 4).")
    parser.add_argument("--retries", type=int, default=5, help="Retries per file (default: 5).")
    parser.add_argument(
        "--accept-tos",
        action="store_true",
        help="Confirm that the user has accepted the ScanNet Terms of Use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate scene IDs and print the planned files without downloading them.",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    return args


def read_text(url: str, timeout: int = 60) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def official_scene_ids() -> set[str]:
    return {line.strip() for line in read_text(RELEASE_LIST_URL).splitlines() if line.strip()}


def file_url(scene_id: str, file_type: str) -> str:
    # ScanNet v2 reuses the v1 sensor streams; all other annotations come from v2.
    release = "v1" if file_type == ".sens" else "v2"
    return f"{BASE_URL}/{release}/scans/{scene_id}/{scene_id}{file_type}"


def remote_metadata(url: str, timeout: int = 60) -> dict[str, Any]:
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        size = response.headers.get("Content-Length")
        return {
            "url": response.geturl(),
            "size_bytes": int(size) if size is not None else None,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def human_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def download_once(url: str, destination: Path, expected_size: int | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if destination.exists():
        local_size = destination.stat().st_size
        if expected_size is None or local_size == expected_size:
            print(f"SKIP {destination} ({human_size(local_size)})")
            return
        partial.unlink(missing_ok=True)
        destination.replace(partial)

    if partial.exists() and expected_size is not None:
        partial_size = partial.stat().st_size
        if partial_size == expected_size:
            os.replace(partial, destination)
            print(f"DONE {destination} ({human_size(partial_size)}, recovered from complete .part)")
            return
        if partial_size > expected_size:
            partial.unlink()

    curl = shutil.which("curl")
    if curl:
        offset = partial.stat().st_size if partial.exists() else 0
        print(f"GET  {url} -> {destination} (resume={human_size(offset)}, transport=curl)")
        completed = subprocess.run(
            [
                curl,
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--continue-at",
                "-",
                "--connect-timeout",
                "30",
                "--speed-time",
                "60",
                "--speed-limit",
                "1024",
                "--output",
                str(partial),
                url,
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise OSError(f"curl exited with code {completed.returncode}")
        final_size = partial.stat().st_size
        if expected_size is not None and final_size != expected_size:
            raise OSError(
                f"Incomplete download for {destination}: expected {expected_size} bytes, got {final_size}"
            )
        os.replace(partial, destination)
        return

    # Standard-library fallback for systems without curl.
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)

    with urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            partial.unlink(missing_ok=True)
            offset = 0
        mode = "ab" if offset and status == 206 else "wb"
        downloaded = offset
        report_step = max((expected_size or 0) // 20, 64 * 1024 * 1024)
        next_report = downloaded + report_step
        print(f"GET  {url} -> {destination} (resume={human_size(offset)})")
        with partial.open(mode) as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(
                        f"     {destination.name}: {human_size(downloaded)}"
                        + (f" / {human_size(expected_size)}" if expected_size else "")
                    )
                    next_report = downloaded + report_step

    final_size = partial.stat().st_size
    if expected_size is not None and final_size != expected_size:
        raise OSError(
            f"Incomplete download for {destination}: expected {expected_size} bytes, got {final_size}"
        )
    os.replace(partial, destination)


def download_with_retries(
    url: str,
    destination: Path,
    retries: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] | None = None
    for attempt in range(1, retries + 1):
        try:
            metadata = remote_metadata(url)
            download_once(url, destination, metadata["size_bytes"])
            local_size = destination.stat().st_size
            return {
                **metadata,
                "path": destination.as_posix(),
                "local_size_bytes": local_size,
                "status": "complete",
            }
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed after {retries} attempts: {url}: {exc}") from exc
            delay = min(2 ** (attempt - 1), 30)
            print(f"RETRY {attempt}/{retries} {url}: {exc}; waiting {delay}s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def confirm_tos(accepted: bool) -> bool:
    if accepted:
        return True
    print("ScanNet data is licensed under its Terms of Use:")
    print(TOS_URL)
    answer = input("Type YES only if you have accepted those terms: ").strip()
    return answer == "YES"


def build_jobs(
    out_dir: Path,
    scene_ids: tuple[str, ...],
    file_types: tuple[str, ...],
) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for scene_id in scene_ids:
        for file_type in file_types:
            jobs.append(
                (
                    file_url(scene_id, file_type),
                    out_dir / "scans" / scene_id / f"{scene_id}{file_type}",
                )
            )
    jobs.append((LABEL_MAP_URL, out_dir / "tasks" / "scannetv2-labels.combined.tsv"))
    return jobs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not confirm_tos(args.accept_tos):
        print("Terms were not confirmed; no files were downloaded.", file=sys.stderr)
        return 2

    scene_ids = BBQ_SCENE_IDS if args.bbq_scenes else tuple(args.scene_ids)
    known_scenes = official_scene_ids()
    unknown = sorted(set(scene_ids) - known_scenes)
    if unknown:
        print(f"Unknown ScanNet v2 scene IDs: {', '.join(unknown)}", file=sys.stderr)
        return 2

    if args.file_type:
        file_types = tuple(dict.fromkeys(args.file_type))
    else:
        file_types = GT_FILE_TYPES if args.profile == "gt" else FULL_FILE_TYPES
    jobs = build_jobs(args.out_dir, scene_ids, file_types)

    print(f"Scenes: {', '.join(scene_ids)}")
    print(f"File types: {', '.join(file_types)} plus the ScanNet v2 label map")
    print(f"Files: {len(jobs)}")
    if args.dry_run:
        for url, destination in jobs:
            print(f"{url} -> {destination}")
        return 0

    started_at = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_with_retries, url, destination, args.retries): url
            for url, destination in jobs
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # keep other resumable downloads running
                message = f"{url}: {exc}"
                failures.append(message)
                print(f"ERROR {message}", file=sys.stderr)

    records.sort(key=lambda item: item["path"])
    manifest = {
        "schema_version": "semanticsplat.scannet_acquisition.v1",
        "source": "official ScanNet release server",
        "release": "v2 annotations with v1 sensor streams as specified by the official downloader",
        "terms_url": TOS_URL,
        "scene_ids": list(scene_ids),
        "file_types": list(file_types),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not failures else "partial",
        "file_count": len(records),
        "total_bytes": sum(record["local_size_bytes"] for record in records),
        "files": records,
        "failures": failures,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Download status: {manifest['status']}; {len(records)}/{len(jobs)} files; "
        f"{human_size(manifest['total_bytes'])}"
    )
    print(f"Manifest: {manifest_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
