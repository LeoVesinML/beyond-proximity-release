#!/usr/bin/env python3
"""Download official ReferIt3D Nr3D and Sr3D+ annotations from project links."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
NR3D_FILE_ID = "1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC"
SR3D_FOLDER_ID = "1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV"
GDOWN_VERSION = "6.1.0"
REQUIRED_FILES = (
    Path("nr3d.csv"),
    Path("sr3d") / "sr3d+.csv",
    Path("sr3d") / "sr3d.csv",
    Path("sr3d") / "sr3d_train.csv",
    Path("sr3d") / "sr3d_test.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gdown(tool_dir: Path, *, install: bool) -> Any:
    try:
        return importlib.import_module("gdown")
    except ImportError:
        pass
    tool_dir.mkdir(parents=True, exist_ok=True)
    if str(tool_dir) not in sys.path:
        sys.path.insert(0, str(tool_dir))
    try:
        return importlib.import_module("gdown")
    except ImportError:
        if not install:
            raise RuntimeError(
                "gdown is required for the official Google Drive links. Re-run with "
                "--install-gdown or install gdown in the active environment."
            )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            "--target",
            str(tool_dir),
            f"gdown=={GDOWN_VERSION}",
        ],
        check=True,
    )
    importlib.invalidate_caches()
    return importlib.import_module("gdown")


def validate_download(out_dir: Path) -> list[dict[str, Any]]:
    missing = [str(relative) for relative in REQUIRED_FILES if not (out_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing downloaded ReferIt3D files: " + ", ".join(missing))
    return [
        {
            "path": str(relative).replace("\\", "/"),
            "bytes": (out_dir / relative).stat().st_size,
            "sha256": sha256(out_dir / relative),
        }
        for relative in REQUIRED_FILES
    ]


def download_annotations(out_dir: Path, *, install_gdown: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nr3d_path = out_dir / "nr3d.csv"
    sr3d_dir = out_dir / "sr3d"
    if not nr3d_path.is_file() or not all((out_dir / item).is_file() for item in REQUIRED_FILES[1:]):
        gdown = load_gdown(ROOT / "data" / "scannet" / ".tools", install=install_gdown)
        if not nr3d_path.is_file():
            result = gdown.download(id=NR3D_FILE_ID, output=str(nr3d_path), quiet=False, resume=True)
            if result is None:
                raise RuntimeError("Nr3D download did not produce a file")
        if not all((out_dir / item).is_file() for item in REQUIRED_FILES[1:]):
            sr3d_dir.mkdir(parents=True, exist_ok=True)
            result = gdown.download_folder(
                id=SR3D_FOLDER_ID,
                output=str(sr3d_dir),
                quiet=False,
                resume=True,
            )
            if not result:
                raise RuntimeError("Sr3D folder download did not produce files")
    files = validate_download(out_dir)
    manifest = {
        "schema_version": "semanticsplat.referit3d_local_download.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_project": "https://referit3d.github.io/",
        "nr3d_file_id": NR3D_FILE_ID,
        "sr3d_folder_id": SR3D_FOLDER_ID,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    (out_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "scannet" / "annotations",
    )
    parser.add_argument("--install-gdown", action="store_true")
    parser.add_argument(
        "--acknowledge-source-terms",
        action="store_true",
        help="Confirm that the official project source and applicable dataset terms were reviewed.",
    )
    args = parser.parse_args()
    if not args.acknowledge_source_terms:
        parser.error("--acknowledge-source-terms is required")
    manifest = download_annotations(args.out_dir.resolve(), install_gdown=args.install_gdown)
    print(
        f"Verified {manifest['file_count']} ReferIt3D files "
        f"({manifest['total_bytes']} bytes) under {args.out_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
