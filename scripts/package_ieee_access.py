#!/usr/bin/env python3
"""Create a deterministic IEEE Access source/PDF submission bundle."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers" / "ieee-access"
DIST = ROOT / "dist" / "ieee-access"
SOURCE_ZIP = DIST / "Beyond_Proximity_IEEE_Access_source.zip"
PDF_COPY = DIST / "Beyond_Proximity_IEEE_Access.pdf"
MANIFEST = DIST / "Beyond_Proximity_IEEE_Access_submission_manifest.json"

EXCLUDED_NAMES = {
    "BUILD.md",
    "COVER_LETTER.md",
    "SUBMISSION_CHECKLIST.md",
    "main.log",
    "main.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> list[Path]:
    files = []
    for path in PAPER.rglob("*"):
        if path.is_file() and path.name not in EXCLUDED_NAMES:
            files.append(path)
    return sorted(files, key=lambda value: value.relative_to(PAPER).as_posix())


def main() -> int:
    pdf = PAPER / "main.pdf"
    if not pdf.is_file():
        raise FileNotFoundError("compile papers/ieee-access/main.pdf before packaging")

    DIST.mkdir(parents=True, exist_ok=True)
    files = source_files()
    with zipfile.ZipFile(SOURCE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(PAPER).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 8, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())

    shutil.copy2(pdf, PDF_COPY)
    if SOURCE_ZIP.stat().st_size >= 40 * 1024 * 1024:
        raise ValueError("IEEE Access source ZIP exceeds the 40 MB limit")
    if PDF_COPY.stat().st_size >= 40 * 1024 * 1024:
        raise ValueError("IEEE Access PDF exceeds the 40 MB limit")

    manifest = {
        "status": "complete",
        "article_type": "Research Article",
        "title": "Beyond Proximity: Graph-Pruned Semantic Search over 3D Scene Records",
        "source_file_count": len(files),
        "source_zip": {
            "path": str(SOURCE_ZIP.relative_to(ROOT)),
            "bytes": SOURCE_ZIP.stat().st_size,
            "sha256": sha256(SOURCE_ZIP),
        },
        "pdf": {
            "path": str(PDF_COPY.relative_to(ROOT)),
            "bytes": PDF_COPY.stat().st_size,
            "sha256": sha256(PDF_COPY),
        },
        "source_files": [path.relative_to(PAPER).as_posix() for path in files],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"IEEE Access package complete: {SOURCE_ZIP}")
    print(f"IEEE Access PDF copy: {PDF_COPY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
