#!/usr/bin/env python3
"""Create a venue-neutral, dataset-safe public repository export."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "dist" / "public-release"
REPOSITORY = OUTPUT_ROOT / "repository"
ARCHIVE = OUTPUT_ROOT / "beyond-proximity-public-release.zip"

ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "COMMANDS.md",
    "README.md",
    "start_all.cmd",
    "start_all.sh",
    "start_backend.cmd",
    "start_backend.ps1",
    "start_frontend.cmd",
    "start_frontend.ps1",
    "start_mcp.cmd",
    "start_mcp.ps1",
}

GENERAL_PREFIXES = (
    ".github/workflows/",
    "backend/",
    "baselines/",
    "configs/",
    "data/",
    "frontend/",
    "notebooks/",
    "outputs/",
    "papers/beyond-proximity/",
    "scripts/",
    "site/",
    "tests/",
)

DOC_PREFIXES = (
    "docs/assets/",
    "docs/baselines/",
    "docs/benchmarks/",
    "docs/datasets/",
    "docs/experiments/",
    "docs/project/PUBLIC_CODE_RELEASE.md",
    "docs/schemas/",
    "docs/validation/",
)

REQUIRED_REPORTS = {
    "docs/reports/final/graph_construction_cost.json",
}

CAPTURE_SCENES = {
    "ConferenceHall-capture-pilot",
    "Museume-capture",
    "Theater-capture",
    "outdoor-street-capture",
    "outdoor-drone-capture",
    "default",
}

TEXT_SCENE_SUFFIXES = {".json", ".md", ".csv", ".txt"}
OUTPUT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".yaml", ".yml"}
RAW_SUFFIXES = {".npy", ".npz", ".ply", ".sens", ".pth", ".pt", ".ckpt", ".onnx"}
MAX_FILE_BYTES = 5 * 1024 * 1024

REVIEW_PATH_MARKERS = (
    "twinworld",
    "openreview",
    "publication_venue",
    "article_compliance_audit",
)

EXPLICIT_REVIEW_PHRASES = (
    "under review at eccv",
    "submitted to eccv",
    "eccv 2026 workshop",
    "twinworld submission",
    "twinworld manuscript",
    "papers/twinworld",
    "check_twinworld",
    "export_twinworld",
)

SECRET_PATTERNS = {
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8", errors="surrogateescape"))
        if (ROOT / relative).is_file():
            paths.append(relative)
    return sorted(set(paths), key=lambda path: path.as_posix())


def scene_file_allowed(path: Path) -> bool:
    parts = path.as_posix().split("/")
    if len(parts) < 5:
        return False
    scene_id = parts[3]
    if scene_id not in CAPTURE_SCENES:
        return False
    if path.as_posix() == "backend/data/scenes/ConferenceHall-capture-pilot/images/v018.png":
        return True
    return path.suffix.lower() in TEXT_SCENE_SUFFIXES


def allow_path(path: Path) -> tuple[bool, str]:
    normalized = path.as_posix()
    lower = normalized.lower()
    if any(marker in lower for marker in REVIEW_PATH_MARKERS):
        return False, "review_path"
    if path.name in {"AGENTS.md", "PROJECT_REPORT.pdf"}:
        return False, "private_or_aggregate"
    if path.suffix.lower() in RAW_SUFFIXES:
        return False, "raw_or_model_binary"
    if "/__pycache__/" in lower or "/node_modules/" in lower or "/dist/" in lower:
        return False, "generated_dependency"
    if normalized in ROOT_FILES:
        return True, "root"
    if normalized.startswith("backend/data/scenes/"):
        return (True, "captured_scene_record") if scene_file_allowed(path) else (False, "scene_data")
    if normalized.startswith("frontend/dist/"):
        return False, "generated_frontend"
    if normalized.startswith("docs/benchmarks/") and path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
    }:
        return False, "derived_preview_image"
    if normalized.startswith("papers/beyond-proximity/presentation/public/images/"):
        return False, "generated_presentation_image"
    if normalized.startswith("outputs/") and path.suffix.lower() not in OUTPUT_SUFFIXES:
        return False, "noncanonical_output"
    if normalized.startswith("docs/"):
        return (
            (True, "documentation")
            if normalized.startswith(DOC_PREFIXES) or normalized in REQUIRED_REPORTS
            else (False, "private_documentation")
        )
    if normalized.startswith(GENERAL_PREFIXES):
        return True, "implementation_or_evidence"
    return False, "not_allowlisted"


def decode_text(path: Path) -> str | None:
    if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico"}:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def content_is_safe(path: Path) -> tuple[bool, str]:
    text = decode_text(path)
    if text is None:
        return True, "binary"
    lower = text.lower()
    if any(phrase in lower for phrase in EXPLICIT_REVIEW_PHRASES):
        return False, "explicit_review_wording"
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            return False, f"secret:{name}"
    return True, "text"


def deterministic_zip(source_dir: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 8, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def build() -> dict:
    if REPOSITORY.exists():
        shutil.rmtree(REPOSITORY)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPOSITORY.mkdir(parents=True)

    excluded: Counter[str] = Counter()
    included: list[str] = []
    for relative in repository_files():
        source = ROOT / relative
        allowed, reason = allow_path(relative)
        if not allowed:
            excluded[reason] += 1
            continue
        if source.stat().st_size > MAX_FILE_BYTES:
            excluded["over_size_limit"] += 1
            continue
        safe, content_reason = content_is_safe(source)
        if not safe:
            excluded[content_reason] += 1
            continue
        target = REPOSITORY / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        included.append(relative.as_posix())

    required = (
        "README.md",
        "CITATION.cff",
        "backend/query/grounding.py",
        "frontend/package.json",
        "papers/beyond-proximity/main.pdf",
        "papers/beyond-proximity/main.tex",
        "scripts/check_academic_paper.py",
        "scripts/build_project_site.py",
        "site/index.html",
        "tests/backend/test_grounding_search.py",
    )
    missing = [path for path in required if not (REPOSITORY / path).is_file()]
    if missing:
        raise RuntimeError(f"public release is missing required files: {missing}")

    notice = """# Data and Licensing Notice

This source release does not redistribute raw ScanNet or Replica data, model
checkpoints, third-party paper PDFs, or the five captured RGB-D sequences.
Obtain public datasets from their official providers and follow their terms.
Frozen JSON scene records, query labels, aggregate metrics, and validation
artifacts are included for reproducibility. One team-owned image is included as
project-page media.

No code license has yet been selected by the author team. Public visibility
does not grant permission to copy, modify, or redistribute the source.
"""
    (REPOSITORY / "DATA_AND_LICENSE_NOTICE.md").write_text(notice, encoding="utf-8")
    (REPOSITORY / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\nnode_modules/\nfrontend/dist/\nsite-dist/\ndist/\n.env\n*.pyc\n",
        encoding="utf-8",
    )

    files = sorted(path for path in REPOSITORY.rglob("*") if path.is_file())
    manifest = {
        "release_name": "Beyond Proximity public research code",
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "included_paths": included,
        "excluded_reason_counts": dict(sorted(excluded.items())),
        "raw_scannet_included": False,
        "raw_replica_included": False,
        "raw_captured_rgbd_included": False,
        "review_material_included": False,
        "secret_scan": "passed",
        "code_license_selected": (REPOSITORY / "LICENSE").is_file(),
        "files": {
            path.relative_to(REPOSITORY).as_posix(): sha256(path)
            for path in files
        },
    }
    manifest_path = OUTPUT_ROOT / "public-release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(manifest_path, REPOSITORY / "PUBLIC_RELEASE_MANIFEST.json")
    deterministic_zip(REPOSITORY, ARCHIVE)
    manifest["archive"] = ARCHIVE.relative_to(ROOT).as_posix()
    manifest["archive_sha256"] = sha256(ARCHIVE)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files" and key != "included_paths"}, indent=2))
    return manifest


if __name__ == "__main__":
    build()
