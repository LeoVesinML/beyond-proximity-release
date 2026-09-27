#!/usr/bin/env python3
"""Build and independently compile the venue-neutral arXiv source archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from sync_academic_from_twinworld import sync as sync_academic_preprint


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "papers" / "beyond-proximity"
DEFAULT_OUTPUT = ROOT / "dist" / "arxiv"
PACKAGE_NAME = "beyond-proximity-arxiv-v1"

SOURCE_FILES = (
    "main.tex",
    "cvpr.sty",
    "refs.bib",
    "figures/fig_tree_traversal.pdf",
    "figures/fig_agent_mcp_workflow.pdf",
    "figures/fig_scene_gallery.pdf",
    "figures/fig_instruction_agent.pdf",
    "figures/fig_raw_rgbd_hierarchy.pdf",
    "figures/fig_hierarchy_construction_variants.pdf",
    "figures/fig_five_scene_summary.pdf",
    "tables/new_experiment_macros.tex",
)

AUTHORS = (
    "Andrey Kuznetsov",
    "Mahmoud Mousatat",
    "Ahmad Taha",
    "Aleksandr Medvedev",
    "Leo Vesin",
    "Telman Nuruzov",
    "Nikita Shankin",
)

FORBIDDEN_MARKERS = (
    "TwinWorld",
    "OpenReview",
    "Paper ID",
    "Anonymous Authors",
    "SemanticSplat",
    "Leo Vizan",
    "Alexandr Medvedev",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_tectonic() -> Path | None:
    executable = shutil.which("tectonic")
    if executable:
        return Path(executable)
    cache = Path.home() / ".codex" / "plugins" / "cache" / "openai-bundled" / "latex"
    candidates = sorted(cache.rglob("tectonic.exe"), reverse=True) if cache.exists() else []
    return candidates[0] if candidates else None


def validate_metadata(metadata: dict, main_source: str) -> None:
    if metadata.get("authors") != list(AUTHORS):
        raise ValueError("arXiv metadata author order does not match the approved order")
    positions = [main_source.find(author) for author in AUTHORS]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ValueError("main.tex author list is missing names or has the wrong order")
    combined = json.dumps(metadata, ensure_ascii=False) + "\n" + main_source
    stale = [marker for marker in FORBIDDEN_MARKERS if marker.lower() in combined.lower()]
    if stale:
        raise ValueError(f"venue/stale markers found in arXiv material: {stale}")


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


def compile_package(package_dir: Path, output_dir: Path) -> dict:
    tectonic = find_tectonic()
    if tectonic is None:
        raise RuntimeError("Tectonic was not found; cannot verify the extracted arXiv package")
    with tempfile.TemporaryDirectory(prefix="arxiv-compile-", dir=output_dir) as temp_name:
        work = Path(temp_name)
        shutil.copytree(package_dir, work / "source", dirs_exist_ok=True)
        (work / "build").mkdir(parents=True, exist_ok=True)
        command = [
            str(tectonic),
            "-X",
            "compile",
            "--outdir",
            str(work / "build"),
            "--outfmt",
            "pdf",
            "--untrusted",
            "main.tex",
        ]
        result = subprocess.run(
            command,
            cwd=work / "source",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        pdf = work / "build" / "main.pdf"
        if result.returncode or not pdf.is_file() or pdf.stat().st_size == 0:
            raise RuntimeError(
                "independent arXiv package compile failed\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )
        return {
            "command": command,
            "pdf_size_bytes": pdf.stat().st_size,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }


def build(output_dir: Path) -> dict:
    sync_result = sync_academic_preprint()
    output_dir = output_dir.resolve()
    package_dir = output_dir / PACKAGE_NAME
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    missing = [relative for relative in SOURCE_FILES if not (PAPER / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing arXiv source files: {missing}")

    metadata_path = PAPER / "arxiv_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    main_source = (PAPER / "main.tex").read_text(encoding="utf-8")
    validate_metadata(metadata, main_source)

    for relative in SOURCE_FILES:
        source = PAPER / relative
        target = package_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    compile_result = compile_package(package_dir, output_dir)
    zip_path = output_dir / f"{PACKAGE_NAME}.zip"
    deterministic_zip(package_dir, zip_path)

    file_manifest = {
        relative: {
            "sha256": sha256(package_dir / relative),
            "size_bytes": (package_dir / relative).stat().st_size,
        }
        for relative in SOURCE_FILES
    }
    manifest = {
        "package": zip_path.relative_to(ROOT).as_posix(),
        "package_sha256": sha256(zip_path),
        "authors": list(AUTHORS),
        "canonical_sync": sync_result,
        "files": file_manifest,
        "independent_compile": compile_result,
        "venue_neutral_scan": "passed",
    }
    manifest_path = output_dir / "beyond-proximity-arxiv-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(metadata_path, output_dir / "arxiv_metadata.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
