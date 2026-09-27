#!/usr/bin/env python3
"""Validate the named preprint against the canonical review manuscript."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "papers" / "beyond-proximity"
PAPER = TARGET / "main.tex"
METADATA = TARGET / "arxiv_metadata.json"
SYNC_MANIFEST = TARGET / "canonical_sync.json"

AUTHORS = (
    "Andrey Kuznetsov",
    "Mahmoud Mousatat",
    "Ahmad Taha",
    "Aleksandr Medvedev",
    "Leo Vesin",
    "Telman Nuruzov",
    "Nikita Shankin",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    failures: list[str] = []
    actual = PAPER.read_text(encoding="utf-8") if PAPER.is_file() else ""

    if not SYNC_MANIFEST.is_file():
        failures.append("named preprint synchronization manifest is missing")
        sync_manifest = {}
    else:
        sync_manifest = json.loads(SYNC_MANIFEST.read_text(encoding="utf-8"))
        if sha256(PAPER) != sync_manifest.get("rendered_source_sha256"):
            failures.append("named preprint source differs from its synchronization manifest")

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("authors") != list(AUTHORS):
        failures.append("arXiv metadata author order differs from the approved order")
    if metadata.get("title") != "Beyond Proximity: Graph-Pruned Semantic Search over 3D Scene Records":
        failures.append("arXiv metadata title differs from the manuscript")

    positions = [actual.find(author) for author in AUTHORS]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        failures.append("named manuscript author order is missing or incorrect")

    for marker in (
        "TwinWorld",
        "OpenReview",
        "Anonymous Authors",
        "Anonymous Institution",
        "Paper ID",
        "#XXXXX",
        "SemanticSplat",
        "Leo Vizan",
        "Alexandr Medvedev",
    ):
        if marker.lower() in actual.lower() or marker.lower() in json.dumps(metadata).lower():
            failures.append(f"venue/stale marker remains in arXiv material: {marker}")

    for relative, expected_digest in sync_manifest.get("dependencies", {}).items():
        academic = TARGET / relative
        if not academic.is_file():
            failures.append(f"named preprint dependency is missing: {academic}")
        elif sha256(academic) != expected_digest:
            failures.append(f"named preprint dependency differs from its manifest: {academic}")

    required = (
        r"\subsection{Hierarchical scene records}",
        r"\subsection{Agent/MCP hierarchy construction}",
        r"\subsection{Instruction-agent traversal}",
        r"\subsection{Automatic raw RGB-D hierarchy}",
        r"figures/fig_instruction_agent.pdf",
        r"figures/fig_raw_rgbd_hierarchy.pdf",
        r"figures/fig_reviewer_controls.pdf",
        r"figures/fig_reviewer_scaling.pdf",
        r"\AgentTotalCalls{} model calls",
        r"\RawConstructionCalls{} model calls",
        r"systems-feasibility",
    )
    for value in required:
        if value not in actual:
            failures.append(f"named preprint is missing canonical content: {value}")

    if failures:
        print("Academic paper synchronization check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        "Academic paper synchronization check passed "
        "(canonical content, authors, figures, metadata, and venue neutrality)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
