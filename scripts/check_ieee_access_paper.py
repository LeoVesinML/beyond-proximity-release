#!/usr/bin/env python3
"""Validate the IEEE Access source, generated evidence, and compiled PDF."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers" / "ieee-access"
TEX = PAPER / "main.tex"
PDF = PAPER / "main.pdf"

AUTHORS = (
    "Andrey Kuznetsov",
    "Mahmoud Mousatat",
    "Ahmad Taha",
    "Aleksandr Medvedev",
    "Leo Vesin",
    "Telman Nuruzov",
    "Nikita Shankin",
)

REQUIRED_PHRASES = (
    r"\documentclass{ieeeaccess}",
    "Recall-controlled traversal",
    "Algorithm 1: Recall-controlled query traversal",
    "Algorithm 2: Raw RGB-D hierarchy construction",
    "Held-Out Efficiency--Quality Frontier",
    "Measured Public-Collection Scale and ANN",
    "Predicted-Map Retrieval",
    "Sensitivity Analysis",
    "Threats to Validity",
    "155,535",
    "same-input",
    "OpenAI Codex assisted",
    r"\bibliographystyle{IEEEtran}",
    r"\history{Prepared for submission, September 2026.}",
    r"\EOD",
)

FORBIDDEN = (
    "Anonymous",
    "TwinWorld",
    "ECCV",
    "Paper ID",
    "XXXXX",
    "TODO",
    "TBD",
    "fig_reviewer_scaling",
    "hollow points",
    "sizes are extrapolated",
    r"\captionof",
    r"\balance",
)

CLASS_HASH = "67b73c4da05479d592a9449c38a81fcc7779d78753085bb44199e7f2746b757d"

EVIDENCE = (
    "outputs/ieee_revision/recall_frontier_v1/frontier_summary.json",
    "outputs/ieee_revision/bge_frontier_v1/bge_frontier_summary.json",
    "outputs/ieee_revision/public_scale_ann_v1/scale_ann_summary.json",
    "outputs/ieee_revision/predicted_map_pair_v1/predicted_map_pair_summary.json",
    "outputs/ieee_revision/sensitivity_v1/sensitivity_summary.json",
    "outputs/ieee_revision/hierarchy_scorer_grid_v1/hierarchy_scorer_grid_summary.json",
    "outputs/reviewer_revision/matched_relation_v1/matched_relation_summary.json",
    "outputs/reviewer_revision/matched_hierarchy_v1/matched_hierarchy_summary.json",
)

FIGURES = (
    "fig_tree_traversal.pdf",
    "fig_agent_mcp_workflow.pdf",
    "fig_scene_gallery.pdf",
    "fig_raw_rgbd_hierarchy.pdf",
    "fig_reviewer_controls.pdf",
    "fig_scorer_frontiers.pdf",
    "fig_mitigation_ablation.pdf",
    "fig_public_scale_ann.pdf",
    "fig_predicted_map_pair.pdf",
    "fig_sensitivity.pdf",
    "fig_instruction_pair.pdf",
    "graphical_abstract.pdf",
)


def fail(messages: list[str]) -> int:
    for message in messages:
        print(f"ERROR: {message}", file=sys.stderr)
    return 1


def main() -> int:
    errors: list[str] = []
    if not TEX.is_file():
        return fail([f"missing manuscript: {TEX}"])
    text = TEX.read_text(encoding="utf-8")

    positions = [text.find(author) for author in AUTHORS]
    if any(value < 0 for value in positions):
        errors.append("one or more authors are missing")
    elif positions != sorted(positions):
        errors.append("authors are not in the agreed order")
    if text.count(r"\begin{IEEEbiographynophoto}") != len(AUTHORS):
        errors.append("one biography block is required for every author")
    if text.count(r"\EOD") != 1:
        errors.append("the official class requires exactly one end marker")

    for phrase in REQUIRED_PHRASES:
        if phrase not in text:
            errors.append(f"missing required phrase: {phrase}")
    for phrase in FORBIDDEN:
        if phrase.lower() in text.lower():
            errors.append(f"forbidden placeholder or obsolete wording: {phrase}")

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
    if not abstract_match:
        errors.append("abstract not found")
    else:
        words = re.findall(r"[A-Za-z0-9@.-]+", abstract_match.group(1))
        if not 150 <= len(words) <= 250:
            errors.append(f"abstract has {len(words)} words; expected 150-250")
        if r"\cite" in abstract_match.group(1):
            errors.append("abstract must not contain citations")

    keyword_match = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", text, re.DOTALL)
    if not keyword_match:
        errors.append("keywords not found")
    else:
        keywords = [value.strip(" .\n") for value in keyword_match.group(1).split(",")]
        if not 3 <= len(keywords) <= 10:
            errors.append(f"found {len(keywords)} keywords; expected 3-10")

    cited = set(re.findall(r"\\cite\{([^}]+)\}", text))
    cited_keys = {key.strip() for group in cited for key in group.split(",")}
    bib_text = (PAPER / "refs.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    missing_keys = sorted(cited_keys - bib_keys)
    if missing_keys:
        errors.append("citations missing from refs.bib: " + ", ".join(missing_keys))
    if len(bib_keys) < 45:
        errors.append(f"journal literature review has only {len(bib_keys)} references; expected at least 45")

    class_path = PAPER / "ieeeaccess.cls"
    actual_hash = hashlib.sha256(class_path.read_bytes()).hexdigest()
    if actual_hash != CLASS_HASH:
        errors.append("ieeeaccess.cls does not match the official 2026-05-13 template")

    for relative in EVIDENCE:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing evidence: {relative}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            errors.append(f"evidence is not complete: {relative}")

    for name in FIGURES:
        if not (PAPER / "figures" / name).is_file():
            errors.append(f"missing generated figure: {name}")

    if not PDF.is_file():
        errors.append("compiled PDF is missing")
    elif PDF.stat().st_size >= 40 * 1024 * 1024:
        errors.append("compiled PDF exceeds the 40 MB submission limit")

    if errors:
        return fail(errors)
    print(
        "IEEE Access technical check passed: template, evidence, measured claims, "
        "references, metadata shape, and submission file constraints."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
