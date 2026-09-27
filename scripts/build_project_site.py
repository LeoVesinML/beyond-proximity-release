#!/usr/bin/env python3
"""Assemble the static project site from tracked research evidence."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "site"
DEFAULT_OUTPUT = ROOT / "site-dist"

ASSETS = (
    (ROOT / "backend/data/scenes/ConferenceHall-capture-pilot/images/v018.png", "assets/hero-conference-hall.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_method_workflow.png", "assets/method-workflow.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_tree_traversal.png", "assets/tree-traversal.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_evaluation_protocol.png", "assets/evaluation-protocol.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_agent_mcp_protocol.png", "assets/agent-mcp-workflow.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_five_scene_summary.png", "assets/five-scene-summary.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_agent_semantic_results.png", "assets/instruction-agent.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_hierarchy_comparison.png", "assets/raw-rgbd-hierarchy.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_hierarchy_comparison.png", "assets/hierarchy-construction-variants.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_scene_gallery.png", "assets/scene-gallery.png"),
    (ROOT / "papers/beyond-proximity/figures/fig_qualitative_bbox.png", "assets/qualitative-bbox.png"),
    (ROOT / "papers/beyond-proximity/main.pdf", "assets/research-preprint.pdf"),
    (ROOT / "docs/reports/final/graph_construction_cost.json", "data/graph-construction-cost.json"),
    (ROOT / "outputs/graph_vs_flat/five_scene_graph_vs_flat_v2/metrics_summary.json", "data/internal-metrics.json"),
    (ROOT / "outputs/agent_semantic/five_scene_four_variant_v1/metrics_summary.json", "data/semantic-controls.json"),
    (ROOT / "outputs/instruction_agent/five_scene_qwen25_05b_graph_v1/metrics_summary.json", "data/instruction-agent.json"),
    (ROOT / "outputs/raw_rgbd_hierarchy/five_scene_clip_v1/metrics_summary.json", "data/raw-rgbd-hierarchy.json"),
    (ROOT / "docs/experiments/public_datasets/phase6_7_v3/scannet_original_variant_comparison.json", "data/scannet-original-v3.json"),
    (ROOT / "docs/experiments/public_datasets/phase6_7_v3/scannet_extended_variant_comparison.json", "data/scannet-extended-v3.json"),
    (ROOT / "docs/experiments/public_datasets/phase6_7_v3/replica_variant_comparison.json", "data/replica-v3.json"),
)

METRIC_SOURCES = {
    "internal": ROOT / "outputs/graph_vs_flat/five_scene_graph_vs_flat_v2/metrics_summary.json",
    "semantic": ROOT / "outputs/agent_semantic/five_scene_four_variant_v1/metrics_summary.json",
    "instruction": ROOT / "outputs/instruction_agent/five_scene_qwen25_05b_graph_v1/metrics_summary.json",
    "raw_rgbd": ROOT / "outputs/raw_rgbd_hierarchy/five_scene_clip_v1/metrics_summary.json",
    "hierarchy": ROOT / "docs/experiments/hierarchy_construction/four_variant_v1/hierarchy_evaluation.json",
    "scannet": ROOT / "docs/experiments/public_datasets/phase6_7_v3/scannet_original_variant_comparison.json",
    "scannet_extended": ROOT / "docs/experiments/public_datasets/phase6_7_v3/scannet_extended_variant_comparison.json",
    "replica": ROOT / "docs/experiments/public_datasets/phase6_7_v3/replica_variant_comparison.json",
}


def is_lfs_pointer(path: Path) -> bool:
    return path.stat().st_size < 1024 and path.read_bytes().startswith(
        b"version https://git-lfs.github.com/spec"
    )


def prepare_output(output: Path) -> None:
    resolved = output.resolve()
    if resolved.parent != ROOT.resolve() or resolved.name != "site-dist":
        raise ValueError(f"Refusing to replace unexpected site output: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metric source is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_variant(payload: dict[str, object], name: str) -> dict[str, object]:
    for variant in payload["variants"]:
        if variant["variant"] == name:
            return variant
    raise ValueError(f"Missing required variant {name!r}")


def build_evidence_summary() -> dict[str, object]:
    evidence = {name: load_json(path) for name, path in METRIC_SOURCES.items()}
    internal = evidence["internal"]["summary"]
    semantic = evidence["semantic"]
    instruction = evidence["instruction"]
    raw_rgbd = evidence["raw_rgbd"]
    hierarchy = evidence["hierarchy"]
    scannet_graph = find_variant(evidence["scannet"], "graph")
    scannet_fallback = find_variant(evidence["scannet"], "graph_fallback")
    scannet_flat = find_variant(evidence["scannet"], "flat_lexical")
    scannet_extended = find_variant(evidence["scannet_extended"], "graph")
    replica = find_variant(evidence["replica"], "graph")

    methods = semantic["methods"]
    qwen = instruction["methods"]["graph_instruction"]
    raw_method = raw_rgbd["methods"]["raw_rgbd_hierarchy_clip"]
    flat_clip = raw_rgbd["methods"]["flat_clip"]
    cursor = hierarchy["variants"]["cursor_agent_mcp"]

    return {
        "schema_version": "graph_pruned_search.public_evidence.v2",
        "generated_from": {name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in METRIC_SOURCES.items()},
        "internal": internal,
        "semantic_control": methods["graph_semantic_embedding"],
        "instruction_agent": qwen,
        "raw_rgbd": {
            "construction": raw_rgbd["construction"],
            "structure": {
                "macro_pairwise_f1": raw_rgbd["structure"]["macro_pairwise_f1"],
                "macro_rand_index": raw_rgbd["structure"]["macro_rand_index"],
            },
            "retrieval": raw_method,
            "flat_clip": flat_clip,
            "views_reduction_pct": raw_rgbd["comparisons"]["raw_vs_flat_views_reduction_pct"],
        },
        "cursor_mcp": {
            "mcp_tool_call_count": hierarchy["agent_evidence"]["mcp_tool_call_count"],
            "provider_usage_status": "not exposed by the Cursor Agent UI; never imputed",
            "pairwise_f1": cursor["assignment_agreement"]["pairwise_f1"],
            "best_zone_jaccard": cursor["assignment_agreement"]["weighted_best_zone_jaccard"],
            "query_metrics": cursor["query_metrics"],
        },
        "public_datasets": {
            "scannet_original": {
                "query_count": 48,
                "graph": scannet_graph,
                "graph_fallback": scannet_fallback,
                "flat_lexical": scannet_flat,
            },
            "scannet_extended": {"query_count": 64, "graph": scannet_extended},
            "replica": {"query_count": 56, "positive_count": 48, "negative_count": 8, "graph": replica},
        },
    }


def build(output: Path) -> dict[str, object]:
    prepare_output(output)
    for source_path in SOURCE.rglob("*"):
        if not source_path.is_file() or source_path.name == "README.md":
            continue
        relative = source_path.relative_to(SOURCE)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    copied_assets: list[str] = []
    for source_path, relative in ASSETS:
        if not source_path.is_file():
            raise FileNotFoundError(f"Required site artifact is missing: {source_path}")
        if is_lfs_pointer(source_path):
            raise RuntimeError(f"Required LFS asset is still a pointer; run git lfs pull: {source_path}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied_assets.append(relative)

    (output / ".nojekyll").write_text("\n", encoding="ascii")
    evidence_summary = build_evidence_summary()
    (output / "data" / "evidence-summary.json").write_text(
        json.dumps(evidence_summary, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(output / "index.html", output / "404.html")
    report = {
        "output": str(output),
        "asset_count": len(copied_assets),
        "assets": copied_assets,
        "licensed_raw_data_included": False,
    }
    (output / "build-manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
