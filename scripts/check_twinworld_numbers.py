#!/usr/bin/env python3
"""Cross-check the anonymous TwinWorld paper against frozen experiment outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checks: list[tuple[str, Any, Any, bool]] = []

    def check(name: str, got: Any, expected: Any, tolerance: float = 0.0) -> None:
        if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
            ok = abs(float(got) - float(expected)) <= tolerance
        else:
            ok = got == expected
        checks.append((name, got, expected, ok))
        print(("OK " if ok else "MISMATCH "), name, f"got={got!r} expected={expected!r}")

    agent = read_json(
        ROOT
        / "outputs"
        / "instruction_agent"
        / "five_scene_qwen25_05b_graph_v1"
        / "metrics_summary.json"
    )
    a = agent["methods"]["graph_instruction"]
    check("agent.status", agent["status"], "complete")
    check("agent.queries", agent["query_count"], 150)
    check("agent.quality_denominator", agent["quality_query_count"], 125)
    check("agent.hit1", a["hit_at_1"], 0.368)
    check("agent.hit3", a["hit_at_3"], 0.696)
    check("agent.mrr", a["mrr"], 0.509333, 0.000001)
    check("agent.views", a["mean_views_checked"], 9.626667, 0.000001)
    check("agent.calls", a["total_model_calls"], 300)
    check("agent.tokens", a["total_tokens"], 239893)
    check("agent.valid", a["valid_response_rate"], 1.0)
    trace_count = len(
        list(
            (
                ROOT
                / "outputs"
                / "instruction_agent"
                / "five_scene_qwen25_05b_graph_v1"
                / "traces"
                / "graph_instruction"
            ).glob("*.json")
        )
    )
    check("agent.trace_count", trace_count, 150)

    raw = read_json(
        ROOT
        / "outputs"
        / "raw_rgbd_hierarchy"
        / "five_scene_clip_v1"
        / "metrics_summary.json"
    )
    c = raw["construction"]
    structure = raw["structure"]
    raw_method = raw["methods"]["raw_rgbd_hierarchy_clip"]
    check("raw.status", raw["status"], "complete")
    check("raw.rgb", c["rgb_image_count"], 97)
    check("raw.depth", c["depth_map_count"], 97)
    check("raw.poses", c["pose_count"], 97)
    check("raw.model_calls", c["model_call_count"], 15)
    check("raw.label_tokens", c["text_tokens"], 151)
    check("raw.manual_viewjson_reads", c["manual_viewjson_files_read"], 0)
    check("raw.manual_zone_reads", c["manual_zone_files_read"], 0)
    check("raw.pairwise_f1", structure["macro_pairwise_f1"], 0.486673, 0.000001)
    check("raw.rand", structure["macro_rand_index"], 0.637318, 0.000001)
    check("raw.hit1", raw_method["hit_at_1"], 0.368)
    check("raw.hit3", raw_method["hit_at_3"], 0.528)
    check("raw.views", raw_method["mean_views_checked"], 10.0)

    hierarchy = read_json(
        ROOT
        / "docs"
        / "experiments"
        / "hierarchy_construction"
        / "four_variant_v1"
        / "hierarchy_evaluation.json"
    )
    cursor = hierarchy["variants"]["cursor_agent_mcp"]
    semantic = hierarchy["variants"]["pose_semantic_merge"]
    check("cursor.direct_mcp_calls", hierarchy["agent_evidence"]["direct_mcp_calls"], True)
    check("cursor.mcp_calls", hierarchy["agent_evidence"]["mcp_tool_call_count"], 5)
    check("cursor.provider_tokens", hierarchy["agent_evidence"]["provider_tokens"], None)
    check("cursor.zones", cursor["structure"]["zone_count"], 18)
    check(
        "cursor.pairwise_f1",
        cursor["assignment_agreement"]["pairwise_f1"],
        0.616174,
        0.000001,
    )
    check("cursor.hit1", cursor["query_metrics"]["hit_at_1"], 0.704)
    check("cursor.hit3", cursor["query_metrics"]["hit_at_3"], 0.784)
    check(
        "cursor.views",
        cursor["query_metrics"]["mean_views_checked"],
        5.533333,
        0.000001,
    )
    check("semantic.hit1", semantic["query_metrics"]["hit_at_1"], 0.784)
    check("semantic.hit3", semantic["query_metrics"]["hit_at_3"], 0.904)

    bootstrap = read_json(
        ROOT / "docs" / "reports" / "final" / "twinworld_bootstrap_ci.json"
    )
    intervals = bootstrap["metrics"]
    check("bootstrap.samples", bootstrap["bootstrap_samples"], 10_000)
    check(
        "bootstrap.view_savings",
        intervals["view_savings_pct"]["estimate"],
        75.4887,
        0.000001,
    )
    check(
        "bootstrap.hit1_low",
        intervals["hit_at_1_graph_minus_flat"]["ci_95"]["low"],
        -0.184,
    )
    check(
        "bootstrap.hit1_high",
        intervals["hit_at_1_graph_minus_flat"]["ci_95"]["high"],
        0.008,
    )
    check(
        "bootstrap.hit3_low",
        intervals["hit_at_3_graph_minus_flat"]["ci_95"]["low"],
        -0.184,
    )
    check(
        "bootstrap.hit3_high",
        intervals["hit_at_3_graph_minus_flat"]["ci_95"]["high"],
        -0.056,
    )

    for track, query_count, acc, objects, tokens in (
        ("phase6_replica_calibrated_v3", 56, 1.0, 12.107143, 302.678571),
        ("phase6_scannet_calibrated_v3", 48, 0.604167, 4.729167, 219.3125),
        ("phase6_scannet_extended_v3", 64, 0.589286, 4.375, 200.0625),
    ):
        metrics = read_json(
            ROOT
            / "outputs"
            / "public_datasets"
            / track
            / "graph"
            / "grounding_summary.json"
        )["metrics"]
        check(f"{track}.queries", metrics["query_count"], query_count)
        check(f"{track}.acc025", metrics["acc_at_0_25"]["value"], acc, 0.000001)
        check(
            f"{track}.objects",
            metrics["efficiency"]["checked_objects"]["mean"],
            objects,
            0.000001,
        )
        check(
            f"{track}.tokens",
            metrics["efficiency"]["estimated_tokens"]["mean"],
            tokens,
            0.000001,
        )

    no_relation = read_json(
        ROOT
        / "outputs"
        / "public_datasets"
        / "phase6_scannet_calibrated_v3"
        / "ablation_no_relation"
        / "grounding_summary.json"
    )["metrics"]
    check("scannet.no_relation_acc025", no_relation["acc_at_0_25"]["value"], 0.5)

    scaling = read_json(
        ROOT / "outputs" / "reviewer_revision" / "scaling_v1" / "scaling_summary.json"
    )
    check("reviewer.scaling.status", scaling["status"], "complete")
    check("reviewer.scaling.measured_points", len(scaling["measured"]), 5)
    check("reviewer.scaling.max_measured_views", scaling["measured"][-1]["view_count"], 97)
    check("reviewer.scaling.provider_inference", scaling["protocol"]["provider_latency_or_cost_inferred"], False)

    matched_relation = read_json(
        ROOT / "outputs" / "reviewer_revision" / "matched_relation_v1" / "matched_relation_summary.json"
    )
    check("reviewer.relation.status", matched_relation["status"], "complete")
    check("reviewer.relation.queries", matched_relation["query_count"], 48)
    check("reviewer.relation.identical_sets", matched_relation["identical_candidate_sets_verified"], True)
    check("reviewer.relation.with_acc025", matched_relation["with_relation"]["acc_at_0_25"], 0.5625)
    check("reviewer.relation.without_acc025", matched_relation["without_relation"]["acc_at_0_25"], 0.5)
    check("reviewer.relation.ci_low", matched_relation["paired_bootstrap"]["acc_at_0_25"]["ci_95"]["low"], -0.020833)
    check("reviewer.relation.ci_high", matched_relation["paired_bootstrap"]["acc_at_0_25"]["ci_95"]["high"], 0.166667)

    matched_hierarchy = read_json(
        ROOT / "outputs" / "reviewer_revision" / "matched_hierarchy_v1" / "matched_hierarchy_summary.json"
    )
    check("reviewer.hierarchy.status", matched_hierarchy["status"], "complete")
    check("reviewer.hierarchy.budget_rows", len(matched_hierarchy["matched_budgets"]), 16)
    check("reviewer.hierarchy.random_rows", len(matched_hierarchy["structure"]["random_matched_complexity"]), 20)

    tex = (ROOT / "papers" / "twinworld" / "main.tex").read_text(encoding="utf-8")
    macros = (
        ROOT / "papers" / "twinworld" / "tables" / "new_experiment_macros.tex"
    ).read_text(encoding="utf-8")
    required = (
    r"\title{Beyond Proximity: Graph-Pruned\\Semantic Search over 3D Scene Records}",
    r"3D Gaussian Splatting \and RGB-D scene understanding",
        r"\author{Anonymous Authors}",
        r"\subsection{Instruction-agent traversal}",
        r"\subsection{Agent/MCP hierarchy construction}",
        r"\subsection{Automatic raw RGB-D hierarchy}",
        r"\subsection{Instruction-model diagnostic stress test}",
        r"\subsection{Raw RGB-D hierarchy versus manual reference}",
        r"\subsection{Semantic-record construction variants}",
        "figures/fig_instruction_agent.pdf",
        "figures/fig_raw_rgbd_hierarchy.pdf",
        "figures/fig_reviewer_controls.pdf",
        "figures/fig_reviewer_scaling.pdf",
        "figures/fig_agent_mcp_workflow.pdf",
        r"SayPlan~\cite{rana2023sayplan}",
        r"Search3D~\cite{takmaz2025search3d}",
        "reads zero manual ViewJSON or zone",
        r"\AgentTotalCalls{} model calls",
        "systems-feasibility",
        "Matched-candidate relation",
        r"1-4.77/19.40=75.4\%",
        r"\HitOneLow{}, \HitOneHigh{}",
        r"\HitThreeLow{}, \HitThreeHigh{}",
        "inconclusive",
        "significant decline",
        "Replica (56; 48+)",
        "ScanNet (48; 48+)",
        "Replica's exact-match ceiling is only a protocol sanity check",
        r"\MatchedRelationLow{},",
        "The superseded reranker underperformed because it applied low-confidence",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "openai/clip-vit-base-patch32",
    )
    for value in required:
        check(f"tex_has:{value}", value in tex, True)

    forbidden = (
        "Paper ID #XXXXX",
        "XXXXX",
        "Person 1",
        "Person 2",
        "Person 3",
        "Person 4",
        "ScanNet oracle-map pilot, graph search matches flat lexical Recall@1 / Acc@0.25 ($0.271$)",
        "Relation reranking does not help",
        "Stub / deterministic lexical matching is not live provider-backed VLM evaluation",
        r"$k=2,\ldots,\min(8,\lfloor n/3\rfloor)$",
        "turn hierarchy from an assumed component",
    )
    for value in forbidden:
        check(f"tex_free_of:{value}", value not in tex, True)

    expected_macros = {
        "AgentHitOne": "0.368",
        "AgentHitThree": "0.696",
        "AgentTotalCalls": "300",
        "AgentTotalTokens": "239893",
        "RawConstructionCalls": "15",
        "RawConstructionTokens": "151",
        "RawPairwiseFOne": "0.487",
        "RawHitThree": "0.528",
        "CursorMcpCalls": "5",
        "CursorPairwiseFOne": "0.616",
        "CursorHitThree": "0.784",
        "ViewSavingsPerQuery": "75.5",
        "HitOneLow": "-0.184",
        "HitOneHigh": "0.008",
        "HitThreeHigh": "-0.056",
    }
    for name, expected in expected_macros.items():
        match = re.search(
            rf"\\newcommand\{{\\{re.escape(name)}\}}\{{([^}}]+)\}}",
            macros,
        )
        check(f"macro:{name}", match.group(1) if match else None, expected)

    reviewer_macros = (
        ROOT / "papers" / "twinworld" / "tables" / "reviewer_revision_macros.tex"
    ).read_text(encoding="utf-8")
    for name, expected in {
        "ScaleMeasuredViews": "97",
        "MatchedRelationWith": "0.562",
        "MatchedRelationWithout": "0.500",
        "MatchedRelationLow": "-0.021",
        "MatchedRelationHigh": "0.167",
        "DisjointReferenceFOne": "0.877",
    }.items():
        match = re.search(rf"\\newcommand\{{\\{re.escape(name)}\}}\{{([^}}]+)\}}", reviewer_macros)
        check(f"reviewer_macro:{name}", match.group(1) if match else None, expected)

    validation_paths = (
        ROOT
        / "outputs"
        / "instruction_agent"
        / "five_scene_qwen25_05b_graph_v1"
        / "validation_report.json",
        ROOT
        / "outputs"
        / "raw_rgbd_hierarchy"
        / "five_scene_clip_v1"
        / "validation_report.json",
    )
    for path in validation_paths:
        check(
            f"validation:{path.parent.name}",
            read_json(path)["status"],
            "passed",
        )

    failures = [value for value in checks if not value[3]]
    report = ROOT / "docs" / "reports" / "final" / "twinworld_number_check.md"
    lines = [
        "# TwinWorld number cross-check",
        "",
        "The anonymous TwinWorld manuscript is checked against the frozen "
        "instruction-agent, raw RGB-D hierarchy, and calibrated public-dataset outputs.",
        "",
        f"Checks: {len(checks)}; mismatches: {len(failures)}.",
        "",
        "| Check | Got | Expected | Pass |",
        "|---|---|---|:---:|",
    ]
    for name, got, expected, ok in checks:
        lines.append(
            f"| `{name}` | `{got}` | `{expected}` | {'yes' if ok else 'NO'} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
