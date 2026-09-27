"""Generate quantitative LaTeX table rows from frozen experiment JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "papers" / "beyond-proximity" / "tables"
SOURCES = {
    "agent": REPO
    / "outputs"
    / "agent_semantic"
    / "five_scene_four_variant_v1"
    / "metrics_summary.json",
    "hierarchy": REPO
    / "docs"
    / "experiments"
    / "hierarchy_construction"
    / "four_variant_v1"
    / "hierarchy_evaluation.json",
    "replica": REPO
    / "docs"
    / "experiments"
    / "public_datasets"
    / "phase6_7_v3"
    / "replica_variant_comparison.json",
    "scannet": REPO
    / "docs"
    / "experiments"
    / "public_datasets"
    / "phase6_7_v3"
    / "scannet_original_variant_comparison.json",
    "scannet_extended": REPO
    / "docs"
    / "experiments"
    / "public_datasets"
    / "phase6_7_v3"
    / "scannet_extended_variant_comparison.json",
    "relation": REPO
    / "docs"
    / "experiments"
    / "public_datasets"
    / "phase6_7_v3"
    / "scannet_relation_audit.json",
    "conceptgraphs_scannet": REPO
    / "outputs"
    / "baselines"
    / "conceptgraphs_scannet_full_v1"
    / "metrics_summary.json",
    "conceptgraphs_replica": REPO
    / "outputs"
    / "baselines"
    / "conceptgraphs_replica_full_v1"
    / "metrics_summary.json",
    "langsplat": REPO
    / "outputs"
    / "baselines"
    / "langsplat_scannet_full_v1"
    / "metrics_summary.json",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decimal(value: float | None, places: int = 3) -> str:
    if value is None:
        return "N/A"
    quantum = Decimal(1).scaleb(-places)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    rendered = f"{rounded:.{places}f}"
    return rendered[1:] if rendered.startswith("0.") else rendered


def number(value: float, places: int = 2) -> str:
    return f"{value:.{places}f}"


def integer(value: float | int) -> str:
    return f"{round(value):,}"


def row(*cells: object) -> str:
    return " & ".join(str(cell) for cell in cells) + r" \\"


def by_variant(data: dict, variant: str) -> dict:
    return next(item for item in data["variants"] if item["variant"] == variant)


def write_if_changed(path: Path, text: str) -> None:
    text = text.rstrip() + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def agent_rows(data: dict) -> str:
    names = {
        "flat_lexical": "Flat lexical",
        "graph_lexical": "Graph lexical",
        "flat_semantic_embedding": "Flat semantic",
        "graph_semantic_embedding": "Graph semantic",
    }
    rows = []
    for key, label in names.items():
        item = data["methods"][key]
        q = item["quality"]
        e = item["efficiency"]
        a = item["accounting"]
        rows.append(
            row(
                label,
                decimal(q["hit_at_1"]),
                decimal(q["hit_at_3"]),
                decimal(q["mrr"]),
                decimal(q["expected_object_hit"]),
                number(e["views_checked"]),
                integer(e["input_tokens"]),
                number(e["elapsed_ms"]),
                number(e["model_call_count"]),
                integer(a["total_input_tokens"]),
            )
        )
    return "\n".join(rows)


def hierarchy_rows(data: dict) -> str:
    names = {
        "manual_reference": "Manual reference",
        "pose_only": "Pose only",
        "pose_semantic_merge": "Pose + semantic",
        "cursor_agent_mcp": "Cursor Agent + MCP",
    }
    rows = []
    for key, label in names.items():
        item = data["variants"][key]
        structure = item["structure"]
        agreement = item["assignment_agreement"]
        query = item["query_metrics"]
        seconds = item["construction_cost"]["measured_local_seconds"]
        rows.append(
            row(
                label,
                structure["zone_count"],
                decimal(agreement["pairwise_f1"]),
                decimal(agreement["weighted_best_zone_jaccard"]),
                decimal(query["hit_at_1"]),
                decimal(query["hit_at_3"]),
                decimal(query["mrr"]),
                number(query["mean_views_checked"]),
                integer(query["mean_input_tokens"]),
                decimal(seconds) if seconds is not None else "N/A",
            )
        )
    return "\n".join(rows)


def public_rows(replica: dict, scannet: dict, extended: dict) -> str:
    tracks = (
        ("Replica, $Q=56$", replica),
        ("ScanNet, $Q=48$", scannet),
        ("ScanNet ext., $Q=64$", extended),
    )
    names = {
        "graph": "Graph",
        "graph_fallback": "Graph+fallback",
        "flat_lexical": "Flat lexical",
    }
    rows: list[str] = []
    for track_index, (label, data) in enumerate(tracks):
        for method_index, (key, method) in enumerate(names.items()):
            item = by_variant(data, key)
            track = rf"\multirow{{3}}{{*}}{{{label}}}" if method_index == 0 else ""
            rows.append(
                row(
                    track,
                    method,
                    decimal(item["recall_at_1"]),
                    decimal(item["recall_at_3"]),
                    decimal(item["recall_at_5"]),
                    decimal(item["mrr"]),
                    decimal(item["acc_at_0_25"]),
                    decimal(item["negative_accuracy"]),
                    number(item["avg_checked_objects"]),
                    integer(item["avg_estimated_input_tokens"]),
                )
            )
        if track_index < len(tracks) - 1:
            rows.append(r"\addlinespace[3pt]")
    return "\n".join(rows)


def relation_rows(data: dict) -> str:
    overall = data["overall"]
    return "\n".join(
        (
            row(
                "hit@1",
                decimal(overall["without_relation_hit_at_1"]),
                decimal(overall["with_relation_hit_at_1"]),
            ),
            row(
                "Parsed / unparsed",
                rf"\multicolumn{{2}}{{c}}{{{data['parsed_relation_count']} / {data['unparsed_relation_count']}}}",
            ),
            row("Beneficial changes", 0, overall["beneficial"]),
            row("Harmful changes", 0, overall["harmful"]),
            row("Validated changed winners", 0, overall["validated_relation_changes"]),
        )
    )


def external_rows(cg_scan: dict, cg_replica: dict, langsplat: dict) -> str:
    def metrics(data: dict) -> dict:
        return data["metrics"]

    scan = metrics(cg_scan)
    replica = metrics(cg_replica)
    lang = metrics(langsplat)
    return "\n".join(
        (
            row(
                "CG--ScanNet",
                8,
                48,
                f"A@.25={decimal(scan['bbox_acc_at_0_25']['value'])}",
                decimal(scan["runtime_seconds"]["mean"]),
                scan["token_usage"]["total"],
            ),
            row(
                "CG--Replica",
                8,
                56,
                f"A@.25={decimal(replica['bbox_acc_at_0_25']['value'])}",
                decimal(replica["runtime_seconds"]["mean"]),
                replica["token_usage"]["total"],
            ),
            row(
                "LangSplat--ScanNet",
                1,
                6,
                f"view hit={decimal(lang['expected_view_hit']['value'])}",
                decimal(lang["runtime_seconds"]["mean"]),
                lang["token_usage"]["total"],
            ),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {name: load(path) for name, path in SOURCES.items()}
    outputs = {
        "agent_variant_rows.tex": agent_rows(data["agent"]),
        "hierarchy_rows.tex": hierarchy_rows(data["hierarchy"]),
        "public_rows.tex": public_rows(
            data["replica"], data["scannet"], data["scannet_extended"]
        ),
        "relation_rows.tex": relation_rows(data["relation"]),
        "external_rows.tex": external_rows(
            data["conceptgraphs_scannet"],
            data["conceptgraphs_replica"],
            data["langsplat"],
        ),
    }
    for name, text in outputs.items():
        # Suppress the end-of-file space so a following booktabs rule remains
        # the first token after the final generated row.
        write_if_changed(out_dir / name, f"{text}%")

    manifest = {
        "schema_version": "academic_table_export.v1",
        "sources": {
            name: {
                "path": str(path.relative_to(REPO)).replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in SOURCES.items()
        },
        "outputs": sorted(outputs),
    }
    write_if_changed(
        out_dir / "table_data_manifest.json",
        json.dumps(manifest, indent=2),
    )
    print(f"Generated {len(outputs)} quantitative table row files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
