#!/usr/bin/env python3
"""Audit every relation-bearing query against its no-relation counterpart."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def expected_ids(query: dict[str, Any]) -> set[str]:
    values = {
        str(value)
        for value in query.get("expected_object_ids", [])
        if value is not None
    }
    if query.get("expected_object_id") is not None:
        values.add(str(query["expected_object_id"]))
    return values


def ranked_ids(result: dict[str, Any]) -> list[str]:
    payload = result.get("result")
    if not isinstance(payload, dict):
        return []
    values = payload.get("ranked_object_ids")
    ranked = (
        [str(value) for value in values if value is not None]
        if isinstance(values, list)
        else []
    )
    selected = payload.get("selected_object_id")
    if selected is not None and str(selected) not in ranked:
        ranked.insert(0, str(selected))
    return ranked


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(data.get("query_id") or item.stem): data
        for item in sorted(path.glob("*.json"))
        if isinstance((data := read_json(item)), dict)
    }


def hit_at_1(ranked: list[str], expected: set[str]) -> bool:
    return bool(ranked and ranked[0] in expected)


def audit(
    benchmark_path: Path,
    relation_results: Path,
    no_relation_results: Path,
) -> dict[str, Any]:
    benchmark = read_json(benchmark_path)
    queries = [
        query
        for query in benchmark.get("queries", [])
        if isinstance(query, dict)
        and str(query.get("query_type")) in {"relational", "multi_hop"}
    ]
    relation = load_results(relation_results)
    no_relation = load_results(no_relation_results)
    rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        with_result = relation.get(query_id, {})
        without_result = no_relation.get(query_id, {})
        plan = (
            with_result.get("structured_plan")
            if isinstance(with_result.get("structured_plan"), dict)
            else {}
        )
        relation_name = str(
            plan.get("relation")
            or query.get("expected_relation")
            or "unparsed"
        )
        wanted = expected_ids(query)
        with_ranked = ranked_ids(with_result)
        without_ranked = ranked_ids(without_result)
        with_hit = hit_at_1(with_ranked, wanted)
        without_hit = hit_at_1(without_ranked, wanted)
        result_payload = (
            with_result.get("result")
            if isinstance(with_result.get("result"), dict)
            else {}
        )
        selected_details = (
            result_payload.get("score_details")
            if isinstance(result_payload.get("score_details"), dict)
            else {}
        )
        audit_payload = (
            result_payload.get("relation_audit")
            if isinstance(result_payload.get("relation_audit"), dict)
            else {}
        )
        rows.append(
            {
                "query_id": query_id,
                "query": str(query.get("query") or ""),
                "scene_id": str(query.get("scene_id")),
                "source_dataset": str(query.get("source_dataset")),
                "query_type": str(query.get("query_type")),
                "relation": relation_name,
                "anchor": plan.get("anchor"),
                "anchor_secondary": plan.get("anchor_secondary"),
                "expected_object_ids": sorted(wanted),
                "with_relation_top1": with_ranked[0] if with_ranked else None,
                "without_relation_top1": without_ranked[0] if without_ranked else None,
                "with_relation_hit_at_1": with_hit,
                "without_relation_hit_at_1": without_hit,
                "top1_changed": with_ranked[:1] != without_ranked[:1],
                "effect": (
                    "beneficial"
                    if with_hit and not without_hit
                    else "harmful"
                    if without_hit and not with_hit
                    else "neutral"
                ),
                "relation_satisfied": result_payload.get("relation_satisfied"),
                "relation_applied": selected_details.get("relation_applied"),
                "validated_relation_change": bool(
                    with_ranked[:1] != without_ranked[:1]
                    and selected_details.get("relation_applied") is True
                ),
                "relation_confidence": selected_details.get("relation_confidence"),
                "relation_source": selected_details.get("relation_source"),
                "relation_ranking_changed": audit_payload.get("ranking_changed"),
                "relation_candidate_count": audit_payload.get(
                    "evaluated_candidate_count"
                ),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["relation"]].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        return {
            "query_count": count,
            "with_relation_hit_at_1": (
                round(
                    sum(row["with_relation_hit_at_1"] for row in group) / count,
                    6,
                )
                if count
                else None
            ),
            "without_relation_hit_at_1": (
                round(
                    sum(row["without_relation_hit_at_1"] for row in group)
                    / count,
                    6,
                )
                if count
                else None
            ),
            "beneficial": sum(row["effect"] == "beneficial" for row in group),
            "harmful": sum(row["effect"] == "harmful" for row in group),
            "neutral": sum(row["effect"] == "neutral" for row in group),
            "top1_changed": sum(row["top1_changed"] for row in group),
            "relation_applied_to_winner": sum(
                row["relation_applied"] is True for row in group
            ),
            "validated_relation_changes": sum(
                row["validated_relation_change"] for row in group
            ),
        }

    return {
        "schema_version": "semanticsplat.relation_reasoning_audit.v1",
        "benchmark_path": display_path(benchmark_path),
        "relation_results": display_path(relation_results),
        "no_relation_results": display_path(no_relation_results),
        "query_count": len(rows),
        "parsed_relation_count": sum(row["relation"] != "unparsed" for row in rows),
        "unparsed_relation_count": sum(row["relation"] == "unparsed" for row in rows),
        "overall": summarize(rows),
        "by_relation": {
            name: summarize(group) for name, group in sorted(groups.items())
        },
        "effect_counts": dict(Counter(row["effect"] for row in rows)),
        "beneficial_examples": [
            row for row in rows if row["effect"] == "beneficial"
        ][:5],
        "harmful_examples": [
            row for row in rows if row["effect"] == "harmful"
        ][:5],
        "per_query": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        "# Relation reasoning audit",
        "",
        f"- Relation-bearing queries inspected: {report['query_count']}",
        f"- Parsed relation: {report['parsed_relation_count']}",
        f"- Unparsed relation: {report['unparsed_relation_count']}",
        f"- Hit@1 with / without relation: "
        f"{overall['with_relation_hit_at_1']} / "
        f"{overall['without_relation_hit_at_1']}",
        f"- Beneficial / harmful / neutral: "
        f"{overall['beneficial']} / {overall['harmful']} / {overall['neutral']}",
        f"- Validated relation-driven top-1 changes: "
        f"{overall['validated_relation_changes']}",
        "",
        "| Relation | Queries | Hit@1 with | Hit@1 without | Beneficial | Harmful | Validated changes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for relation, record in report["by_relation"].items():
        lines.append(
            f"| {relation} | {record['query_count']} | "
            f"{record['with_relation_hit_at_1']} | "
            f"{record['without_relation_hit_at_1']} | "
            f"{record['beneficial']} | {record['harmful']} | "
            f"{record['validated_relation_changes']} |"
        )
    lines.extend(
        [
            "",
            "Relations are allowed to affect ranking only after anchor-confidence, "
            "bbox-validity, relation-confidence, and lexical score-margin gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["query_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["expected_object_ids"] = ";".join(row["expected_object_ids"])
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--with-relation", required=True, type=Path)
    parser.add_argument("--without-relation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = audit(
        args.benchmark.resolve(),
        args.with_relation.resolve(),
        args.without_relation.resolve(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "relation_audit.json", report)
    write_csv(args.out / "relation_audit.csv", report["per_query"])
    (args.out / "relation_audit.md").write_text(
        markdown(report),
        encoding="utf-8",
    )
    print(f"Wrote relation audit for {report['query_count']} queries to {args.out}")


if __name__ == "__main__":
    main()
