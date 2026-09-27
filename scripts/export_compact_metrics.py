#!/usr/bin/env python3
"""Export a public-safe aggregate metrics snapshot without per-query GT IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def compact_metrics(data: dict[str, Any]) -> dict[str, Any]:
    required = (
        "generated_at",
        "run_id",
        "benchmark_id",
        "coverage",
        "mode_counts",
        "per_source_dataset",
        "metrics",
        "failure_categories",
        "warnings",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("Metrics summary is missing required fields: " + ", ".join(missing))
    return {
        "schema_version": "semanticsplat.public_metrics_snapshot.v1",
        "source_schema_version": data.get("schema_version"),
        "generated_at": data["generated_at"],
        "run_id": data["run_id"],
        "benchmark_id": data["benchmark_id"],
        "evidence_scope": (
            "Aggregate metrics only; licensed ScanNet-derived per-query GT IDs, boxes, and "
            "utterances remain in the local generated benchmark and run outputs."
        ),
        "coverage": data["coverage"],
        "mode_counts": data["mode_counts"],
        "per_source_dataset": data["per_source_dataset"],
        "metrics": data["metrics"],
        "failure_categories": data["failure_categories"],
        "warnings": data["warnings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compact_metrics(read_json(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote aggregate public metrics snapshot to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
