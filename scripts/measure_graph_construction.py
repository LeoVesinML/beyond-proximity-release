#!/usr/bin/env python3
"""Measure deterministic graph construction over the five captured scenes.

The builder does not call a language-model provider. This script records that
measured zero-token fact separately from serialized-text token equivalents and
from the saved manual-annotation artifact workload.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_scene_tree_from_viewjson import build_scene_tree  # noqa: E402


DEFAULT_SCENES = (
    "ConferenceHall-capture-pilot",
    "Museume-capture",
    "Theater-capture",
    "outdoor-street-capture",
    "outdoor-drone-capture",
)


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_size(paths: list[Path]) -> dict[str, int]:
    chars = 0
    token_equivalent = 0
    for path in paths:
        text = canonical_json(json.loads(path.read_text(encoding="utf-8")))
        chars += len(text)
        token_equivalent += len(text) // 4
    return {
        "record_count": len(paths),
        "serialized_characters": chars,
        "estimated_tokens_chars_div_4": token_equivalent,
    }


def checked_in_node_ids(scene_dir: Path) -> set[str]:
    manifest_path = scene_dir / "tree" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(node_id) for node_id in manifest.get("node_ids", [])}


def measure_scene(scene_id: str, repeats: int) -> dict[str, Any]:
    scene_dir = ROOT / "backend" / "data" / "scenes" / scene_id
    views_dir = scene_dir / "views"
    if not scene_dir.is_dir() or not views_dir.is_dir():
        raise FileNotFoundError(f"Missing captured scene or ViewJSON directory: {scene_dir}")

    view_files = sorted(views_dir.glob("*.json"))
    source_size = record_size(view_files)
    durations: list[float] = []
    output_size: dict[str, int] | None = None
    manifest: dict[str, Any] | None = None
    rebuilt_node_ids: set[str] = set()

    for _ in range(repeats):
        with tempfile.TemporaryDirectory(prefix=f"semanticsplat-{scene_id}-") as tmp:
            output_dir = Path(tmp) / "tree"
            started = time.perf_counter()
            current_manifest = build_scene_tree(scene_dir, views_dir, output_dir)
            durations.append(time.perf_counter() - started)
            if manifest is None:
                manifest = current_manifest
                node_files = sorted(output_dir.glob("node_*.json"))
                output_size = record_size(node_files)
                rebuilt_node_ids = {
                    str(json.loads(path.read_text(encoding="utf-8")).get("node_id"))
                    for path in node_files
                }

    assert manifest is not None and output_size is not None
    expected_ids = checked_in_node_ids(scene_dir)
    return {
        "scene_id": scene_id,
        "view_count": int(manifest["view_count"]),
        "semantic_item_count": int(manifest["semantic_item_count"]),
        "node_count": int(manifest["node_count"]),
        "source_viewjson": source_size,
        "constructed_tree_nodes": output_size,
        "runtime_seconds": {
            "repeats": repeats,
            "median": round(statistics.median(durations), 6),
            "minimum": round(min(durations), 6),
            "maximum": round(max(durations), 6),
        },
        "checked_in_tree_node_ids_match": rebuilt_node_ids == expected_ids,
    }


def build_report(scene_ids: list[str], repeats: int) -> dict[str, Any]:
    scenes = [measure_scene(scene_id, repeats) for scene_id in scene_ids]
    totals = {
        "scene_count": len(scenes),
        "view_count": sum(scene["view_count"] for scene in scenes),
        "semantic_item_count": sum(scene["semantic_item_count"] for scene in scenes),
        "node_count": sum(scene["node_count"] for scene in scenes),
        "source_viewjson_serialized_characters": sum(
            scene["source_viewjson"]["serialized_characters"] for scene in scenes
        ),
        "source_viewjson_estimated_tokens_chars_div_4": sum(
            scene["source_viewjson"]["estimated_tokens_chars_div_4"] for scene in scenes
        ),
        "constructed_tree_serialized_characters": sum(
            scene["constructed_tree_nodes"]["serialized_characters"] for scene in scenes
        ),
        "constructed_tree_estimated_tokens_chars_div_4": sum(
            scene["constructed_tree_nodes"]["estimated_tokens_chars_div_4"] for scene in scenes
        ),
        "median_runtime_seconds_sum": round(
            sum(scene["runtime_seconds"]["median"] for scene in scenes), 6
        ),
    }
    totals["serialized_io_characters_total"] = (
        totals["source_viewjson_serialized_characters"]
        + totals["constructed_tree_serialized_characters"]
    )
    totals["serialized_io_estimated_tokens_chars_div_4"] = (
        totals["source_viewjson_estimated_tokens_chars_div_4"]
        + totals["constructed_tree_estimated_tokens_chars_div_4"]
    )
    return {
        "schema_version": "semanticsplat.graph_construction_cost.v2",
        "scope": "five captured manual-ViewJSON pilot scenes",
        "builder": "scripts/build_scene_tree_from_viewjson.py",
        "provider_usage": {
            "model_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reason": "The measured builder is deterministic local Python and has no provider call path.",
        },
        "annotation_usage": {
            "measurement_basis": "saved manual ViewJSON artifacts",
            "viewjson_record_count": totals["view_count"],
            "semantic_item_count": totals["semantic_item_count"],
            "canonical_serialized_characters": totals["source_viewjson_serialized_characters"],
            "estimated_tokens_chars_div_4": totals[
                "source_viewjson_estimated_tokens_chars_div_4"
            ],
            "scope_note": (
                "This meters the produced annotation payload. Historical annotator wall-clock "
                "time was not logged and is not inferred."
            ),
        },
        "token_equivalent_definition": (
            "For each canonical compact JSON record, floor(serialized characters / 4), then sum. "
            "This is a local representation-size proxy, not provider billing usage."
        ),
        "runtime_definition": (
            "Wall-clock rebuild in a fresh temporary output directory; median/min/max over repeats."
        ),
        "totals": totals,
        "scenes": scenes,
    }


def markdown_report(report: dict[str, Any]) -> str:
    totals = report["totals"]
    provider = report["provider_usage"]
    lines = [
        "# Graph Construction Cost",
        "",
        "This audit rebuilds the five captured-scene semantic graphs from their saved manual ViewJSON records.",
        "",
        "## Result",
        "",
        f"- Provider model calls: **{provider['model_call_count']}**",
        f"- Provider construction tokens: **{provider['total_tokens']}** input + output tokens",
        f"- Views: **{totals['view_count']}**",
        f"- Semantic items: **{totals['semantic_item_count']}**",
        f"- Constructed nodes: **{totals['node_count']}**",
        f"- Manual annotation records: **{totals['view_count']}**",
        f"- Manual semantic items: **{totals['semantic_item_count']}**",
        f"- Manual annotation characters: **{totals['source_viewjson_serialized_characters']:,}** canonical serialized characters",
        f"- Source ViewJSON size: **{totals['source_viewjson_estimated_tokens_chars_div_4']:,}** estimated tokens (characters/4 proxy)",
        f"- Constructed tree size: **{totals['constructed_tree_estimated_tokens_chars_div_4']:,}** estimated tokens (characters/4 proxy)",
        f"- Total serialized build I/O: **{totals['serialized_io_estimated_tokens_chars_div_4']:,}** estimated tokens (input + output)",
        f"- Sum of per-scene median local build times: **{totals['median_runtime_seconds_sum']:.6f} s**",
        "",
        "Provider tokens and serialized token-equivalents are different quantities. Provider usage is an instrumented zero because the graph builder is local and deterministic. Manual annotation workload is metered from the frozen artifacts as record count, semantic-item count, canonical characters, and token-equivalent payload. Historical annotator wall-clock time was not logged, so no person-hour value is inferred.",
        "",
        "## Per Scene",
        "",
        "| Scene | Views | Items | Nodes | Source est. tokens | Tree est. tokens | Median build (s) | Node IDs match |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for scene in report["scenes"]:
        lines.append(
            "| {scene_id} | {view_count} | {semantic_item_count} | {node_count} | {source:,} | {tree:,} | {runtime:.6f} | {match} |".format(
                scene_id=scene["scene_id"],
                view_count=scene["view_count"],
                semantic_item_count=scene["semantic_item_count"],
                node_count=scene["node_count"],
                source=scene["source_viewjson"]["estimated_tokens_chars_div_4"],
                tree=scene["constructed_tree_nodes"]["estimated_tokens_chars_div_4"],
                runtime=scene["runtime_seconds"]["median"],
                match="yes" if scene["checked_in_tree_node_ids_match"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python -B scripts\\measure_graph_construction.py --repeats 7",
            "```",
            "",
            f"Definition: {report['token_equivalent_definition']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", dest="scenes", help="Captured scene ID; repeat as needed")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "docs" / "reports" / "final" / "graph_construction_cost.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=ROOT / "docs" / "reports" / "final" / "graph_construction_cost.md",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    report = build_report(args.scenes or list(DEFAULT_SCENES), args.repeats)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["totals"], indent=2))
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
