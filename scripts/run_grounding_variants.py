#!/usr/bin/env python3
"""Run fair object-grounding variants over one experiment config."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.grounding import build_candidates, search_grounding
from scripts.evaluate_grounding import evaluate, write_outputs
from scripts.run_experiment import canonical_result, load_scene


DEFAULT_RUNS = {
    "graph": {"variant": "graph"},
    "graph_fallback": {"variant": "graph_fallback"},
    "flat_lexical": {"variant": "flat_lexical"},
    "flat_embedding": {"variant": "flat_embedding"},
    "ablation_no_hierarchy": {"variant": "graph_fallback", "no_hierarchy": True},
    "ablation_no_relation": {"variant": "graph_fallback", "no_relation": True},
    "ablation_no_fallback": {"variant": "graph_fallback", "no_fallback": True},
    "ablation_flat_only": {
        "variant": "flat_lexical",
        "no_hierarchy": True,
        "no_relation": True,
        "no_fallback": True,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


class FastEmbedAdapter:
    """Small local CPU embedding backend with explicit model/cache provenance."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed import TextEmbedding

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            threads=4,
            cuda=False,
        )

    def encode(self, texts: Iterable[str], **_: Any) -> np.ndarray:
        return np.vstack(list(self.model.embed(list(texts))))


def scene_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = {
        str(item["scene_id"]): item
        for item in config.get("scenes", [])
        if isinstance(item, dict) and item.get("scene_id")
    }
    for scene_id in config.get("scene_ids", []):
        entries.setdefault(
            str(scene_id),
            {"scene_id": str(scene_id), "path": f"backend/data/scenes/{scene_id}"},
        )
    return entries


def checked_trace(
    candidates: list[Any],
    checked_ids: list[str],
    selected: Any | None,
) -> tuple[list[str], list[str], list[str]]:
    by_id = {candidate.object_id: candidate for candidate in candidates}
    checked = [by_id[item] for item in checked_ids if item in by_id]
    nodes = sorted(
        {
            node_id
            for candidate in checked
            for node_id in [*candidate.ancestor_node_ids, candidate.node_id]
            if node_id
        }
    )
    views = sorted({candidate.view_id for candidate in checked if candidate.view_id})
    traversal = (
        [*reversed(selected.ancestor_node_ids), selected.node_id]
        if selected is not None and selected.node_id
        else []
    )
    return nodes, views, [str(item) for item in traversal if item]


def run_variant(
    *,
    run_name: str,
    options: dict[str, Any],
    config: dict[str, Any],
    benchmark: dict[str, Any],
    output_root: Path,
    embedder: Any | None,
    embedding_model: str,
    embedding_cache: Path,
    limit: int | None,
) -> dict[str, Any]:
    run_dir = output_root / run_name
    results_dir = run_dir / "query_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    entries = scene_entries(config)
    bundles: dict[str, dict[str, Any]] = {}
    construction_seconds = 0.0
    queries = [
        item
        for item in benchmark.get("queries", [])
        if isinstance(item, dict) and str(item.get("scene_id")) in entries
    ]
    if limit is not None:
        queries = queries[:limit]

    for query in queries:
        scene_id = str(query["scene_id"])
        if scene_id not in bundles:
            loaded = load_scene(resolve(entries[scene_id]["path"]))
            started = time.perf_counter()
            candidates = build_candidates(loaded["existing_views"], loaded["existing_tree"])
            construction_seconds += time.perf_counter() - started
            bundles[scene_id] = {
                "views": loaded["existing_views"],
                "tree": loaded["existing_tree"],
                "candidates": candidates,
            }
        bundle = bundles[scene_id]
        started = time.perf_counter()
        search = search_grounding(
            bundle,
            str(query["query"]),
            embedder=embedder if options["variant"] == "flat_embedding" else None,
            embedding_model=embedding_model,
            embedding_cache=str(embedding_cache),
            **options,
        )
        runtime = time.perf_counter() - started
        selected = search.candidates[0] if search.status == "ok" and search.candidates else None
        checked_ids = list(
            search.metadata.get("same_input_candidate_ids", [])
            if options.get("no_hierarchy") or options["variant"].startswith("flat_")
            else search.metadata.get("initial_candidate_ids", [])
        )
        if search.fallback_triggered:
            checked_ids = list(search.metadata.get("same_input_candidate_ids", []))
        checked_nodes, checked_views, traversal = checked_trace(
            bundle["candidates"], checked_ids, selected
        )
        context_chars = sum(
            len(candidate.text)
            for candidate in bundle["candidates"]
            if candidate.object_id in set(checked_ids)
        )
        relation_score = (
            selected.score_details.get("relation_score") if selected is not None else None
        )
        ranked_candidates = [
            {
                "rank": rank,
                "object_id": candidate.object_id,
                "label": candidate.label,
                "score": candidate.score,
                "relation_confidence": candidate.score_details.get("relation_confidence"),
                "relation_applied": candidate.score_details.get("relation_applied"),
            }
            for rank, candidate in enumerate(search.candidates, start=1)
        ]
        answer = {
            "structured_plan": {
                "target": search.parsed_query.target,
                "relation": search.parsed_query.relation,
                "anchor": search.parsed_query.anchor,
                "anchor_secondary": search.parsed_query.anchor_secondary,
            },
            "visited_nodes": traversal,
            "traversal_path": traversal,
            "selected_views": [selected.view_id] if selected is not None else [],
            "checked_node_ids": checked_nodes,
            "checked_view_ids": checked_views,
            "checked_object_ids": checked_ids,
            "checked_object_count": search.searched_candidate_count,
            "search_variant": run_name,
            "fallback_used": search.fallback_triggered,
            "fallback_reason": ", ".join(search.fallback_reasons) or None,
            "result": {
                "found": selected is not None,
                "matched_object": selected.label if selected is not None else None,
                "selected_object_id": selected.object_id if selected is not None else None,
                "selected_node_id": selected.node_id if selected is not None else None,
                "selected_view_id": selected.view_id if selected is not None else None,
                "ranked_object_ids": [
                    candidate.object_id for candidate in search.candidates
                ],
                "ranked_candidates": ranked_candidates,
                "bbox_2d": list(selected.bbox_2d) if selected and selected.bbox_2d else None,
                "bbox_3d": selected.bbox_3d if selected is not None else None,
                "camera_pose": None,
                "confidence": selected.score if selected is not None else 0.0,
                "search_score": selected.score if selected is not None else None,
                "score_details": selected.score_details if selected is not None else {},
                "relation_satisfied": (
                    bool(selected.score_details.get("relation_applied"))
                    if relation_score is not None and selected is not None
                    else None
                ),
                "relation_audit": search.metadata.get("relation_audit"),
                "explanation": (
                    f"{run_name} selected object {selected.object_id}."
                    if selected is not None
                    else f"{run_name} produced status={search.status}."
                ),
            },
            "warnings": (
                [f"Embedding unavailable: {search.metadata.get('embedding')}"]
                if search.status == "blocked"
                else []
            ),
        }
        result = canonical_result(
            run_id=f"{config.get('run_id', 'grounding')}_{run_name}",
            method="semantic_splat_grounding",
            mode="stub",
            query=query,
            answer=answer,
            runtime_seconds=runtime,
            usage={},
            context_metrics={
                "context_size_chars": context_chars,
                "prompt_size_chars": context_chars + len(str(query["query"])),
                "estimated_input_tokens": round(
                    (context_chars + len(str(query["query"]))) / 4
                ),
                "estimated_token_method": "chars_div_4",
            },
            depth_validation={"status": "reliable", "reason": "official dataset GT boxes"},
            geometry_eval_allowed=bool(entries[scene_id].get("geometry_eval_allowed", True)),
        )
        if search.status == "blocked":
            result["availability_status"] = "unavailable"
            result["availability_reason"] = "embedding_backend_unavailable"
        write_json(results_dir / f"{query['query_id']}.json", result)

    run_config = {
        "schema_version": "semanticsplat.grounding_variant_run.v1",
        "run_name": run_name,
        "search_options": options,
        "fixed_search_policy": {
            "top_k": 10,
            "confidence_threshold": 0.45,
            "branch_keep_ratio": 0.5,
            "anchor_confidence_threshold": 0.55,
            "anchor_margin_threshold": 0.10,
            "relation_confidence_threshold": 0.65,
            "relation_score_margin": 0.15,
            "relation_boost": 0.20,
        },
        "benchmark_path": str(resolve(config["benchmark_path"])),
        "query_count": len(queries),
        "scene_ids": sorted(bundles),
        "construction_cost": {
            "runtime_seconds": round(construction_seconds, 6),
            "candidate_count": sum(len(bundle["candidates"]) for bundle in bundles.values()),
            "scope": "one-time canonical candidate construction",
        },
        "embedding": (
            {
                "backend": "fastembed",
                "model": embedding_model,
                "cache_dir": str(embedding_cache),
                "device": "cpu",
            }
            if options["variant"] == "flat_embedding"
            else None
        ),
    }
    write_json(run_dir / "run_config.json", run_config)
    summary = evaluate(
        resolve(config["benchmark_path"]),
        results_dir,
        run_config_path=run_dir / "run_config.json",
    )
    write_outputs(summary, run_dir)
    return summary


def write_comparison(output_root: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows = []
    for name, summary in summaries.items():
        metrics = summary["metrics"]
        efficiency = metrics["efficiency"]
        rows.append(
            {
                "variant": name,
                "recall_at_1": metrics["recall_at_1_exact_object_id"]["value"],
                "recall_at_3": metrics["recall_at_3_exact_object_id"]["value"],
                "recall_at_5": metrics["recall_at_5_exact_object_id"]["value"],
                "mrr": metrics["mrr_exact_object_id"]["value"],
                "negative_accuracy": metrics["negative_accuracy"]["value"],
                "acc_at_0_1": metrics["acc_at_0_1"]["value"],
                "acc_at_0_25": metrics["acc_at_0_25"]["value"],
                "acc_at_0_5": metrics["acc_at_0_5"]["value"],
                "avg_runtime_seconds": efficiency["runtime_seconds"]["mean"],
                "avg_checked_nodes": efficiency["checked_nodes"]["mean"],
                "avg_checked_views": efficiency["checked_views"]["mean"],
                "avg_checked_objects": efficiency["checked_objects"]["mean"],
                "avg_estimated_input_tokens": efficiency["estimated_tokens"]["mean"],
            }
        )
    write_json(output_root / "variant_comparison.json", {"variants": rows})
    with (output_root / "variant_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Grounding variant comparison", ""]
    if rows:
        lines.extend(
            [
                "| Variant | R@1 | R@3 | R@5 | MRR | Neg. acc. | Acc@0.25 | Checked objects | Runtime (s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['variant']} | {row['recall_at_1']} | {row['recall_at_3']} | "
                f"{row['recall_at_5']} | {row['mrr']} | {row['negative_accuracy']} | "
                f"{row['acc_at_0_25']} | "
                f"{row['avg_checked_objects']} | {row['avg_runtime_seconds']} |"
            )
    (output_root / "variant_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run", action="append", choices=sorted(DEFAULT_RUNS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / ".cache" / "fastembed")
    args = parser.parse_args()

    config = read_json(args.config.resolve())
    benchmark = read_json(resolve(config["benchmark_path"]))
    run_names = args.run or list(DEFAULT_RUNS)
    embedder = None
    if "flat_embedding" in run_names:
        embedder = FastEmbedAdapter(args.embedding_model, args.embedding_cache.resolve())
    summaries = {
        name: run_variant(
            run_name=name,
            options=DEFAULT_RUNS[name],
            config=config,
            benchmark=benchmark,
            output_root=args.out.resolve(),
            embedder=embedder,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache.resolve(),
            limit=args.limit,
        )
        for name in run_names
    }
    write_comparison(args.out.resolve(), summaries)
    print(f"Wrote {len(summaries)} grounding runs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
