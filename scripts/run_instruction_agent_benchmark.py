#!/usr/bin/env python3
"""Run flat and hierarchical instruction-agent search on the 150-query track."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.graph_vs_flat import (
    filter_five_scene_queries,
    load_benchmark_queries,
    load_scene_bundle,
    resolve_scene_dir,
)


SCHEMA_VERSION = "semanticsplat.instruction_agent_benchmark.v1"
TRACE_SCHEMA_VERSION = "semanticsplat.instruction_agent_trace.v1"
ALL_METHODS = ("flat_instruction", "graph_instruction")
SYSTEM_PROMPT = (
    "Judge scene-query relevance using semantic meaning and affordances. "
    "Answer exactly Yes or No."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def compact_view(view_id: str, view: dict[str, Any]) -> str:
    parts = [
        str(view.get("summary") or view.get("scene_summary") or ""),
        str(view.get("room_type") or ""),
    ]
    for region in view.get("visible_regions", []):
        if isinstance(region, dict):
            parts.append(str(region.get("label") or ""))
    for obj in view.get("visible_objects", view.get("objects", [])):
        if not isinstance(obj, dict):
            continue
        label = str(obj.get("label") or "")
        parts.append(label)
    for landmark in view.get("landmarks", []):
        if isinstance(landmark, dict):
            parts.append(str(landmark.get("label") or ""))
    text = "; ".join(part.strip() for part in parts if part and part.strip())
    text = re.sub(r"\s+", " ", text)
    return f"{view_id}: {text[:280]}"


def root_id(tree: dict[str, dict[str, Any]]) -> str:
    return next(
        (
            key
            for key, value in tree.items()
            if str(value.get("node_type") or value.get("type")) == "root"
        ),
        "root",
    )


def zone_candidates(bundle: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    tree = bundle["tree"]
    root = tree[root_id(tree)]
    zones = []
    for node_id in root.get("children_ids", []):
        node = tree.get(str(node_id), {})
        view_ids = [str(value) for value in node.get("view_ids", []) if str(value) in bundle["views"]]
        if not view_ids:
            continue
        description = " ".join(
            str(node.get(key) or "") for key in ("name", "summary", "approx_location")
        ).strip()
        zones.append((str(node_id), re.sub(r"\s+", " ", description)[:420], view_ids))
    return zones


def alias_candidates(
    candidates: Iterable[tuple[str, str]],
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    rows = []
    aliases = {}
    for index, (candidate_id, description) in enumerate(candidates):
        alias = f"C{index}"
        rows.append((alias, candidate_id, description))
        aliases[alias] = candidate_id
    return rows, aliases


def parse_aliases(response: str, aliases: dict[str, str], limit: int) -> list[str]:
    ranked: list[str] = []
    for alias in re.findall(r"\bC\d+\b", response.upper()):
        candidate_id = aliases.get(alias)
        if candidate_id and candidate_id not in ranked:
            ranked.append(candidate_id)
        if len(ranked) >= limit:
            break
    return ranked


class InstructionAgent:
    def __init__(
        self,
        model_id: str,
        *,
        cache_dir: Path,
        threads: int,
        max_new_tokens: int,
    ) -> None:
        torch.manual_seed(20260724)
        torch.set_num_threads(max(1, threads))
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

    def call(
        self,
        *,
        phase: str,
        query: str,
        rows: list[tuple[str, str, str]],
        aliases: dict[str, str],
        limit: int,
    ) -> tuple[list[str], dict[str, Any]]:
        user_prompts = [
            (
                f"Query: {query}\n"
                f"Candidate {alias}: {description}\n"
                "Relevant? Answer exactly Yes or No."
            )
            for alias, _candidate_id, description in rows
        ]
        rendered_prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for user_prompt in user_prompts
        ]
        encoded = self.tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
        )
        input_tokens = int(encoded["attention_mask"].sum().item())
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        prompt_width = int(encoded["input_ids"].shape[-1])
        generated = output.sequences[:, prompt_width:]
        output_tokens = int(generated.numel())
        responses = [
            self.tokenizer.decode(value, skip_special_tokens=True).strip()
            for value in generated
        ]
        logits = output.scores[0]
        binary_logits = torch.stack(
            (logits[:, self.no_token_id], logits[:, self.yes_token_id]),
            dim=1,
        )
        yes_probabilities = torch.softmax(binary_logits, dim=1)[:, 1].tolist()
        scored = sorted(
            (
                (float(score), alias, aliases[alias], response)
                for (alias, _candidate_id, _description), score, response in zip(
                    rows,
                    yes_probabilities,
                    responses,
                    strict=True,
                )
            ),
            key=lambda value: (-value[0], value[1]),
        )
        ranked = [candidate_id for _score, _alias, candidate_id, _response in scored[:limit]]
        response = ",".join(alias for _score, alias, _candidate_id, _response in scored[:limit])
        call = {
            "phase": phase,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompts": user_prompts,
            "rendered_prompts": rendered_prompts,
            "raw_response": response,
            "candidate_scores": [
                {
                    "alias": alias,
                    "candidate_id": candidate_id,
                    "relevance_probability": round(score, 8),
                    "generated_decision": generated_response,
                }
                for score, alias, candidate_id, generated_response in scored
            ],
            "alias_map": aliases,
            "ranked_ids": ranked,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": round(latency_ms, 6),
            "token_method": f"transformers_native:{self.model_id}",
            "valid_response": len(ranked) == min(limit, len(rows)),
            "scoring_method": "instruction_yes_no_next_token_probability",
        }
        return ranked, call


def expected_views(query: dict[str, Any]) -> list[str]:
    return [str(value) for value in query.get("expected_view_ids", []) if value]


def reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    for index, value in enumerate(ranked, 1):
        if value in expected_set:
            return 1.0 / index
    return 0.0


def quality(ranked: list[str], query: dict[str, Any]) -> dict[str, Any]:
    expected = expected_views(query)
    eligible = bool(expected)
    return {
        "eligible": eligible,
        "expected_view_ids": expected,
        "hit_at_1": bool(ranked[:1] and ranked[0] in expected) if eligible else None,
        "hit_at_3": bool(set(ranked[:3]) & set(expected)) if eligible else None,
        "reciprocal_rank": reciprocal_rank(ranked, expected) if eligible else None,
    }


def run_method(
    agent: InstructionAgent,
    *,
    method: str,
    query: dict[str, Any],
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_text = str(query["query"])
    calls: list[dict[str, Any]] = []
    selected_zones: list[str] = []

    if method == "flat_instruction":
        candidate_views = sorted(bundle["views"])
    else:
        zones = zone_candidates(bundle)
        zone_rows, zone_aliases = alias_candidates(
            (zone_id, description) for zone_id, description, _view_ids in zones
        )
        selected_zones, zone_call = agent.call(
            phase="hierarchy_traversal",
            query=query_text,
            rows=zone_rows,
            aliases=zone_aliases,
            limit=2,
        )
        calls.append(zone_call)
        views_by_zone = {zone_id: view_ids for zone_id, _description, view_ids in zones}
        candidate_views = sorted(
            {
                view_id
                for zone_id in selected_zones
                for view_id in views_by_zone.get(zone_id, [])
            }
        )

    ranked: list[str] = []
    if candidate_views:
        view_rows, view_aliases = alias_candidates(
            (view_id, compact_view(view_id, bundle["views"][view_id]))
            for view_id in candidate_views
        )
        ranked, view_call = agent.call(
            phase="view_grounding",
            query=query_text,
            rows=view_rows,
            aliases=view_aliases,
            limit=3,
        )
        calls.append(view_call)

    accounting = {
        "model_call_count": len(calls),
        "input_tokens": sum(int(call["input_tokens"]) for call in calls),
        "output_tokens": sum(int(call["output_tokens"]) for call in calls),
        "total_tokens": sum(int(call["total_tokens"]) for call in calls),
        "latency_ms": round(sum(float(call["latency_ms"]) for call in calls), 6),
    }
    result = {
        "ranked_view_ids": ranked,
        "selected_view_id": ranked[0] if ranked else None,
        "found": bool(ranked),
        "selected_zone_ids": selected_zones,
        "candidate_view_ids": candidate_views,
        "views_checked": len(candidate_views),
        "response_valid": bool(calls) and all(bool(call["valid_response"]) for call in calls),
        "accounting": accounting,
        "quality": quality(ranked, query),
    }
    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "query_id": str(query["query_id"]),
        "scene_id": str(query["scene_id"]),
        "query": query_text,
        "method": method,
        "model": agent.model_id,
        "model_revision": agent.revision,
        "calls": calls,
        "result": result,
    }
    return result, trace


def optional_mean(values: Iterable[float | int | bool | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(mean(clean), 6) if clean else None


def method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    results = [row["methods"][method] for row in rows]
    quality_rows = [value["quality"] for value in results if value["quality"]["eligible"]]
    return {
        "query_count": len(results),
        "quality_denominator": len(quality_rows),
        "hit_at_1": optional_mean(value["hit_at_1"] for value in quality_rows),
        "hit_at_3": optional_mean(value["hit_at_3"] for value in quality_rows),
        "mrr": optional_mean(value["reciprocal_rank"] for value in quality_rows),
        "valid_response_rate": optional_mean(value["response_valid"] for value in results),
        "mean_views_checked": optional_mean(value["views_checked"] for value in results),
        "mean_model_calls": optional_mean(
            value["accounting"]["model_call_count"] for value in results
        ),
        "mean_input_tokens": optional_mean(
            value["accounting"]["input_tokens"] for value in results
        ),
        "mean_output_tokens": optional_mean(
            value["accounting"]["output_tokens"] for value in results
        ),
        "mean_total_tokens": optional_mean(
            value["accounting"]["total_tokens"] for value in results
        ),
        "total_input_tokens": sum(
            int(value["accounting"]["input_tokens"]) for value in results
        ),
        "total_output_tokens": sum(
            int(value["accounting"]["output_tokens"]) for value in results
        ),
        "total_tokens": sum(int(value["accounting"]["total_tokens"]) for value in results),
        "mean_latency_ms": optional_mean(
            value["accounting"]["latency_ms"] for value in results
        ),
        "total_model_calls": sum(
            int(value["accounting"]["model_call_count"]) for value in results
        ),
    }


def bootstrap_difference(
    rows: list[dict[str, Any]],
    field: str,
    *,
    samples: int = 10_000,
    seed: int = 20260724,
) -> dict[str, Any]:
    paired = [
        (
            row["methods"]["graph_instruction"]["quality"][field],
            row["methods"]["flat_instruction"]["quality"][field],
        )
        for row in rows
        if row["methods"]["flat_instruction"]["quality"]["eligible"]
    ]
    diffs = np.asarray([float(graph) - float(flat) for graph, flat in paired])
    rng = np.random.default_rng(seed)
    resampled = diffs[rng.integers(0, len(diffs), size=(samples, len(diffs)))].mean(axis=1)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return {
        "difference_graph_minus_flat": round(float(diffs.mean()), 6),
        "ci_95": [round(float(low), 6), round(float(high), 6)],
        "samples": samples,
        "seed": seed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "query_id",
        "scene_id",
        "query",
        "query_type",
        "method",
        "quality_eligible",
        "hit_at_1",
        "hit_at_3",
        "reciprocal_rank",
        "selected_view_id",
        "views_checked",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_ms",
        "response_valid",
        "trace_path",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_value in rows:
            for method in row_value["methods"]:
                result = row_value["methods"][method]
                q = result["quality"]
                a = result["accounting"]
                writer.writerow(
                    {
                        "query_id": row_value["query_id"],
                        "scene_id": row_value["scene_id"],
                        "query": row_value["query"],
                        "query_type": row_value["query_type"],
                        "method": method,
                        "quality_eligible": q["eligible"],
                        "hit_at_1": q["hit_at_1"],
                        "hit_at_3": q["hit_at_3"],
                        "reciprocal_rank": q["reciprocal_rank"],
                        "selected_view_id": result["selected_view_id"],
                        "views_checked": result["views_checked"],
                        "model_calls": a["model_call_count"],
                        "input_tokens": a["input_tokens"],
                        "output_tokens": a["output_tokens"],
                        "total_tokens": a["total_tokens"],
                        "latency_ms": a["latency_ms"],
                        "response_valid": result["response_valid"],
                        "trace_path": result["trace_path"],
                    }
                )


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Instruction-Agent Benchmark",
        "",
        f"- Model: `{summary['model']['model_id']}`",
        f"- Model revision: `{summary['model']['revision']}`",
        f"- Queries: {summary['query_count']} total / {summary['quality_query_count']} quality-eligible",
        f"- Device: `{summary['hardware']['device']}`",
        "",
        "| Method | hit@1 | hit@3 | MRR | Views | Calls | Input tokens | Output tokens | ms/query | Valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in summary["methods"]:
        value = summary["methods"][method]
        lines.append(
            f"| {method} | {value['hit_at_1']:.3f} | {value['hit_at_3']:.3f} | "
            f"{value['mrr']:.3f} | {value['mean_views_checked']:.2f} | "
            f"{value['mean_model_calls']:.2f} | {value['mean_input_tokens']:.1f} | "
            f"{value['mean_output_tokens']:.1f} | {value['mean_latency_ms']:.1f} | "
            f"{value['valid_response_rate']:.3f} |"
        )
    comparison = summary.get("comparison")
    if comparison:
        lines.extend(
            [
                "",
                f"- Graph input-token reduction: {comparison['input_token_reduction_pct']:.2f}%",
                f"- Graph views reduction: {comparison['views_reduction_pct']:.2f}%",
                f"- Paired hit@1 difference: {comparison['hit_at_1']['difference_graph_minus_flat']:.3f} "
                f"(95% CI {comparison['hit_at_1']['ci_95']})",
                f"- Paired hit@3 difference: {comparison['hit_at_3']['difference_graph_minus_flat']:.3f} "
                f"(95% CI {comparison['hit_at_3']['ci_95']})",
            ]
        )
    lines.extend(
        [
            "",
            "Token values are exact native tokenizer counts for the rendered chat prompts and generated responses.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = filter_five_scene_queries(load_benchmark_queries(args.benchmark.resolve()))
    if args.limit is not None:
        queries = queries[: args.limit]

    process = psutil.Process()
    agent = InstructionAgent(
        args.model,
        cache_dir=args.model_cache.resolve(),
        threads=args.threads,
        max_new_tokens=args.max_new_tokens,
    )
    bundles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    methods_to_run = tuple(args.methods)
    started_run = time.perf_counter()

    for index, query in enumerate(queries, 1):
        scene_id = str(query["scene_id"])
        if scene_id not in bundles:
            scene_dir = resolve_scene_dir(scene_id, args.scene_root.resolve())
            bundles[scene_id] = load_scene_bundle(scene_dir)
        bundle = bundles[scene_id]
        methods: dict[str, Any] = {}
        for method in methods_to_run:
            trace_path = (
                out_dir
                / "traces"
                / method
                / f"{scene_id}__{query['query_id']}.json"
            )
            if args.resume and trace_path.is_file():
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                if (
                    trace.get("model") != args.model
                    or trace.get("method") != method
                    or trace.get("query_id") != str(query["query_id"])
                ):
                    raise ValueError(f"resume trace does not match run: {trace_path}")
                result = dict(trace["result"])
            else:
                result, trace = run_method(agent, method=method, query=query, bundle=bundle)
                write_json(trace_path, trace)
            result["trace_path"] = str(trace_path.relative_to(ROOT)).replace("\\", "/")
            methods[method] = result
        rows.append(
            {
                "query_id": str(query["query_id"]),
                "scene_id": scene_id,
                "query": str(query["query"]),
                "query_type": str(query.get("query_type") or "unspecified"),
                "methods": methods,
            }
        )
        if index % 10 == 0 or index == len(queries):
            print(f"Completed {index}/{len(queries)} instruction-agent queries", flush=True)

    methods = {method: method_summary(rows, method) for method in methods_to_run}
    comparison = None
    if set(methods_to_run) == set(ALL_METHODS):
        flat = methods["flat_instruction"]
        graph = methods["graph_instruction"]
        comparison = {
            "input_token_reduction_pct": round(
                100 * (1 - graph["total_input_tokens"] / flat["total_input_tokens"]), 6
            ),
            "views_reduction_pct": round(
                100 * (1 - graph["mean_views_checked"] / flat["mean_views_checked"]), 6
            ),
            "hit_at_1": bootstrap_difference(rows, "hit_at_1"),
            "hit_at_3": bootstrap_difference(rows, "hit_at_3"),
            "mrr": bootstrap_difference(rows, "reciprocal_rank"),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "complete",
        "query_count": len(rows),
        "quality_query_count": sum(
            1
            for row in rows
            if row["methods"][methods_to_run[0]]["quality"]["eligible"]
        ),
        "model": {
            "model_id": args.model,
            "revision": agent.revision,
            "instruction_tuned": True,
            "framework": "transformers",
            "torch_version": torch.__version__,
            "load_seconds": round(agent.load_seconds, 6),
            "decision_tokens_per_candidate": 1,
            "temperature": 0,
            "do_sample": False,
        },
        "hardware": {
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor(),
            "threads": args.threads,
            "peak_process_rss_mb": round(process.memory_info().rss / (1024 * 1024), 3),
        },
        "benchmark": {
            "path": str(args.benchmark.resolve().relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(args.benchmark.resolve()),
        },
        "methods": methods,
        "comparison": comparison,
        "elapsed_seconds": round(time.perf_counter() - started_run, 6),
        "git_commit": git_commit(),
    }
    write_json(out_dir / "per_query_results.json", rows)
    write_json(out_dir / "metrics_summary.json", summary)
    write_json(
        out_dir / "run_config.json",
        {
            "schema_version": "semanticsplat.instruction_agent_run_config.v1",
            "model": args.model,
            "model_cache": "huggingface_default_cache",
            "benchmark": summary["benchmark"],
            "scene_root": str(args.scene_root.resolve().relative_to(ROOT)).replace("\\", "/"),
            "methods": list(methods_to_run),
            "system_prompt": SYSTEM_PROMPT,
            "seed": 20260724,
            "decision_tokens_per_candidate": 1,
            "threads": args.threads,
            "environment": {
                "python": platform.python_version(),
                "transformers": __import__("transformers").__version__,
                "torch": torch.__version__,
                "hf_home": os.getenv("HF_HOME"),
            },
        },
    )
    write_csv(out_dir / "per_query_metrics.csv", rows)
    (out_dir / "metrics_summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "benchmark_queries_v2.json",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "backend" / "data" / "scenes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs"
        / "instruction_agent"
        / "five_scene_qwen25_05b_v1",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
    )
    parser.add_argument("--threads", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--max-new-tokens", type=int, default=1, choices=(1,))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=["graph_instruction"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete matching per-method traces already present in --out.",
    )
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
