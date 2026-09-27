#!/usr/bin/env python3
"""LLM cascade experiments for the IEEE revision (A2 and B10).

A2: Qwen Yes/No relevance scoring (prompt, candidate serialization and p(Yes)
normalization reused verbatim from scripts/run_instruction_agent_benchmark.py)
applied to (i) all views, (ii) flat BGE top-N_q with N_q = |C(q)|, (iii) the
Algorithm 1 candidate set C(q) built from the BGE proxy, plus the LLM-only
two-branch gate rerun on the same model.

B10: listwise answer selection. The model receives the query and a numbered
list of serialized records and must output one record ID.

The BGE proxy is BAAI/bge-small-en-v1.5 via the FastEmbed ONNX export
(qdrant/bge-small-en-v1.5-onnx-q). FastEmbed itself is not installed on this
machine, so the ONNX graph is executed directly with onnxruntime using the
FastEmbed preprocessing (tokenizer.json, truncation 512, CLS pooling, L2
normalisation). The script verifies that this reproduces the saved BGE
frontier outputs before running any LLM condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.query.graph_vs_flat import (  # noqa: E402
    DEFAULT_SCENE_MAP,
    filter_five_scene_queries,
    load_benchmark_queries,
    load_scene_bundle,
    resolve_scene_dir,
)
from backend.query.recall_controlled import (  # noqa: E402
    PreparedScene,
    RecallControlledConfig,
    _choose_branches,
    _soft_branch_views,
    paired_bootstrap_difference,
    prepare_scene,
)
from scripts.run_instruction_agent_benchmark import (  # noqa: E402
    SYSTEM_PROMPT,
    alias_candidates,
    compact_view,
    zone_candidates,
)

SEED = 20260901
BOOTSTRAP_SAMPLES = 10_000
SPLIT_PATH = ROOT / "outputs/ieee_revision/recall_frontier_v1/split.json"
BGE_FRONTIER = ROOT / "outputs/ieee_revision/bge_frontier_v1"
DEFAULT_BGE_DIR = Path(
    "/Users/leo/beyond-proximity-week6-person2/.cache/fastembed/"
    "models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/"
    "52398278842ec682c6f32300af41344b1c0b0bb2"
)

ANSWER_SYSTEM_PROMPT = (
    "You select the single scene record that best answers a query. "
    "Answer with the record ID only."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- BGE proxy
class OnnxBGE:
    """FastEmbed-equivalent BGE-small ONNX encoder (CLS pooling + L2 norm)."""

    def __init__(self, model_dir: Path, threads: int = 4) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = model_dir
        config = json.loads((model_dir / "config.json").read_text())
        tok_config = json.loads((model_dir / "tokenizer_config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        max_length = min(int(tok_config.get("model_max_length", 512)), 512)
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(
            pad_id=int(config.get("pad_token_id", 0)), pad_token=str(tok_config["pad_token"])
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_dir / "model_optimized.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {value.name for value in self.session.get_inputs()}
        self.onnx_sha256 = hashlib.sha256((model_dir / "model_optimized.onnx").read_bytes()).hexdigest()

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode_batch([str(t) for t in texts[start : start + batch_size]])
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)
            output = self.session.run(None, feeds)[0]
            chunks.append(output[:, 0].astype(np.float32))
        vectors = np.vstack(chunks)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


def _rank(scores: dict[str, float]) -> list[tuple[float, str]]:
    # identical to scripts/run_ieee_bge_frontier.py::_rank
    return sorted(((float(score), key) for key, score in scores.items()), reverse=True)


def _softmax(ranked: list[tuple[float, str]], temperature: float) -> dict[str, float]:
    import math

    maximum = ranked[0][0] if ranked else 0.0
    weights = {item_id: math.exp((score - maximum) / temperature) for score, item_id in ranked}
    total = sum(weights.values()) or 1.0
    return {item_id: value / total for item_id, value in weights.items()}


def bge_select(
    scene: PreparedScene,
    view_matrix: np.ndarray,
    branch_matrix: np.ndarray,
    vector: np.ndarray,
    config: RecallControlledConfig,
    expected: set[str],
) -> dict[str, Any]:
    """Algorithm 1 with the BGE proxy; mirrors run_ieee_bge_frontier._search."""
    view_values = view_matrix @ vector
    view_scores = {view_id: float(view_values[pos]) for pos, view_id in enumerate(scene.view_ids)}
    ranked_global = _rank(view_scores)
    memberships = _soft_branch_views(scene, config.boundary_epsilon)
    if config.branch_score_mode == "node_text":
        branch_values = branch_matrix @ vector
        branch_scores = {b: float(branch_values[pos]) for pos, b in enumerate(scene.branch_ids)}
    else:
        branch_scores = {}
        for branch_id in scene.branch_ids:
            values = sorted((view_scores[v] for v in memberships[branch_id]), reverse=True)
            branch_scores[branch_id] = (
                values[0] if config.branch_score_mode == "max_leaf_proxy" else float(np.mean(values[:3]))
            ) if values else -1.0
    ranked_branches = _rank(branch_scores)
    probabilities = _softmax(ranked_branches, config.score_temperature)
    selected = _choose_branches(scene, ranked_branches, probabilities, config, expected, memberships)
    candidates = set().union(*(memberships[b] for b in selected)) if selected else set()
    candidates.update(v for _, v in ranked_global[: config.safety_shortlist_size])
    ranked_candidates = [(s, v) for s, v in ranked_global if v in candidates]
    if config.candidate_budget is not None:
        ranked_candidates = ranked_candidates[: config.candidate_budget]
    return {
        "selected_branches": selected,
        "candidate_ids_bge_order": [v for _, v in ranked_candidates],
        "global_bge_order": [v for _, v in ranked_global],
    }


# ---------------------------------------------------------------- metrics
def rr(ranked: list[str], expected: set[str]) -> float:
    for index, value in enumerate(ranked, 1):
        if value in expected:
            return 1.0 / index
    return 0.0


def quality(ranked: list[str], expected: set[str]) -> dict[str, Any]:
    return {
        "hit_at_1": bool(ranked[:1] and ranked[0] in expected),
        "hit_at_3": bool(set(ranked[:3]) & expected),
        "mrr": rr(ranked, expected),
        "mrr_at_3": rr(ranked[:3], expected),
        "target_in_candidates": bool(set(ranked) & expected),
    }


# ---------------------------------------------------------------- Qwen
class QwenScorer:
    def __init__(self, model_path: str, model_id: str, dtype: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.manual_seed(20260724)
        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, dtype), low_cpu_mem_usage=True, attn_implementation="sdpa"
        ).to(device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        self.generation_config_repr = self.model.generation_config.to_dict()

    def _sync(self) -> None:
        if self.device == "mps":
            self.torch.mps.synchronize()

    def score(self, query: str, rows: list[tuple[str, str, str]], aliases: dict[str, str]) -> dict[str, Any]:
        """One batched call; prompt and p(Yes) exactly as InstructionAgent.call."""
        torch = self.torch
        started = time.perf_counter()
        user_prompts = [
            (
                f"Query: {query}\n"
                f"Candidate {alias}: {description}\n"
                "Relevant? Answer exactly Yes or No."
            )
            for alias, _candidate_id, description in rows
        ]
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in user_prompts
        ]
        # Right padding: on MPS (torch 2.7.1) left padding + SDPA yields NaN for
        # padded rows (0 * NaN from fully masked pad queries). With right padding
        # and causal attention every real position only attends to real tokens,
        # so reading the logits at each row's last real token equals the first
        # generation step of the original left-padded generate(max_new_tokens=1).
        encoded = self.tokenizer(rendered, return_tensors="pt", padding=True, padding_side="right")
        input_tokens = int(encoded["attention_mask"].sum().item())
        yes_no_in_prompt = bool(
            ((encoded["input_ids"] == self.yes_token_id) | (encoded["input_ids"] == self.no_token_id)).any().item()
        )
        lengths = encoded["attention_mask"].sum(dim=1)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        gen_started = time.perf_counter()
        with torch.inference_mode():
            hidden = self.model.model(**encoded).last_hidden_state
            last = hidden[torch.arange(hidden.shape[0], device=hidden.device), (lengths - 1).to(hidden.device)]
            full_logits = self.model.lm_head(last)
            greedy = full_logits.argmax(dim=-1)
            weight = self.model.lm_head.weight
            # binary Yes/No logits recomputed in float32 from the last hidden state
            yes_no = last.float() @ weight[[self.no_token_id, self.yes_token_id]].float().T
            probs = torch.softmax(yes_no, dim=1)[:, 1].cpu().tolist()
            # log-odds of the same two-way softmax; monotone in p(Yes) but does not
            # saturate at 1.0 in float32, so it is used as the ranking key.
            margins = (yes_no[:, 1] - yes_no[:, 0]).cpu().tolist()
            nan_count = int(torch.isnan(yes_no).any(dim=1).sum().item())
            greedy = greedy.cpu()
        self._sync()
        generate_ms = (time.perf_counter() - gen_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        generated = greedy.view(-1, 1)
        responses = [self.tokenizer.decode(v, skip_special_tokens=True).strip() for v in generated]
        scored = sorted(
            (
                (float(mg), float(pr), alias, aliases[alias], r)
                for (alias, _cid, _d), pr, mg, r in zip(rows, probs, margins, responses)
            ),
            key=lambda value: (-value[0], value[2]),
        )
        scored = [(pr, alias, cid, r, mg) for mg, pr, alias, cid, r in scored]
        return {
            "ranked_ids": [cid for _s, _a, cid, _r, _m in scored],
            "candidate_scores": [
                {"alias": a, "candidate_id": cid, "relevance_probability": round(s, 8),
                 "yes_minus_no_logit": round(m, 5), "generated_decision": r}
                for s, a, cid, r, m in scored
            ],
            "prompt_sha256": [sha256_text(p) for p in rendered],
            "prompt_count": len(rendered),
            "input_tokens": input_tokens,
            "output_tokens": int(generated.numel()),
            "forward_ms": round(generate_ms, 4),
            "llm_ms": round(total_ms, 4),
            "yes_no_token_in_prompt": yes_no_in_prompt,
            "nan_rows": nan_count,
        }

    def answer(
        self, query: str, records: list[tuple[str, str]], max_new_tokens: int = 12, constrained: bool = False
    ) -> dict[str, Any]:
        """B10 listwise single-ID selection (greedy)."""
        torch = self.torch
        started = time.perf_counter()
        lines = [f"[{i}] {description}" for i, (_vid, description) in enumerate(records, 1)]
        user = (
            f"Query: {query}\n"
            "Records:\n" + "\n".join(lines) + "\n"
            "Which single record best answers the query? "
            "Output only its record ID (the token before the colon, for example "
            f"{records[0][0]})."
        )
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer([rendered], return_tensors="pt")
        input_tokens = int(encoded["attention_mask"].sum().item())
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        extra: dict[str, Any] = {}
        prompt_width = int(encoded["input_ids"].shape[-1])
        if constrained:
            # Greedy decoding restricted to a trie of the valid record-ID token
            # sequences followed by <|im_end|>.
            eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            seqs = [self.tokenizer.encode(vid, add_special_tokens=False) + [eos] for vid, _ in records]

            def allowed(_batch_id: int, ids: Any) -> list[int]:
                prefix = ids[prompt_width:].tolist()
                nxt = {seq[len(prefix)] for seq in seqs if len(seq) > len(prefix) and seq[: len(prefix)] == prefix}
                return sorted(nxt) or [eos]

            extra = {"prefix_allowed_tokens_fn": allowed}
            max_new_tokens = max(len(seq) for seq in seqs)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **extra,
            )
        self._sync()
        latency_ms = (time.perf_counter() - started) * 1000
        width = int(encoded["input_ids"].shape[-1])
        gen = output[0, width:].cpu()
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        valid_ids = [vid for vid, _ in records]
        chosen = None
        for token in re.findall(r"[A-Za-z]*\d+", text):
            if token in valid_ids:
                chosen = token
                break
        if chosen is None:
            match = re.search(r"\[?(\d+)\]?", text)
            if match and 1 <= int(match.group(1)) <= len(records) and not text.lower().startswith("v"):
                chosen = valid_ids[int(match.group(1)) - 1]
        return {
            "record_ids_in_prompt": valid_ids,
            "prompt_sha256": sha256_text(rendered),
            "decoding": "trie_constrained_greedy" if constrained else "free_greedy_parsed",
            "raw_output": text,
            "chosen_id": chosen,
            "valid": chosen is not None,
            "input_tokens": input_tokens,
            "output_tokens": int(gen.numel()),
            "latency_ms": round(latency_ms, 4),
        }


# ---------------------------------------------------------------- data
def load_data(benchmark: Path, scene_root: Path):
    queries = filter_five_scene_queries(load_benchmark_queries(benchmark))
    by_id = {str(q["query_id"]): q for q in queries}
    split = json.loads(SPLIT_PATH.read_text())
    bundles, scenes = {}, {}
    for sid in sorted(DEFAULT_SCENE_MAP):
        bundle = load_scene_bundle(resolve_scene_dir(sid, scene_root))
        bundles[sid] = bundle
        scenes[sid] = prepare_scene(bundle)
    return by_id, split, bundles, scenes


def verify_bge(encoder: OnnxBGE, by_id, split, scenes, config: RecallControlledConfig) -> dict[str, Any]:
    """Reproduce saved BGE frontier rows (budget_3 operating point and unlimited)."""
    saved_op = {r["query_id"]: r for r in json.loads((BGE_FRONTIER / "operating_point_per_query.json").read_text())}
    saved_frontier = json.loads((BGE_FRONTIER / "frontier_per_query.json").read_text())
    saved_unl = {r["query_id"]: r for r in saved_frontier if r["frontier_label"] == "budget_unlimited"}
    mism_op, mism_unl, mism_unl_set, flat_hits = [], [], [], []
    for qid in split["test"]:
        q = by_id[qid]
        scene = scenes[str(q["scene_id"])]
        vm, bm, qv = ENC_CACHE["views"][str(q["scene_id"])], ENC_CACHE["branches"][str(q["scene_id"])], ENC_CACHE["queries"][qid]
        expected = {str(v) for v in q.get("expected_view_ids", []) if v}
        op = bge_select(scene, vm, bm, qv, replace(config, candidate_budget=3), expected)
        unl = bge_select(scene, vm, bm, qv, replace(config, candidate_budget=None), expected)
        if op["candidate_ids_bge_order"] != saved_op[qid]["ranked_view_ids"]:
            mism_op.append(qid)
        if unl["candidate_ids_bge_order"] != saved_unl[qid]["ranked_view_ids"]:
            mism_unl.append(qid)
        if set(unl["candidate_ids_bge_order"]) != set(saved_unl[qid]["ranked_view_ids"]):
            mism_unl_set.append(qid)
        flat_hits.append(quality(unl["global_bge_order"], expected))
    summary = json.loads((BGE_FRONTIER / "bge_frontier_summary.json").read_text())
    return {
        "test_queries": len(split["test"]),
        "budget_3_ranked_list_mismatches": mism_op,
        "budget_unlimited_ranked_list_order_mismatches": mism_unl,
        "budget_unlimited_candidate_set_mismatches": mism_unl_set,
        "flat_bge_test_hit_at_1_recomputed": round(statistics.fmean(h["hit_at_1"] for h in flat_hits), 6),
        "flat_bge_test_hit_at_3_recomputed": round(statistics.fmean(h["hit_at_3"] for h in flat_hits), 6),
        "flat_bge_test_hit_at_1_saved": summary["flat_test"]["hit_at_1"],
        "flat_bge_test_hit_at_3_saved": summary["flat_test"]["hit_at_3"],
        "saved_final_config": summary["tuning"]["final_config"],
        "saved_operating_label": summary["decision"]["selected_frontier_label"],
    }


ENC_CACHE: dict[str, dict[str, np.ndarray]] = {}


def build_encodings(encoder: OnnxBGE, by_id, scenes) -> None:
    ENC_CACHE["views"] = {sid: encoder.encode([s.view_texts[v] for v in s.view_ids]) for sid, s in scenes.items()}
    ENC_CACHE["branches"] = {sid: encoder.encode([s.branch_texts[b] for b in s.branch_ids]) for sid, s in scenes.items()}
    ids = sorted(by_id)
    matrix = encoder.encode([str(by_id[i]["query"]) for i in ids])
    ENC_CACHE["queries"] = {qid: matrix[pos] for pos, qid in enumerate(ids)}


def dev_config() -> RecallControlledConfig:
    summary = json.loads((BGE_FRONTIER / "bge_frontier_summary.json").read_text())
    return RecallControlledConfig(**summary["tuning"]["final_config"])


# ---------------------------------------------------------------- A2
def a2_query(scorer: QwenScorer, encoder: OnnxBGE, q, bundle, scene, config, conditions) -> dict[str, Any]:
    qid, text = str(q["query_id"]), str(q["query"])
    expected = {str(v) for v in q.get("expected_view_ids", []) if v}
    # proxy timing (single-query encode + selection), for end-to-end latency
    t0 = time.perf_counter()
    qv = encoder.encode([text])[0]
    t_enc = (time.perf_counter() - t0) * 1000
    vm, bm = ENC_CACHE["views"][str(q["scene_id"])], ENC_CACHE["branches"][str(q["scene_id"])]
    t1 = time.perf_counter()
    alg1 = bge_select(scene, vm, bm, qv, replace(config, candidate_budget=None), expected)
    t_sel = (time.perf_counter() - t1) * 1000
    alg1_maxleaf = bge_select(
        scene, vm, bm, qv, replace(config, candidate_budget=None, branch_score_mode="max_leaf_proxy"), expected
    )
    alg1_b3 = bge_select(scene, vm, bm, qv, replace(config, candidate_budget=3), expected)
    flat_order = alg1["global_bge_order"]
    n_q = len(alg1["candidate_ids_bge_order"])
    proxy_ms = t_enc + t_sel

    sets = {
        "exhaustive": sorted(bundle["views"]),
        "flat_bge_topN_matched": sorted(flat_order[:n_q]),
        "alg1_bge": sorted(alg1["candidate_ids_bge_order"]),
        "alg1_bge_maxleaf": sorted(alg1_maxleaf["candidate_ids_bge_order"]),
        "alg1_bge_budget3": sorted(alg1_b3["candidate_ids_bge_order"]),
        "flat_bge_top3": sorted(flat_order[:3]),
    }
    out: dict[str, Any] = {
        "query_id": qid,
        "scene_id": str(q["scene_id"]),
        "query": text,
        "query_type": q.get("query_type"),
        "expected_view_ids": sorted(expected),
        "proxy": {
            "alg1_selected_branches": alg1["selected_branches"],
            "alg1_candidate_ids_bge_order": alg1["candidate_ids_bge_order"],
            "alg1_maxleaf_selected_branches": alg1_maxleaf["selected_branches"],
            "alg1_maxleaf_candidate_ids_bge_order": alg1_maxleaf["candidate_ids_bge_order"],
            "alg1_budget3_ids_bge_order": alg1_b3["candidate_ids_bge_order"],
            "flat_bge_order": flat_order,
            "n_q": n_q,
            "query_encode_ms": round(t_enc, 4),
            "selection_ms": round(t_sel, 4),
        },
        "conditions": {},
    }
    for name in conditions:
        if name == "gate_llm":
            zones = zone_candidates(bundle)
            zrows, zal = alias_candidates((z, d) for z, d, _v in zones)
            zcall = scorer.score(text, zrows, zal)
            chosen = zcall["ranked_ids"][:2]
            views_by_zone = {z: v for z, _d, v in zones}
            cand = sorted({v for z in chosen for v in views_by_zone.get(z, [])})
            vrows, val = alias_candidates((v, compact_view(v, bundle["views"][v])) for v in cand)
            vcall = scorer.score(text, vrows, val)
            ranked = vcall["ranked_ids"]
            calls = [dict(zcall, phase="hierarchy_traversal"), dict(vcall, phase="view_grounding")]
            extra_ms = 0.0
            candidates = cand
        else:
            candidates = sets[name]
            rows, al = alias_candidates((v, compact_view(v, bundle["views"][v])) for v in candidates)
            call = scorer.score(text, rows, al)
            ranked = call["ranked_ids"]
            calls = [dict(call, phase="view_grounding")]
            extra_ms = 0.0 if name == "exhaustive" else proxy_ms
        llm_ms = sum(c["llm_ms"] for c in calls)
        out["conditions"][name] = {
            "candidate_ids": candidates,
            "n_candidates": len(candidates),
            "ranked_ids": ranked,
            "quality": quality(ranked, expected),
            "quality_top3": quality(ranked[:3], expected),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "llm_calls": len(calls),
            "llm_prompts": sum(c["prompt_count"] for c in calls),
            "llm_ms": round(llm_ms, 4),
            "proxy_ms": round(extra_ms, 4),
            "end_to_end_ms": round(llm_ms + extra_ms, 4),
            "calls": calls,
        }
    return out


# ---------------------------------------------------------------- B10
def b10_query(scorer: QwenScorer, q, bundle, scene, config) -> dict[str, Any]:
    qid, text = str(q["query_id"]), str(q["query"])
    expected = {str(v) for v in q.get("expected_view_ids", []) if v}
    vm, bm, qv = ENC_CACHE["views"][str(q["scene_id"])], ENC_CACHE["branches"][str(q["scene_id"])], ENC_CACHE["queries"][qid]
    alg1_b3 = bge_select(scene, vm, bm, qv, replace(config, candidate_budget=3), expected)
    flat3 = alg1_b3["global_bge_order"][:3]
    sets = {
        "all_records": sorted(bundle["views"]),
        "alg1_bge_top3": sorted(alg1_b3["candidate_ids_bge_order"]),
        "flat_bge_top3": sorted(flat3),
    }
    out = {
        "query_id": qid, "scene_id": str(q["scene_id"]), "query": text,
        "query_type": q.get("query_type"), "expected_view_ids": sorted(expected), "conditions": {},
    }
    for name, ids in sets.items():
        records = [(v, compact_view(v, bundle["views"][v])) for v in ids]
        for suffix, constrained in (("", False), ("__constrained", True)):
            result = scorer.answer(text, records, constrained=constrained)
            result["correct"] = bool(result["chosen_id"] in expected)
            result["target_in_context"] = bool(set(ids) & expected)
            out["conditions"][name + suffix] = result
    return out


# ---------------------------------------------------------------- summaries
def summarize_a2(rows: list[dict[str, Any]], ids: list[str], conditions: list[str]) -> dict[str, Any]:
    sel = [r for r in rows if r["query_id"] in set(ids)]
    res: dict[str, Any] = {"query_count": len(sel)}
    for name in conditions:
        for variant in ("full", "top3"):
            key = "quality" if variant == "full" else "quality_top3"
            c = [r["conditions"][name] for r in sel]
            lat = [x["llm_ms"] for x in c]
            e2e = [x["end_to_end_ms"] for x in c]
            res[f"{name}__{variant}"] = {
                "hit_at_1": round(statistics.fmean(x[key]["hit_at_1"] for x in c), 6),
                "hit_at_3": round(statistics.fmean(x[key]["hit_at_3"] for x in c), 6),
                "mrr": round(statistics.fmean(x[key]["mrr"] for x in c), 6),
                "mrr_at_3": round(statistics.fmean(x[key]["mrr_at_3"] for x in c), 6),
                "target_in_candidates": round(statistics.fmean(x["quality"]["target_in_candidates"] for x in c), 6),
                "mean_candidates_scored": round(statistics.fmean(x["n_candidates"] for x in c), 6),
                "mean_retained_records": round(
                    statistics.fmean(x["n_candidates"] if variant == "full" else min(3, x["n_candidates"]) for x in c), 6
                ),
                "mean_input_tokens": round(statistics.fmean(x["input_tokens"] for x in c), 3),
                "mean_llm_calls": round(statistics.fmean(x["llm_calls"] for x in c), 3),
                "mean_llm_prompts": round(statistics.fmean(x["llm_prompts"] for x in c), 3),
                "median_llm_ms": round(statistics.median(lat), 3),
                "mean_llm_ms": round(statistics.fmean(lat), 3),
                "median_end_to_end_ms": round(statistics.median(e2e), 3),
                "mean_end_to_end_ms": round(statistics.fmean(e2e), 3),
            }
    comparisons = {}
    pairs = [
        ("alg1_bge__full", "exhaustive__full"),
        ("alg1_bge__full", "flat_bge_topN_matched__full"),
        ("alg1_bge__top3", "exhaustive__top3"),
        ("alg1_bge__top3", "flat_bge_topN_matched__top3"),
        ("alg1_bge_budget3__full", "flat_bge_top3__full"),
        ("alg1_bge__full", "gate_llm__full"),
    ]
    for left, right in pairs:
        ln, lv = left.split("__")
        rn, rv = right.split("__")
        if ln not in conditions or rn not in conditions:
            continue
        lk = "quality" if lv == "full" else "quality_top3"
        rk = "quality" if rv == "full" else "quality_top3"
        comp = {}
        for metric in ("hit_at_1", "hit_at_3", "mrr"):
            a = [float(r["conditions"][ln][lk][metric]) for r in sel]
            b = [float(r["conditions"][rn][rk][metric]) for r in sel]
            comp[metric] = paired_bootstrap_difference(a, b, samples=BOOTSTRAP_SAMPLES, seed=SEED)
        comparisons[f"{left} - {right}"] = comp
    res["paired_bootstrap"] = comparisons
    return res


B10_CONDITIONS = tuple(
    n + sfx for sfx in ("", "__constrained") for n in ("all_records", "alg1_bge_top3", "flat_bge_top3")
)


def summarize_b10(rows: list[dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    sel = [r for r in rows if r["query_id"] in set(ids)]
    res: dict[str, Any] = {"query_count": len(sel)}
    for name in B10_CONDITIONS:
        c = [r["conditions"][name] for r in sel]
        lat = [x["latency_ms"] for x in c]
        res[name] = {
            "answer_accuracy": round(statistics.fmean(x["correct"] for x in c), 6),
            "valid_rate": round(statistics.fmean(x["valid"] for x in c), 6),
            "target_in_context": round(statistics.fmean(x["target_in_context"] for x in c), 6),
            "mean_records_in_context": round(statistics.fmean(len(x["record_ids_in_prompt"]) for x in c), 3),
            "mean_input_tokens": round(statistics.fmean(x["input_tokens"] for x in c), 3),
            "median_latency_ms": round(statistics.median(lat), 3),
            "mean_latency_ms": round(statistics.fmean(lat), 3),
        }
    comps = {}
    pairs = (("alg1_bge_top3", "all_records"), ("alg1_bge_top3", "flat_bge_top3"), ("flat_bge_top3", "all_records"))
    for sfx in ("", "__constrained"):
        for base_left, base_right in pairs:
            left, right = base_left + sfx, base_right + sfx
            a = [float(r["conditions"][left]["correct"]) for r in sel]
            b = [float(r["conditions"][right]["correct"]) for r in sel]
            comps[f"{left} - {right}"] = paired_bootstrap_difference(a, b, samples=BOOTSTRAP_SAMPLES, seed=SEED)
    res["paired_bootstrap_accuracy"] = comps
    return res


def saved_gate_15b(ids_by_split: dict[str, list[str]]) -> dict[str, Any]:
    """(iv) Recompute the saved Qwen2.5-1.5B LLM-only gate / flat numbers on split subsets."""
    path = ROOT / "outputs/ieee_revision/qwen25_15b_pair_v1/per_query_results.json"
    rows = {r["query_id"]: r for r in json.loads(path.read_text())}
    meta = json.loads((ROOT / "outputs/ieee_revision/qwen25_15b_pair_v1/metrics_summary.json").read_text())
    out: dict[str, Any] = {
        "source": str(path.relative_to(ROOT)),
        "model_revision": meta["model"]["revision"],
        "hardware": meta["hardware"],
        "torch": meta["model"].get("torch_version"),
        "note": "saved view_grounding call kept only the top-3 ranked IDs, so MRR here is MRR@3",
    }
    for split_name, ids in ids_by_split.items():
        block = {}
        for method in ("flat_instruction", "graph_instruction"):
            c = [rows[i]["methods"][method] for i in ids]
            lat = [x["accounting"]["latency_ms"] for x in c]
            block[method] = {
                "n": len(c),
                "hit_at_1": round(statistics.fmean(bool(x["quality"]["hit_at_1"]) for x in c), 6),
                "hit_at_3": round(statistics.fmean(bool(x["quality"]["hit_at_3"]) for x in c), 6),
                "mrr_at_3": round(statistics.fmean(float(x["quality"]["reciprocal_rank"]) for x in c), 6),
                "mean_views_checked": round(statistics.fmean(x["views_checked"] for x in c), 6),
                "mean_input_tokens": round(statistics.fmean(x["accounting"]["input_tokens"] for x in c), 3),
                "mean_model_calls": round(statistics.fmean(x["accounting"]["model_call_count"] for x in c), 3),
                "median_latency_ms_cpu": round(statistics.median(lat), 3),
                "mean_latency_ms_cpu": round(statistics.fmean(lat), 3),
            }
        comp = {}
        for metric, key in (("hit_at_1", "hit_at_1"), ("hit_at_3", "hit_at_3"), ("mrr_at_3", "reciprocal_rank")):
            a = [float(rows[i]["methods"]["graph_instruction"]["quality"][key]) for i in ids]
            b = [float(rows[i]["methods"]["flat_instruction"]["quality"][key]) for i in ids]
            comp[metric] = paired_bootstrap_difference(a, b, samples=BOOTSTRAP_SAMPLES, seed=SEED)
        block["paired_bootstrap_graph_minus_flat"] = comp
        out[split_name] = block
    return out



# ---------------------------------------------------------------- report
A2_LABELS = {
    "exhaustive": "(i) exhaustive Qwen, all views",
    "flat_bge_topN_matched": "(ii) flat BGE top-N_q -> Qwen (N_q=|C(q)|)",
    "alg1_bge": "(iii) Alg. 1 (BGE) C(q) -> Qwen",
    "gate_llm": "(iv') LLM-only two-branch gate, rerun",
    "alg1_bge_maxleaf": "sens. Alg. 1 max_leaf_proxy C(q) -> Qwen",
    "alg1_bge_budget3": "extra: Alg. 1 BGE budget-3 -> Qwen",
    "flat_bge_top3": "extra: flat BGE top-3 -> Qwen",
}


def _ci(block: dict[str, Any]) -> str:
    return f"{block['estimate']:+.3f} [{block['ci_95']['low']:+.3f}, {block['ci_95']['high']:+.3f}]"


def build_report(out: Path, tag: str) -> None:
    a2 = json.loads((out / tag / "summary_a2.json").read_text())
    b10 = json.loads((out / tag / "summary_b10.json").read_text())
    saved = json.loads((out / "saved_qwen25_15b_gate_by_split.json").read_text())
    check = json.loads((out / "bge_reproduction_check.json").read_text())
    combined = {
        "schema_version": "beyond_proximity.ieee_llm_cascade.summary.v1",
        "generated_at": utc_now(),
        "bge_reproduction_check": check,
        "a2": a2,
        "b10": b10,
        "saved_qwen25_15b_gate_by_split": saved,
    }
    write_json(out / "summary.json", combined)
    lines: list[str] = []
    for split_name in ("test75", "all125"):
        res = a2["results"][split_name]
        lines += [f"### A2 {a2['model']['model_id']} ({a2['model']['dtype']}, MPS) -- {split_name}", "",
                  "| Condition | k | hit@1 | hit@3 | MRR | MRR@3 | target in C | retained | scored/query | input tok/query | calls/query | median ms | mean ms | median e2e ms |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in a2["conditions"]:
            for variant in ("full", "top3"):
                if variant == "top3" and name not in ("flat_bge_topN_matched", "alg1_bge", "exhaustive"):
                    continue
                v = res[f"{name}__{variant}"]
                lines.append(
                    f"| {A2_LABELS.get(name, name)} | {'all' if variant == 'full' else '3'} | {v['hit_at_1']:.3f} | {v['hit_at_3']:.3f} | "
                    f"{v['mrr']:.3f} | {v['mrr_at_3']:.3f} | {v['target_in_candidates']:.3f} | {v['mean_retained_records']:.2f} | "
                    f"{v['mean_candidates_scored']:.2f} | {v['mean_input_tokens']:.1f} | {v['mean_llm_calls']:.2f} | "
                    f"{v['median_llm_ms']:.0f} | {v['mean_llm_ms']:.0f} | {v['median_end_to_end_ms']:.0f} |"
                )
        lines += ["", "Paired bootstrap (10,000 resamples, seed 20260901), estimate [95% CI]:", "",
                  "| Comparison | hit@1 | hit@3 | MRR |", "|---|---:|---:|---:|"]
        for comp, block in res["paired_bootstrap"].items():
            lines.append(f"| {comp} | {_ci(block['hit_at_1'])} | {_ci(block['hit_at_3'])} | {_ci(block['mrr'])} |")
        lines.append("")
    for split_name in ("test75", "all125"):
        blk = saved[split_name]
        lines += [f"### (iv) Saved Qwen2.5-1.5B traces (CPU fp32, Windows) restricted to {split_name}", "",
                  "| Method | hit@1 | hit@3 | MRR@3 | views checked | input tok/query | calls/query | median ms (CPU) | mean ms (CPU) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method in ("flat_instruction", "graph_instruction"):
            v = blk[method]
            lines.append(f"| {method} | {v['hit_at_1']:.3f} | {v['hit_at_3']:.3f} | {v['mrr_at_3']:.3f} | {v['mean_views_checked']:.2f} | "
                         f"{v['mean_input_tokens']:.1f} | {v['mean_model_calls']:.2f} | {v['median_latency_ms_cpu']:.0f} | {v['mean_latency_ms_cpu']:.0f} |")
        c = blk["paired_bootstrap_graph_minus_flat"]
        lines += ["", f"graph - flat: hit@1 {_ci(c['hit_at_1'])}; hit@3 {_ci(c['hit_at_3'])}; MRR@3 {_ci(c['mrr_at_3'])}", ""]
    for split_name in ("test75", "all125"):
        res = b10["results"][split_name]
        lines += [f"### B10 answer selection, {b10['model']['model_id']} ({b10['model']['dtype']}, MPS) -- {split_name}", "",
                  "| Context | accuracy | valid ID rate | target in context | records | input tok/query | median ms | mean ms |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in B10_CONDITIONS:
            v = res[name]
            lines.append(f"| {name} | {v['answer_accuracy']:.3f} | {v['valid_rate']:.3f} | {v['target_in_context']:.3f} | "
                         f"{v['mean_records_in_context']:.2f} | {v['mean_input_tokens']:.1f} | {v['median_latency_ms']:.0f} | {v['mean_latency_ms']:.0f} |")
        lines += ["", "| Comparison | accuracy difference [95% CI] |", "|---|---:|"]
        for comp, block in res["paired_bootstrap_accuracy"].items():
            lines.append(f"| {comp} | {_ci(block)} |")
        lines.append("")
    (out / "RESULTS_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("report tables written")


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "docs/benchmarks/benchmark_queries_v2.json")
    parser.add_argument("--scene-root", type=Path, default=ROOT / "backend/data/scenes")
    parser.add_argument("--bge-dir", type=Path, default=DEFAULT_BGE_DIR)
    parser.add_argument("--model-path", required=False)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--experiment", choices=("verify", "a2", "b10", "saved_gate", "report"), required=True)
    parser.add_argument("--tag", required=False, help="output subdirectory, e.g. qwen25_7b")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ieee_revision2/llm_cascade")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--conditions", nargs="+",
        default=["exhaustive", "flat_bge_topN_matched", "alg1_bge", "alg1_bge_maxleaf",
                 "alg1_bge_budget3", "flat_bge_top3", "gate_llm"],
    )
    args = parser.parse_args()

    by_id, split, bundles, scenes = load_data(args.benchmark.resolve(), args.scene_root.resolve())
    test_ids, dev_ids = list(split["test"]), list(split["dev"])
    all_ids = sorted(test_ids + dev_ids)
    ids_by_split = {"test75": test_ids, "all125": all_ids}

    if args.experiment == "report":
        build_report(args.out, args.tag)
        return
    if args.experiment == "saved_gate":
        write_json(args.out / "saved_qwen25_15b_gate_by_split.json", saved_gate_15b(ids_by_split))
        print("saved gate summary written")
        return

    encoder = OnnxBGE(args.bge_dir)
    build_encodings(encoder, by_id, scenes)
    config = dev_config()
    verification = verify_bge(encoder, by_id, split, scenes, config)
    if args.experiment == "verify":
        write_json(args.out / "bge_reproduction_check.json", {
            **verification, "bge_onnx_sha256": encoder.onnx_sha256, "bge_dir": str(args.bge_dir),
            "generated_at": utc_now(),
        })
        print(json.dumps(verification, indent=2))
        return
    if verification["budget_3_ranked_list_mismatches"] or verification["budget_unlimited_candidate_set_mismatches"]:
        raise SystemExit(f"BGE reproduction failed: {verification}")

    import torch
    import transformers

    scorer = QwenScorer(args.model_path, args.model_id, args.dtype, args.device)
    out_dir = args.out / args.tag
    trace_dir = out_dir / f"traces_{args.experiment}"
    run_ids = all_ids[: args.limit] if args.limit else all_ids

    # warm-up (not timed / not saved)
    warm_q = by_id[run_ids[0]]
    warm_bundle = bundles[str(warm_q["scene_id"])]
    rows_w, al_w = alias_candidates((v, compact_view(v, warm_bundle["views"][v])) for v in sorted(warm_bundle["views"]))
    for _ in range(2):
        scorer.score("warm-up query", rows_w, al_w)
    if args.experiment == "b10":
        scorer.answer("warm-up query", [(v, d) for _a, v, d in rows_w])
        scorer.answer("warm-up query", [(v, d) for _a, v, d in rows_w], constrained=True)

    started = time.perf_counter()
    rows = []
    for index, qid in enumerate(run_ids, 1):
        trace_path = trace_dir / f"{qid}.json"
        if args.resume and trace_path.is_file():
            rows.append(json.loads(trace_path.read_text()))
            continue
        q = by_id[qid]
        sid = str(q["scene_id"])
        if args.experiment == "a2":
            row = a2_query(scorer, encoder, q, bundles[sid], scenes[sid], config, args.conditions)
        else:
            row = b10_query(scorer, q, bundles[sid], scenes[sid], config)
        row["split"] = "test" if qid in set(test_ids) else "dev"
        row["model"] = args.model_id
        row["model_revision"] = args.model_revision
        write_json(trace_path, row)
        rows.append(row)
        if index % 10 == 0 or index == len(run_ids):
            print(f"[{args.experiment}] {index}/{len(run_ids)} elapsed {time.perf_counter() - started:.1f}s", flush=True)

    ids_present = {r["query_id"] for r in rows}
    splits_present = {k: [i for i in v if i in ids_present] for k, v in ids_by_split.items()}
    if args.experiment == "a2":
        results = {k: summarize_a2(rows, v, args.conditions) for k, v in splits_present.items() if v}
    else:
        results = {k: summarize_b10(rows, v) for k, v in splits_present.items() if v}
    summary = {
        "schema_version": f"beyond_proximity.ieee_llm_cascade.{args.experiment}.v1",
        "generated_at": utc_now(),
        "experiment": args.experiment,
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "local_path": args.model_path,
            "dtype": args.dtype,
            "device": args.device,
            "load_seconds": round(scorer.load_seconds, 3),
            "generation_config": scorer.generation_config_repr,
            "yes_token_id": scorer.yes_token_id,
            "no_token_id": scorer.no_token_id,
        },
        "bge": {"onnx_dir": str(args.bge_dir), "onnx_sha256": encoder.onnx_sha256, "verification": verification},
        "alg1_config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "conditions": args.conditions if args.experiment == "a2" else list(B10_CONDITIONS),
        "query_ids_run": len(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }
    if args.experiment == "a2":
        summary["yes_no_token_ever_in_prompt"] = any(
            c["yes_no_token_in_prompt"] for r in rows for cond in r["conditions"].values() for c in cond["calls"]
        )
    write_json(out_dir / f"summary_{args.experiment}.json", summary)
    print(json.dumps(results, indent=1)[:4000])


if __name__ == "__main__":
    main()
