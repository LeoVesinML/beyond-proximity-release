# AGENTS.md - SemanticSplat

This is the operating guide for Codex, Cursor, and future agents working in this
repository. The goal is to keep every change evidence-based, reproducible, and
honest about what the project has actually proven.

Status date: 2026-07-24.

---

## 1. Project Identity

Project name: Beyond Proximity: Graph-Pruned Semantic Search over 3D Scene Records.

Current role of the system: a reproducible semantic-map evaluation prototype
with complete local instruction-agent and raw RGB-D hierarchy experiments, not
a completed main-conference quantitative result.

Basic idea:

```text
captured RGB-D views / visual observations
  -> ViewJSON annotations or automatic semantic extraction
  -> semantic index
  -> scene tree / semantic graph
  -> query pipeline
  -> result JSON
  -> evaluation reports
```

The project adds a semantic understanding layer over captured 3D/visual scenes.
It should help a system answer:

```text
What is around me?
Where should I go?
What does this place mean?
What changed in the environment?
```

It does not replace VPS, localization, AR tracking, or reconstruction. It adds
meaning, hierarchy, and query-time reasoning above existing map context.

Safest current claim:

```text
SemanticSplat provides a reproducible, test-backed evaluation prototype over
five captured RGB-D pilot scenes, plus official Replica and ScanNet
oracle-map retrieval tracks. The 150-query deterministic control reduces
checked views by 75.5% and estimated context by 68.2%, but loses hit@k. A
complete local Qwen2.5-0.5B-Instruct run executes all 150 queries with 150
traces, 300 model calls, 239,893 exact native tokens, hit@1 0.368, hit@3 0.696,
and 9.63 checked views/query. A separate CLIP ViT-B/32 constructor builds
hierarchies from all 97 RGB images, depth arrays, and poses with zero manual
ViewJSON or zone reads during construction. It reaches macro pairwise F1 0.487
against manual overlapping zones and checks 10.0 views/query with hit@3 0.528,
versus flat CLIP at 19.4 views and hit@3 0.624.

Calibrated v3 oracle-map pilots preserve flat lexical Acc@0.25 while reducing
checked objects: Replica reaches 1.000 on 48 positives; ScanNet reaches 0.604
on 48 Nr3D/Sr3D+ queries. ConceptGraphs has native eight-scene ScanNet and
eight-scene Replica predicted-map runs; LangSplat has a hardware-adapted
end-to-end one-scene ScanNet run. These prove execution and quantify quality
gaps; they do not prove automatic-map, LLM, or external-baseline superiority.
```

---

## 2. Current Evidence

Strong completed evidence:

```text
5 captured scenes
97 RGB-D captured views
97 manual ViewJSON annotations
1,066 semantic items
5 semantic indexes
5 generated trees
150 five-scene graph-vs-flat v2 benchmark queries
125 verified-view-label quality queries
graph ablation study outputs
graph scaling stress outputs
LangSplat ScanNet end-to-end output: 1 scene, 6 queries, view hit 0.6667
ConceptGraphs ScanNet output: 8 scenes, 48 queries, Acc@0.25 0.0625
ConceptGraphs Replica sampled-map output: 8 scenes, 56 queries, Acc@0.25 0.0625
50 default-scene stub benchmark outputs
8 official Replica scenes, 575 GT boxes, 56 GT-backed queries
8 official ScanNet scenes, 39 selected RGB-D views, 392 GT boxes
699 mapped Nr3D rows and 661 mapped unique Sr3D+ triplets
48-query ScanNet pilot with saved object-ID, bbox, runtime, context, and token metrics
145 passing backend tests as of the 2026-07-15 baseline/article check
150 complete Qwen2.5 instruction-agent queries with prompt/call/token traces
97-view raw RGB-D/pose CLIP hierarchy with zero manual construction reads
```

Current scene set:

```text
backend/data/scenes/ConferenceHall-capture-pilot
backend/data/scenes/Museume-capture
backend/data/scenes/Theater-capture
backend/data/scenes/outdoor-street-capture
backend/data/scenes/outdoor-drone-capture
backend/data/scenes/replica_*
backend/data/scenes/scannet_*
```

Important: `Museume-capture` is intentionally spelled this way in current data
and docs unless the project renames it globally.

Current blocked or out-of-scope work:

```text
live model evaluation = out of scope for the next phase
cached_live = out of scope for the next phase; no verified live cache exists
ConceptGraphs = ScanNet and Replica native public-data runs completed; protocol differs from SemanticSplat oracle-map pilots
LangSplat = reduced-resource one-scene ScanNet end-to-end run completed; eight-scene/paper-resource comparison not run
Replica = official BBQ-aligned object-box pilot completed
ScanNet = official eight-scene RGB-D/object-box and Nr3D/Sr3D+ pilot completed
semantic segmentation accuracy = unavailable without independent model predictions
3D IoU = measured for oracle-GT-map object retrieval; not predicted localization
```

---

## 3. Next-Phase Direction

Do not rebuild the project from scratch. The repo already has:

- legacy graph-vs-flat benchmark code on `default`;
- a query pipeline and Query Flow UI;
- spatial graph helpers;
- five captured semantic indexes with 1,066 semantic items;
- stub evaluation infrastructure and baseline adapters.

The next technical milestone is:

```text
Improve the current graph/search pipeline with explicit target-anchor spatial
reasoning, calibrated pruning, or fallback expansion. Then run fair same-scene
baselines over the completed Replica/ScanNet inputs and identical queries.
```

The main target claim to prove next is:

```text
Graph-based semantic reasoning scans fewer entries and uses less context/tokens
than flat search over the same semantic map. The current lexical implementation
does not yet preserve flat-search hit@k; the next research target is to recover
quality with calibrated pruning while keeping most of the cost reduction.
```

This claim is not proven until saved benchmark outputs include matching methods,
matching scenes, matching queries, context/token counts, runtime metrics, and GT
or manually verified expected IDs for quality metrics.

---

## 4. Repository Map

High-value paths:

```text
backend/                  backend API, query code, scene data
backend/query/            model client, stub/live/cached query logic
backend/geometry/         spatial graph and clustering helpers
backend/tree/             tree node/storage types
backend/data/scenes/      captured scenes, images, depths, views, trees

frontend/                 React UI / viewer / Query Flow
scripts/                  experiment, validation, import, adapters
configs/                  experiment configs
tests/                    backend tests

docs/project/             status, acceptance, plans, final project claims
docs/benchmarks/          benchmark queries, evaluation protocol/design
docs/schemas/             result/ViewJSON/schema docs
docs/datasets/            scene inventory, capture/annotation docs
docs/validation/          capture/geometry/semantic validation evidence
docs/baselines/           ConceptGraphs/LangSplat status and blockers
docs/experiments/         stub/live/cached_live/semantic retrieval notes
docs/reports/             week reports and final reports
docs/maintenance/         cleanup/audit docs

outputs/week2/            Week 2 stub/evaluation outputs
outputs/week3/            Week 3 final gate, live attempts, baseline logs
outputs/baselines/        baseline smoke outputs
papers/                   local PDFs and article/presentation assets
papers/beyond-proximity/  current LaTeX article draft
```

---

## 5. Read First

Before substantial changes, inspect these if present:

```text
docs/maintenance/plan_vs_project_audit.md
docs/project/final_week3_acceptance.md
docs/project/week3_results.md
docs/project/real_evaluation_status.md
docs/project/chatgpt_project_description.md
docs/benchmarks/evaluation_protocol.md
docs/benchmarks/benchmark_queries_v1.json
docs/datasets/scene_inventory.md
docs/baselines/baseline_status.md
semantic_splat_research_direction(1).md
plan2.md
paper-publication-plan.md
```

If a file moved, search by name or keywords with `rg` rather than guessing.

---

## 6. Truth Rules

Do not fake or overclaim results.

Never claim:

- live model evaluation succeeded unless there is at least one real `mode: live`
  output;
- `cached_live` succeeded unless replay used a verified live cache and avoided
  provider calls;
- ConceptGraphs or LangSplat ran unless their official/native command produced
  output or a logged execution attempt exists;
- semantic accuracy unless there is independent Ground Truth;
- 3D IoU/localization unless GT boxes/masks and predicted boxes exist;
- Replica/ScanNet results unless official dataset scenes are ingested and
  evaluated;
- automatic/headless pipeline if manual browser capture, manual ViewJSON, or
  manual tree building was required;
- live/cached_live is in scope unless the user explicitly reopens that scope;
- algorithm superiority without a fair same-input benchmark.

Use these labels exactly:

```text
stub = deterministic local mode, no real model call
live = real provider/API model call
cached_live = replay of verified real live output
manual semantic index = human/Cursor-assisted annotation, not independent GT
GT = Ground Truth, expected correct answer used for accuracy
```

Without GT, report only:

```text
schema validity
availability
coverage
runtime
mode
model_call_count
cache_hit_count
context/token counts if measured
```

Use `N/A` for:

```text
semantic accuracy
negative correctness
3D IoU
provider-backed token/latency/cost
baseline quality comparison
```

---

## 7. Metrics And Benchmark Rules

GT means expected correct answer.

For a query such as:

```text
Where is the staircase?
```

GT should include at least:

```json
{
  "expected_views": ["v001"],
  "expected_objects": ["stairs"],
  "expected_regions": ["stair area"],
  "expected_answer_contains": ["staircase", "left side"]
}
```

For graph-vs-flat, compare methods on the same scenes, same semantic items, same
query strings, and same GT labels.

Required efficiency metrics:

```text
latency_ms
semantic_entries_scanned
objects_checked
regions_checked
views_checked
context_size_chars
estimated_input_tokens or actual_input_tokens
speedup_vs_flat
token_reduction_vs_flat
```

Quality metrics are valid only with verified GT:

```text
hit@1
hit@3
expected_object_hit
expected_region_hit
expected_zone_hit
expected_view_hit
wrong_zone_rate
wrong_room_rate
```

Construction/map-building cost must be reported separately from per-query cost.

---

## 8. Modes And Provider Scope

Live and cached-live are not part of the next phase. Do not ask for provider
keys, quota, or billing work unless the user explicitly reopens live/cached-live
scope.

Legacy live environment variables:

```powershell
$env:SEMANTICSPLAT_PROVIDER = "openai"
$env:SEMANTICSPLAT_MODEL = "gpt-5.5"
$env:SEMANTICSPLAT_API_KEY = "sk-..."
```

Do not commit keys. Do not print full keys in logs. Do not ask the user to paste
keys into chat/code. Read keys only from environment variables.

Historical live issue:

```text
Latest live attempt reached OpenAI but failed with HTTP 429 insufficient_quota.
Correct status: live attempted historically but produced 0 successful results;
live/cached-live are now out of scope.
```

For `gpt-5*`, `o1*`, `o3*`, and `o4*`, do not send unsupported sampling params
unless explicitly verified.

Avoid sending unless supported:

```text
temperature
top_p
presence_penalty
frequency_penalty
logprobs
top_logprobs
logit_bias
```

---

## 9. Common Commands

Run tests:

```powershell
python -B -m pytest -q
```

If Windows temp permissions fail, use workspace-local or user AppData temp and
record the workaround.

Check formatting / whitespace:

```powershell
git diff --check
```

Run Week 3 stub gate:

```powershell
python -B scripts\run_experiment.py --config configs\week3_replica.yaml --mode stub --out outputs\week3\final_stub_semantic_gate_v1
python -B scripts\evaluate_results.py --benchmark docs\benchmarks\benchmark_queries_v1.json --results outputs\week3\final_stub_semantic_gate_v1\query_results --out outputs\week3\final_stub_semantic_gate_v1
```

Validate scene geometry:

```powershell
python -B scripts\validate_scene_geometry.py --scene backend\data\scenes\<scene_id> --output docs\validation\geometry\<scene_id>_geometry.json
```

Validate semantic index:

```powershell
python -B scripts\validate_semantic_index.py --scene backend\data\scenes\<scene_id> --out docs\validation\semantic_index\<scene_id>_semantic_index.json
```

---

## 10. Baseline Rules

ConceptGraphs is not a fair executed baseline unless evidence includes:

```text
repo checkout or Docker/image provenance
environment setup attempt
official/demo entrypoint identified
smoke command attempted
logs saved
native or canonical output produced, or exact blocker logged
```

Expected places:

```text
docs/baselines/conceptgraphs/
outputs/baselines/conceptgraphs_smoke_v1/
external/baselines/concept-graphs/ only if external repos are allowed
```

LangSplat is not a fair executed baseline unless evidence includes:

```text
repo checkout or Docker/image provenance
environment setup attempt
official/demo entrypoint identified
smoke command attempted
logs saved
native or canonical output produced, or exact blocker logged
```

Expected places:

```text
docs/baselines/langsplat/
outputs/baselines/langsplat_smoke_v1/
external/baselines/LangSplat/ only if external repos are allowed
```

If checkout/env/checkpoints/native layout are missing, status is:

```text
blocked before successful smoke execution
```

not:

```text
baseline ran
```

---

## 11. Documentation Wording

Prefer:

```text
manual captured scenes
manual semantic index
stub semantic gate
schema-valid outputs
credential-independent pipeline gate
same-input graph-vs-flat benchmark
verified GT labels
live/cached-live out of scope for next phase
baseline smoke attempted / blocked with exact error
not independent GT
accuracy unavailable
3D IoU unavailable
```

Avoid unless proven:

```text
fully automatic
zero human-in-the-loop
Replica evaluation
ScanNet evaluation
live reasoning result
cached_live verified
baseline comparison
semantic accuracy
3D localization
superiority
production-ready
publication-ready result
```

When asked "what is done?", classify requirements as:

```text
DONE
PARTIALLY_DONE
BLOCKED
NOT_DONE
DONE_BUT_INCORRECTLY_REPORTED
UNCLEAR_NEEDS_REVIEW
```

Use evidence paths.

Preferred audit format:

```text
Requirement | Status | Evidence | Gap | Needed from user
```

---

## 12. User-Dependent Decisions

Ask the user only for decisions that cannot be solved from code:

```text
Allow external repo clone? -> needed for ConceptGraphs/LangSplat expansion
GPU/CUDA available? -> may be needed for baselines
Official Replica/ScanNet data available? -> needed for original dataset plan
Manual GT annotation allowed? -> needed for accuracy
Should scope remain manual/stub? -> needed for honest final wording
Target venue/deadline? -> needed for final article format
```

Do not ask the user to do code-level work unless it requires credentials,
external downloads, licensing, or manual annotation.

---

## 13. Cleanup Rules

Keep docs organized. Do not leave unrelated files flat under `docs/`.

Preferred layout:

```text
docs/project/
docs/schemas/
docs/datasets/
docs/validation/geometry/
docs/validation/semantic_index/
docs/validation/capture_quality/
docs/benchmarks/
docs/baselines/conceptgraphs/
docs/baselines/langsplat/
docs/experiments/stub/
docs/experiments/live/
docs/experiments/cached_live/
docs/experiments/graph_vs_flat/
docs/reports/week1/
docs/reports/week2/
docs/reports/week3/
docs/reports/final/
docs/maintenance/
docs/archive/
```

Before deleting generated or ignored files, inspect first:

```powershell
git clean -ndX
git clean -nd
```

Do not delete evidence files, validation reports, benchmark queries, final docs,
scene data, or outputs needed for reproducibility. Move uncertain old docs to:

```text
docs/archive/review_required/
```

---

## 14. Before Finishing Any Agent Task

Run or document why you cannot run:

```powershell
python -B -m pytest -q
git diff --check
git status --short
```

In the final response/report include:

```text
what changed
what evidence was produced
what remains blocked
commands run
test result
whether push was performed
```

Default: do not push.
