# Four-Person Article Sprint Plan

Status date: 2026-06-30.

This plan covers Phase B: a 1-2 week sprint for four people to turn the current
SemanticSplat prototype into defensible graph-vs-flat evidence and an article
draft. It assumes Phase A is covered by the refreshed root `AGENTS.md`.

The missing file name in the user request was empty, so this plan is based on
the repository structure plus the strongest local planning inputs:

- `semantic_splat_research_direction(1).md`
- `plan2.md`
- `paper-publication-plan.md`
- `docs/project/chatgpt_project_description.md`
- `docs/maintenance/plan_vs_project_audit.md`
- `docs/project/final_week3_acceptance.md`

## Project Structure Summary

SemanticSplat is a semantic-map intelligence layer for 3D or
RGB-D scene observations. The current prototype stores captured views, manual
ViewJSON annotations, semantic indexes, generated trees, query outputs, and
evaluation reports.

High-value areas:

| Area | Purpose |
|---|---|
| `backend/query/` | Stub/live/cached model clients, query pipeline, graph-vs-flat benchmark helpers. |
| `backend/geometry/` | Spatial graph and view clustering helpers. |
| `backend/data/scenes/` | Five captured scenes plus legacy `default` scene data. |
| `scripts/` | Experiment runner, evaluators, validators, baseline adapters. |
| `docs/benchmarks/` | Query set, protocol, legacy graph-vs-flat report. |
| `docs/validation/` | Scene/capture/semantic-index evidence. |
| `docs/baselines/` | Related work notes and baseline status. |
| `papers/` and `papers/beyond-proximity/` | Local papers and current LaTeX article draft. |

Current safest status:

```text
5 captured scenes, 96 manual ViewJSON annotations, 1,046 semantic items,
5 generated trees, 40 five-scene stub outputs, 50 default-scene stub outputs,
and 91 passing backend tests as of the 2026-06-30 local check.
```

Current core gap:

```text
The legacy graph-vs-flat benchmark works mainly on `default`, while the five
captured scenes are executable through manual semantic indexes and stub search.
The next milestone is to run graph-vs-flat on the same five-scene semantic items
and verified query labels.
```

## Success Criteria

Minimum 1-week success:

| Deliverable | Evidence path |
|---|---|
| Same-input graph-vs-flat runner for five captured scenes | `scripts/` plus tests under `tests/backend/` |
| 80-100 verified query labels with expected view/object/region/zone where possible | `docs/benchmarks/benchmark_queries_v2.json` |
| Graph-vs-flat result table with runtime/context/scan-count metrics | `outputs/graph_vs_flat/<run_id>/` |
| Literature comparison matrix | `docs/baselines/related_work_article_matrix.md` |
| Article draft updated with honest results and limitations | `papers/beyond-proximity/main.tex` or `docs/reports/final/article_draft.md` |

Strong 2-week success:

| Deliverable | Evidence path |
|---|---|
| 120-150 verified queries across simple, relational, intent, ambiguity, and freshness cases | `docs/benchmarks/benchmark_queries_v2.json` |
| Enriched semantic graph with zone-region-object-affordance links | `backend/data/scenes/*/tree/` or versioned graph outputs |
| Flat lexical, graph lexical, graph-affordance-pruned, and optional flat embedding comparison | `outputs/graph_vs_flat/<run_id>/metrics_summary.json` |
| At least one extended baseline smoke or a documented blocker | `outputs/baselines/`, `docs/baselines/*/status.md` |
| Full article draft with figures/tables, related work, method, experiments, limitations, and reproducibility appendix | `papers/beyond-proximity/main.tex`, `papers/beyond-proximity/main.pdf` |

Do not claim semantic accuracy, 3D IoU, official Replica/ScanNet evaluation, or
baseline superiority unless the required evidence exists.

## Team Roles

### Person 1: Graph And Benchmark Engineer

Primary mission: make graph-vs-flat work fairly on the five captured semantic
indexes.

Tasks:

| Day | Work | Output |
|---|---|---|
| 1 | Audit `backend/query/benchmark.py`, `backend/query/model_client.py`, captured tree format, and legacy `default` assumptions. | Implementation notes in `docs/experiments/graph_vs_flat/`. |
| 2-3 | Implement same-input flat and graph runners over captured semantic items. Count scanned entries, regions, objects, views, context chars, and tokens. | Runner script and focused tests. |
| 3-4 | Add deterministic intent/affordance expansion: sit, exit, charge, reception/help, restroom/facility if present. | Affordance map module and tests. |
| 5 | Produce first graph-vs-flat run on 40 existing five-scene queries. | `outputs/graph_vs_flat/<run_id>/`. |
| 6-8 | Support `benchmark_queries_v2.json` and verified labels from Person 2. Add hit@1/hit@3 only where GT exists. | Metrics summary with GT-aware denominators. |
| 9-10 | Freeze final run and export paper tables/figures. | `metrics_summary.md`, CSV/JSON tables, figure assets. |

Key rule: graph and flat must use the same scenes, same semantic items, same
query strings, and same GT labels.

### Person 2: Data, GT, And Evaluation Lead

Primary mission: make benchmark labels reliable enough for quality claims.

Tasks:

| Day | Work | Output |
|---|---|---|
| 1 | Inventory five captured scenes and current 40 five-scene queries. | Coverage table by scene/query type. |
| 2-3 | Create `benchmark_queries_v2.json` with expected objects, views, regions, zones, and verification status. | 80-100 verified queries minimum. |
| 4 | Add query type distribution: simple, relational, multi-hop, intent, ambiguity, freshness. | `docs/benchmarks/benchmark_status_v2.md`. |
| 5 | Review negative and ambiguity cases to avoid fake accuracy. | Missing/ambiguous GT report. |
| 6-8 | Expand to 120-150 queries if the team takes the 2-week path. | Versioned benchmark JSON and changelog. |
| 9 | Validate evaluator outputs and all denominators. | `docs/benchmarks/gt_coverage_report.md`. |
| 10 | Lock final benchmark used in article. | Immutable benchmark ID and evidence paths. |

Quality metrics are allowed only where expected IDs are filled and verified.
Manual semantic labels are reference labels, not independent semantic GT unless a
separate reviewer verifies them.

### Person 3: Baselines And Literature Lead

Primary mission: read similar papers, define fair comparisons, and prevent
unsupported claims.

Local sources to start with:

| Source | Use |
|---|---|
| `docs/baselines/related_work_notes.md` | Existing baseline positioning. |
| `docs/baselines/related_work_outline.md` | Article section skeleton. |
| `papers/` and `markdown-papers/` | Local PDFs/converted notes. |
| `papers/beyond-proximity/refs.bib` | Current bibliography seed. |

Mandatory method families:

| Family | Examples | What to compare |
|---|---|---|
| Flat semantic search | exhaustive lexical or embedding search over same semantic items | context/tokens, entries scanned, latency, quality. |
| 3D language fields | LERF, LangSplat, Semantic Gaussians, LEGS | object localization and absence of explicit hierarchy. |
| Open-vocabulary 3D graphs | ConceptGraphs, BBQ, OpenScene/ConceptFusion as context | object graph and relational reasoning. |
| Semantic hierarchy / spatial graphs | Hydra, SceneGraph++ and related robotics scene graphs | hierarchy, room/zone structure, traversal. |

Tasks:

| Day | Work | Output |
|---|---|---|
| 1-2 | Verify local bibliography and collect official citation metadata. | Updated `refs.bib` draft or notes file. |
| 2-4 | Build a related-work matrix: inputs, outputs, hierarchy support, query support, runtime evidence, setup blockers. | `docs/baselines/related_work_article_matrix.md`. |
| 4-5 | Decide which methods can be measured in repo and which are literature-only context. | Baseline feasibility table. |
| 6-8 | If allowed and feasible, extend one baseline smoke; otherwise document exact blockers. | Updated `docs/baselines/*/status.md`. |
| 8-10 | Write article related work and comparison framing. | Related-work section patch. |

Important: literature can motivate that our graph should scale better in context,
but speed superiority must be proven by our same-input benchmark, not by mixing
paper-reported numbers from different datasets.

### Person 4: Paper, Integration, And Reproducibility Lead

Primary mission: turn the technical run into an honest article draft.

Tasks:

| Day | Work | Output |
|---|---|---|
| 1 | Reconcile current article claims with actual evidence. | Claim audit for `papers/beyond-proximity/main.tex`. |
| 2 | Define paper story and contribution wording. | Outline and figure/table checklist. |
| 3-5 | Update method section to current real pipeline: manual semantic index, graph adaptation, same-input benchmark. | Draft method section. |
| 5-7 | Insert first result tables and limitations from Persons 1-3. | Draft experiments section. |
| 8 | Create reproducibility appendix: commands, configs, outputs, limitations. | Appendix or `docs/reports/final/`. |
| 9 | Build PDF and fix LaTeX/doc issues. | `main.pdf` or documented build blocker. |
| 10 | Final proofread: no live/cached/ScanNet/accuracy overclaim. | Final article readiness checklist. |

The article title and claims should reflect the measured result. If graph-vs-flat
metrics are not ready, write a technical report rather than a research paper
claiming speedup.

## Day-By-Day Plan

### Days 1-2: Freeze Scope And Data Contract

Team outputs:

- project claim sheet;
- benchmark v2 schema;
- graph-vs-flat runner design;
- related-work matrix skeleton;
- article outline.

Go/no-go checkpoint:

```text
Can we compare graph and flat over exactly the same five-scene semantic items?
Can we verify at least 80 queries within the available time?
```

### Days 3-5: Build And Run First Evidence

Team outputs:

- first same-input graph-vs-flat implementation;
- 80-100 verified queries;
- first metrics summary;
- related-work notes expanded;
- article method draft.

Go/no-go checkpoint:

```text
If graph does not beat flat on tokens/context or latency, preserve the result
honestly and analyze why. Do not tune the benchmark by deleting hard cases.
```

### Days 6-8: Strengthen Results

Team outputs:

- expanded query set if taking the 2-week path;
- affordance/intent evaluation;
- failure cases and wrong-zone analysis;
- baseline status or blocker reports;
- figures/tables for the article.

### Days 9-10: Write Article Draft

Team outputs:

- frozen run IDs;
- final tables;
- related work;
- experiment section;
- limitations;
- reproducibility commands;
- built PDF or documented build blocker.

### Days 11-14: Optional Polish And Baseline Expansion

Use only if the team has the second week:

- expand to 120-150 verified queries;
- add optional flat embedding baseline;
- extend ConceptGraphs/LangSplat only if environment/data blockers are solved;
- improve figures and abstract;
- do an independent claim audit before sharing the article.

## Article Structure

Recommended article sections:

1. Introduction: indoor maps give coordinates, not enough meaning.
2. Problem: environment-understanding queries need semantic hierarchy and context
   control.
3. Method: captured views, semantic index, enriched graph, graph traversal, flat
   baselines.
4. Benchmark: five manually captured scenes, query types, GT rules, metrics.
5. Experiments: same-input graph-vs-flat efficiency and quality.
6. Related Work: language fields, 3DGS semantic methods, object graphs,
   hierarchical scene graphs.
7. Limitations: manual annotations, no independent accuracy unless verified, no
   3D IoU, no official Replica/ScanNet, baseline smokes only.
8. Reproducibility: commands, configs, output paths.

## Evidence Checklist Before Final Article Claims

Before writing "faster", require:

```text
same scenes
same semantic items
same query set
same GT labels
runtime or cost metric saved
context/tokens metric saved
quality denominator saved
run config saved
code revision saved
```

Before writing "accurate", require:

```text
verified expected objects/regions/zones/views
GT source documented
missing/ambiguous labels excluded from denominator
negative cases verified
```

Before writing "baseline comparison", require:

```text
native baseline command or official smoke evidence
native output saved
canonical adapter output saved
same scene/query subset or explicit mismatch caveat
```

## Commands To Run At Sprint End

At minimum:

```powershell
python -B -m pytest -q
git diff --check
git status --short
```

Recommended evidence commands after implementation:

```powershell
python -B scripts\validate_semantic_index.py --scenes backend\data\scenes\ConferenceHall-capture-pilot backend\data\scenes\Museume-capture backend\data\scenes\Theater-capture backend\data\scenes\outdoor-drone-capture backend\data\scenes\outdoor-street-capture --out docs\validation\semantic_index
python -B scripts\run_experiment.py --config configs\week3_replica.yaml --mode stub --out outputs\week3\final_stub_semantic_gate_v1
```

Add the new graph-vs-flat command here once Person 1 implements it.

```powershell
python -B scripts\run_graph_vs_flat.py --config configs\graph_vs_flat_v2.yaml
```

Outputs: `outputs/graph_vs_flat/<run_id>/` (`metrics_summary.json`, `.md`, `.csv`, `per_query_results.json`).

v1 config (`graph_vs_flat_five_scenes.yaml`) remains for the 40-query efficiency baseline.

## Decisions Needed From User

The team can proceed with local docs/code immediately. These decisions may be
needed for the strong 2-week target:

| Decision | Why it matters |
|---|---|
| Target venue or format | Determines article length, template, and deadline. |
| Is manual GT annotation allowed? | Needed for accuracy-style metrics. |
| Are external repo clones allowed? | Needed for ConceptGraphs/LangSplat expansion. |
| Is official ScanNet/Replica data available? | Needed before any dataset claim. |
| Should live/cached-live remain out of scope? | Current plan assumes yes. |
