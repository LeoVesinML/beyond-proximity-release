# Benchmark V2 Execution Report

Status date: 2026-07-01.

## Scope

This report locks the Person 2 benchmark package and the first same-input
graph-vs-flat efficiency run for article work.

Benchmark:

```text
docs/benchmarks/benchmark_queries_v2.json
benchmark_id = five_capture_manual_semantic_index_v2
```

Latest semantic gate:

```text
outputs/week3/benchmark_v2_stub_manual_bbox_gt_v4
mode = stub
method = semantic_splat
```

Latest graph-vs-flat run:

```text
outputs/week3/graph_vs_flat_benchmark_v2
mode = deterministic_local
```

All labels are manual semantic-index reference GT, not independent dataset GT.

## Person 2 Status

| Plan output | Status | Evidence |
|---|---|---|
| Coverage table by scene/query type | DONE | `docs/benchmarks/person2_coverage_table_v2.md` |
| 80-100 verified queries minimum | DONE | 150 queries in `docs/benchmarks/benchmark_queries_v2.json` |
| Expected objects, views, regions, zones | DONE | `docs/benchmarks/benchmark_queries_v2.json`, `docs/benchmarks/manual_zone_gt_v2.json` |
| Query type distribution | DONE | `docs/benchmarks/benchmark_status_v2.md` |
| Missing/ambiguous GT report | DONE | `docs/benchmarks/ambiguous_and_negative_review_v2.md` |
| 120-150 query expansion | DONE | 150-query v2 benchmark |
| Evaluator denominators | DONE | `docs/benchmarks/gt_coverage_report.md` |
| Final benchmark lock | DONE | `docs/benchmarks/final_benchmark_lock_v2.md` |

Freshness/change queries remain blocked because the captured scenes have no
temporal recapture or change labels. They were not fabricated.

## Tree And GT Status

| Item | Status |
|---|---|
| Semantic index validation | 5/5 scenes allowed |
| Geometry validation | 4/5 scenes 3D allowed; `outdoor-drone-capture` blocked by depth quality |
| Manual zone nodes | 26 stable `manual_zone_*` nodes across five scenes |
| Positive queries with expected zones | 125/125 |
| Positive queries with manual coarse boxes | 125/125 |
| Independent dataset GT | unavailable |

## Latest Stub Gate Result

Command:

```powershell
python -B scripts\run_experiment.py --config configs\week3_benchmark_v2_bbox_gt.yaml --mode stub --out outputs\week3\benchmark_v2_stub_manual_bbox_gt_v4
```

Coverage:

| Metric | Result |
|---|---:|
| Results | 150/150 |
| Schema-valid results | 150/150 |
| Evaluated results | 150 |
| Accuracy-eligible against manual labels | 150 |
| Missing results | 0 |
| Unavailable results | 0 |

Measured stub metrics:

| Metric | Result |
|---|---:|
| Retrieval success | 142/150 = 0.9467 |
| Expected view hit | 89/125 = 0.7120 |
| Expected node hit | 74/125 = 0.5920 |
| Expected zone hit | 113/125 = 0.9040 |
| Negative correctness | 22/25 = 0.8800 |
| Mean runtime | 0.0015 s |
| Mean checked views | 19.4000 |
| Mean visited nodes | 3.8800 |
| Mean coarse 3D IoU | 0.5592 over 88 eligible records |
| Token usage | unavailable in stub mode |

Failure categories:

| Category | Count |
|---|---:|
| wrong room / zone | 12 |
| wrong object | 40 |
| wrong view | 36 |
| bad traversal | 5 |
| bad bbox | 79 |
| missing GT | 0 |

## Graph Vs Flat Result

Command:

```powershell
python -B scripts\run_graph_vs_flat_benchmark_v2.py
```

Both methods use the same 150 queries, same semantic entries, and same manual
GT. Graph search first narrows by tree-native manual zones; flat search scans
every semantic entry in the scene.

Efficiency:

| Metric | Graph mean | Flat mean | Graph improvement |
|---|---:|---:|---:|
| latency_ms | 0.2492 | 0.2976 | 1.5865x speedup |
| semantic_entries_scanned | 154.2267 | 206.6000 | 25.32% fewer |
| context_size_chars | 33682.2400 | 45084.3133 | 25.29% smaller |
| estimated_input_tokens | 8420.5467 | 11271.1000 | 25.26% fewer |
| views_checked | 14.9600 | 19.4000 | 22.89% fewer |

Quality parity:

| Metric | Graph | Flat |
|---|---:|---:|
| retrieval_success | 143/150 = 0.9533 | 143/150 = 0.9533 |
| expected_view_hit | 78/125 = 0.6240 | 78/125 = 0.6240 |
| expected_node_hit | 62/125 = 0.4960 | 62/125 = 0.4960 |
| expected_zone_hit | 105/125 = 0.8400 | 105/125 = 0.8400 |
| expected_object_hit | 73/125 = 0.5840 | 73/125 = 0.5840 |
| not_found_correctness | 22/25 = 0.8800 | 22/25 = 0.8800 |

Allowed claim: on this deterministic same-input manual-reference benchmark,
graph search scans fewer entries, uses smaller context, uses fewer estimated
input tokens, and has lower measured local latency than flat search while
matching flat quality metrics.

Do not report this as live-provider latency, independent semantic accuracy, or
official dataset accuracy.

## Locked Evidence Paths

```text
docs/benchmarks/benchmark_queries_v2.json
docs/benchmarks/manual_zone_gt_v2.json
docs/benchmarks/manual_bbox_gt_v2.json
docs/benchmarks/person2_coverage_table_v2.md
docs/benchmarks/benchmark_status_v2.md
docs/benchmarks/gt_coverage_report.md
docs/benchmarks/ambiguous_and_negative_review_v2.md
docs/benchmarks/benchmark_v2_changelog.md
docs/benchmarks/final_benchmark_lock_v2.md
docs/project/limitations_resolution_status.md
configs/week3_benchmark_v2.yaml
configs/week3_benchmark_v2_bbox_gt.yaml
outputs/week3/benchmark_v2_stub_manual_bbox_gt_v4/metrics_summary.json
outputs/week3/benchmark_v2_stub_manual_bbox_gt_v4/metrics_summary.md
outputs/week3/benchmark_v2_stub_manual_bbox_gt_v4/failure_modes.md
outputs/week3/graph_vs_flat_benchmark_v2/graph_vs_flat_metrics.json
outputs/week3/graph_vs_flat_benchmark_v2/per_query_results.json
docs/experiments/graph_vs_flat/graph_vs_flat_benchmark_v2.md
```

## Remaining External Limits

| Limit | Status |
|---|---|
| Independent semantic accuracy | Needs separate reviewer or official dataset GT; not fabricated. |
| Official/pixel-accurate 3D IoU | Needs official or independently reviewed tight boxes/masks and predicted 3D boxes. |
| Freshness/change reasoning | Needs temporal recapture/change labels. |
| Official Replica/ScanNet claim | Needs official dataset ingestion and GT-backed evaluation. |
