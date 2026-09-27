# Final Benchmark Lock V2

Status date: 2026-07-01.

Immutable benchmark ID: `five_capture_manual_semantic_index_v2`.

This is the locked Person 2 benchmark package for article-facing graph-vs-flat work. It is manual semantic-index reference GT, not independent dataset GT.

## Required Person 2 Outputs

| Plan output | Status | Evidence |
|---|---|---|
| Coverage table by scene/query type | DONE | `docs/benchmarks/person2_coverage_table_v2.md` |
| 80-100 verified queries minimum | DONE | 150 queries in `docs/benchmarks/benchmark_queries_v2.json` |
| Expected objects, views, regions, zones | DONE | `docs/benchmarks/benchmark_queries_v2.json`, `docs/benchmarks/manual_zone_gt_v2.json` |
| Query distribution | DONE | `docs/benchmarks/benchmark_status_v2.md` |
| Missing/ambiguous GT report | DONE | `docs/benchmarks/ambiguous_and_negative_review_v2.md` |
| 120-150 two-week expansion | DONE | 150-query v2 benchmark |
| Denominator validation / GT coverage | DONE | `docs/benchmarks/gt_coverage_report.md` |
| Final benchmark lock | DONE | this file |

## Evidence Paths

```text
docs/benchmarks/benchmark_queries_v2.json
docs/benchmarks/manual_zone_gt_v2.json
docs/benchmarks/manual_bbox_gt_v2.json
docs/benchmarks/person2_coverage_table_v2.md
docs/benchmarks/benchmark_status_v2.md
docs/benchmarks/gt_coverage_report.md
docs/benchmarks/ambiguous_and_negative_review_v2.md
docs/benchmarks/benchmark_v2_changelog.md
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

## Locked Limitations

- Freshness/change queries remain blocked because no temporal recapture/change labels exist.
- Manual labels and boxes are not official Replica, ScanNet, or independent dataset GT.
- Zone expectations are manual reference labels emitted as stable tree-native nodes after rebuild.
