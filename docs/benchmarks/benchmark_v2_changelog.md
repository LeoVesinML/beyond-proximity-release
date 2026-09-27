# Benchmark V2 Changelog

Status date: 2026-07-01.

| Version item | Change | Evidence |
|---|---|---|
| v1 inventory | Audited 90-query v1 file, including 40 five captured-scene queries with missing expected IDs. | `docs/benchmarks/person2_coverage_table_v2.md` |
| v2 benchmark | Created 150 five-scene verified manual semantic-index queries, 30 per scene. | `docs/benchmarks/benchmark_queries_v2.json` |
| query types | Balanced evaluator query types: simple_object, attribute, relational, multi_hop, functional, negative. | `docs/benchmarks/benchmark_status_v2.md` |
| plan buckets | Added Person 2 buckets: simple, relational, multi_hop, intent, ambiguity review, freshness blocker. | `docs/benchmarks/benchmark_status_v2.md` |
| zone references | Added manual expected zone IDs and labels for every positive query. | `docs/benchmarks/manual_zone_gt_v2.json` |
| negative review | Kept absence checks separate from positive accuracy and documented limitations. | `docs/benchmarks/ambiguous_and_negative_review_v2.md` |
| bbox package | BBox builder adds manual coarse 2D/3D boxes for positive queries. | `docs/benchmarks/manual_bbox_gt_v2.json` |
