# Benchmark Status V2

Status date: 2026-07-01.

Benchmark file: `docs/benchmarks/benchmark_queries_v2.json`.

This is a verified manual-semantic-index benchmark over the five captured scenes. It is not independent dataset GT.

## Counts

- Total queries: 150
- Verified-from-ViewJSON queries: 150
- Scene count: 5
- Query types: attribute, functional, multi_hop, negative, relational, simple_object
- Positive queries: 125
- Negative queries: 25

## Query Type Distribution

| Query type | Count |
|---|---:|
| attribute | 25 |
| functional | 25 |
| multi_hop | 25 |
| negative | 25 |
| relational | 25 |
| simple_object | 25 |

## Sprint-Plan Bucket Coverage

The sprint plan requested simple, relational, multi-hop, intent, ambiguity, and freshness coverage. The benchmark keeps the existing schema query types for evaluator compatibility and maps them as follows.

| Sprint bucket | Status | Benchmark query types | Count | Notes |
|---|---|---|---:|---|
| simple | executable | `simple_object`, `attribute` | 50 |  |
| relational | executable | `relational` | 25 |  |
| multi_hop | executable | `multi_hop` | 25 |  |
| intent | executable | `functional` | 25 |  |
| ambiguity | reviewed_not_labeled_ambiguous | N/A | 0 | Ambiguous cases are documented separately. Multi-answer positives keep multiple expected IDs instead of forcing a single canonical target. Multi-answer positive queries: 85. |
| freshness | blocked_no_temporal_gt | N/A | 0 | The captured scenes have no temporal recapture or change labels, so freshness/change queries would be fabricated. |
| negative_absence_review | executable_for_absence_checks | `negative` | 25 | Negative cases are reviewed separately to avoid fake accuracy. |

## Scene Distribution

| Scene scope | Count |
|---|---:|
| `ConferenceHall` | 30 |
| `Museume` | 30 |
| `Theater` | 30 |
| `outdoor-drone` | 30 |
| `outdoor-street` | 30 |

## Expected Field Coverage

| Field | Non-empty queries | Notes |
|---|---:|---|
| `expected_view_ids` | 125 | Positive queries have expected views; negatives intentionally empty. |
| `expected_node_ids` | 125 | Positive queries map to manual semantic tree nodes. |
| `expected_region_ids` | 50 | Region/functional/multi-hop queries include region nodes where available. |
| `expected_zone_ids` | 125 | Manual zone references exist for positives and are emitted as stable zone nodes in rebuilt captured-scene trees. |

## Use

Use this benchmark for same-input graph-vs-flat semantic-index evaluation. Report it as manual semantic-index reference GT, not independent semantic accuracy.
