# Person 2 Coverage Table V2

Status date: 2026-07-01.

## Day 1 Inventory

| Source | Total queries | Five captured-scene queries | Five-scene queries with non-empty GT fields |
|---|---:|---:|---:|
| `benchmark_queries_v1.json` | 90 | 40 | 0 |
| `benchmark_queries_v2.json` | 150 | 150 | 150 |

The previous v1 file contained the 40 five-scene queries requested for inventory, but those captured-scene queries did not have reliable expected IDs. V2 replaces that gap with 150 verified manual semantic-index reference queries.

## V2 Scene By Query Type

| Scene | simple_object | attribute | relational | multi_hop | functional / intent | negative | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ConferenceHall` | 5 | 5 | 5 | 5 | 5 | 5 | 30 |
| `Museume` | 5 | 5 | 5 | 5 | 5 | 5 | 30 |
| `Theater` | 5 | 5 | 5 | 5 | 5 | 5 | 30 |
| `outdoor-street` | 5 | 5 | 5 | 5 | 5 | 5 | 30 |
| `outdoor-drone` | 5 | 5 | 5 | 5 | 5 | 5 | 30 |

## Plan Bucket Distribution

| Required bucket | Status | Count |
|---|---|---:|
| simple | executable | 50 |
| relational | executable | 25 |
| multi_hop | executable | 25 |
| intent | executable | 25 |
| ambiguity | reviewed_not_labeled_ambiguous | 0 |
| freshness | blocked_no_temporal_gt | 0 |

Additional reviewed negative absence cases: 25.
