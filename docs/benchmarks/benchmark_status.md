# Benchmark Status

Status date: 2026-06-26.

## Current Count

| Item | Count |
|---|---:|
| Total benchmark queries | 90 |
| Verified queries | 50 |
| Candidate unverified queries | 0 |
| Missing-GT queries | 40 |
| Plan target | 150-300 verified queries |
| Gap to 150 verified | 100 |
| Gap to 300 verified | 250 |

## Query Types

The benchmark already has query labels, but some are legacy buckets. The evaluator now reports Week 3 complexity strata by normalizing labels during aggregation only.

| Week 3 stratum | Source labels | Query count | Verified | Missing GT |
|---|---|---:|---:|---:|
| simple | `simple_object`, `negative` | 30 | 17 | 13 |
| compound | `attribute` | 15 | 8 | 7 |
| relational | `relational` | 15 | 10 | 5 |
| multi_hop | `multi_hop` | 15 | 7 | 8 |
| functional | `functional` | 15 | 8 | 7 |

## Verification Status

`verified_from_view_json` means the expected fields were checked against the existing semantic index, not independent dataset GT. These queries are usable for semantic-index retrieval metrics, but not for independent accuracy claims.

`candidate_unverified` is currently unused. No GT was inferred from the five manual semantic indexes in this pass because that would require manual visual review.

`missing_gt` means the query is executable for schema/runtime/stub coverage but excluded from accuracy and hit-rate denominators.

## Missing GT

The missing-GT set is `q051` through `q090`, covering:

| Scene | Missing-GT query count |
|---|---:|
| ConferenceHall | 8 |
| Museume | 8 |
| outdoor-drone | 8 |
| outdoor-street | 8 |
| Theater | 8 |

## Exact Next Manual GT Work

1. Decide whether the Week 3 target is 150 or 300 verified queries.
2. For each captured scene, review the captured RGB frames and manual ViewJSON side by side.
3. For every query, fill expected fields only when visible and defensible: `expected_view_ids`, `expected_node_ids`, `expected_zone_ids`, `expected_object_labels`, and optionally `expected_answer_contains`.
4. Mark visually checked entries as `verification_status: candidate_unverified` until a second pass verifies them.
5. Promote entries to a verified status only after independent review. Do not use manual semantic-index labels as independent GT.
