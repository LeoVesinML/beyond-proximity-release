# Benchmark Scene Coverage

Status date: 2026-06-22. Benchmark: `six_scene_mixed_gt_v1`.

## Scene Coverage

| scene_id | queries | with GT | without GT | valid metrics | N/A or unavailable metrics |
|---|---:|---:|---:|---|---|
| `default` | 50 | 50 | 0 | retrieval; expected view/node/zone/object where declared; negative correctness; schema and output coverage | 3D bbox IoU is `N/A` because reliable depth plus GT and predicted 3D boxes are unavailable |
| `ConferenceHall` | 8 | 0 | 8 | schema conformance; output coverage; mode provenance | semantic accuracy metrics are unavailable; 3D bbox IoU is `N/A` |
| `Museume` | 8 | 0 | 8 | schema conformance; output coverage; mode provenance | semantic accuracy metrics are unavailable; 3D bbox IoU is `N/A` |
| `outdoor-drone` | 8 | 0 | 8 | schema conformance; output coverage; mode provenance | semantic accuracy metrics are unavailable; 3D bbox IoU is `N/A` |
| `outdoor-street` | 8 | 0 | 8 | schema conformance; output coverage; mode provenance | semantic accuracy metrics are unavailable; 3D bbox IoU is `N/A` |
| `Theater` | 8 | 0 | 8 | schema conformance; output coverage; mode provenance | semantic accuracy metrics are unavailable; 3D bbox IoU is `N/A` |
| **Total** | **90** | **50** | **40** | mixed by GT availability | mixed by GT availability |

The 50 legacy `default` expectations are verified against the existing ViewJSON and semantic tree, not independent visual or dataset GT. The five new scenes are PLY-only assets without frame records, semantic labels, ViewJSON, tree nodes, zones, or independent GT.

## Category Balance

| query_type | count |
|---|---:|
| `simple_object` | 15 |
| `attribute` | 15 |
| `relational` | 15 |
| `multi_hop` | 15 |
| `functional` | 15 |
| `negative` | 15 |

Each of the five new scenes has at least one query in every category. The extra records were distributed to make all six benchmark categories exactly balanced.

## Missing-GT Policy

Every query for the five PLY-only scenes has:

```json
{
  "expected_zone_ids": [],
  "expected_node_ids": [],
  "expected_view_ids": [],
  "expected_object_labels": [],
  "notes": "GT unavailable; excluded from accuracy metrics",
  "verification_status": "missing_gt",
  "verification_source": []
}
```

The query text is a gate input, not a claim that its target exists or is absent. In particular, a `negative` record with `verification_status: missing_gt` is not scored as a correct absence.

## Evaluator Behavior

- Queries are scoped to `run_config.scene_ids`; the Week 3 five-scene gate excludes the 50 legacy `default` queries.
- `missing_gt` records may contribute to schema and output coverage but never to semantic accuracy denominators.
- Explicit `availability_status: unavailable` results are schema checked but excluded from measured runtime and accuracy metrics.
- 3D IoU is `N/A` unless reliable depth, reliable GT 3D boxes, and predicted `bbox_3d` are all present.

## Week 3 Stub Gate

```powershell
python -B scripts/run_experiment.py --config configs/week3_replica.yaml --mode stub

python -B scripts/evaluate_results.py `
  --benchmark docs/benchmarks/benchmark_queries_v1.json `
  --results outputs/week3/week3_stub_five_scene_batch_v1/query_results `
  --out outputs/week3/week3_stub_five_scene_batch_v1
```

Output: `outputs/week3/week3_stub_five_scene_batch_v1/`.

| scene_id | canonical results | mode | execution availability | accuracy eligibility |
|---|---:|---|---|---|
| `ConferenceHall` | 8 | `stub` | unavailable - no semantic views | excluded - no GT |
| `Museume` | 8 | `stub` | unavailable - no semantic views | excluded - no GT |
| `outdoor-drone` | 8 | `stub` | unavailable - no semantic views | excluded - no GT |
| `outdoor-street` | 8 | `stub` | unavailable - no semantic views | excluded - no GT |
| `Theater` | 8 | `stub` | unavailable - no semantic views | excluded - no GT |

Gate result:

- Scoped query outputs: 40/40.
- Canonical schema-valid outputs: 40/40.
- Stub outputs marked as live: 0.
- Model calls: 0.
- Accuracy-eligible outputs: 0.
- 3D IoU: `N/A` with denominator 0.

## Live and Cached-Live Status

The live gate was not attempted because `SEMANTICSPLAT_PROVIDER`, `SEMANTICSPLAT_MODEL`, and `SEMANTICSPLAT_API_KEY` were absent. The missing provider configuration did not fail the stub gate. No `live` or `cached_live` output, raw response cache, token usage, or secret-bearing artifact was created.

## Final Interpretation

- Real measured metrics: benchmark/scene counts, 40/40 output coverage, 40/40 canonical schema validity, and `stub` mode provenance.
- Unavailable metrics: five-scene semantic accuracy, negative correctness, live token/latency/cost measurements, and cached-live equivalence.
- `N/A` metrics: camera-view coverage statistics and all 3D localization/IoU metrics.
- Remaining blockers: no frames/cameras/semantic index/GT for the five scenes and no provider configuration/cache.
- ScanNet remains postponed and excluded.
