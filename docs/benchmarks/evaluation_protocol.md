# Canonical Evaluation Protocol

Version: `semanticsplat-eval-v1`
Scope: Week 2 and Week 3, Replica and available pilot scenes only.

## Required Inputs

1. A scene manifest following `docs/schemas/scene_manifest_schema.md`.
2. `docs/benchmarks/benchmark_queries_v1.json`.
3. Per-query results following `docs/schemas/query_result_schema.md`.
4. An immutable `run_config.json` identifying method, mode, code revision, and scene/query inputs.

## Required Output Layout

```text
outputs/week2/<run_id>/
  run_config.json
  logs.json
  query_results/<query_id>.json
  metrics_summary.json
  metrics_summary.md
  failure_modes.md
  model_cache/
```

Week 3 uses the same layout under `outputs/week3/<run_id>/`.

## Mode Labelling

- `stub`: no live model call occurred.
- `live`: the run made new provider calls and logged provider/model/version.
- `cached_live`: responses originated from a prior live run and were replayed.
- A result may not infer its mode from a directory name; `mode` is required in every result.
- Legacy simulated benchmark sessions are not live or cached-live evidence.

## Metric Rules

- Aggregate only saved results that pass minimum schema validation.
- Report result coverage separately from accuracy denominators.
- Compute expected ID hit metrics only when the corresponding GT list is non-empty.
- Record absent runtime or token data as `null`, not zero and not an estimate.
- Keep construction cost separate from per-query cost.
- Preserve per-query warnings and schema errors in the report.
- Report compact per-query-type coverage using Week 3 complexity strata: `simple`, `compound`, `relational`, `multi_hop`, and `functional`.
- Legacy benchmark labels are normalized for aggregation only: `simple_object -> simple`, `attribute -> compound`, and `negative -> simple`. Negative correctness still requires `expected_output_type: not_found` and verified GT.

## 3D IoU Rule

3D IoU is valid only when all conditions hold:

1. The scene manifest marks depth as reliable.
2. The query has a reliable GT 3D box.
3. The result contains a valid predicted 3D box in the documented coordinate frame.

If any condition fails, the aggregate value is `null`, status is `N/A`, and the reason is saved. No 2D, semantic, or retrieval metric may be renamed to 3D IoU.

## Failure Categories

- `wrong room / zone`
- `wrong object`
- `wrong view`
- `missing object in semantic description`
- `bad query parsing`
- `bad traversal`
- `bad bbox`
- `invalid depth`
- `ambiguous ground truth`
- `baseline adapter issue`
- `missing GT`

Multiple categories may apply to one query. Missing result files are reported as run-coverage gaps, not silently counted as semantic failures.

## Reproduction

```powershell
python scripts/evaluate_results.py `
  --benchmark docs/benchmarks/benchmark_queries_v1.json `
  --run-dir outputs/week2/<run_id>
```

The evaluator writes `metrics_summary.json`, `metrics_summary.md`, and `failure_modes.md` into the run directory.
