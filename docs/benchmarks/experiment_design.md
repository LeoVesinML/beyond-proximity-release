# Week 2-3 Experiment Design

Status: canonical plan, 2026-06-21.

## Objective

Convert the current demo and simulated benchmark into a reproducible evaluation prototype over Replica or available pilot scenes. ScanNet is excluded from Week 2 and Week 3.

## Experimental Units

| Unit | Definition |
|---|---|
| Scene | One versioned scene manifest with posed RGB(-D), intrinsics, semantic index status, and depth validation. |
| Query | One record from `docs/benchmarks/benchmark_queries_v1.json`. |
| Result | One canonical `query_result.json` identified by scene, query, method, mode, and run ID. |
| Run | One immutable config plus logs, per-query results, metrics, warnings, and optional model cache. |

## Execution Modes

| Mode | Model behavior | Scientific use |
|---|---|---|
| `stub` | Deterministic local fallback; no model credentials. | Pipeline and schema validation only. |
| `live` | New provider calls with model/version and measured usage logged. | Primary model evidence when credentials are available. |
| `cached_live` | Replays raw responses produced by a documented live run. | Reproducibility and offline reruns; not a new live measurement. |

## Week 2 Scope

- Establish canonical schemas and an automated evaluation harness.
- Restore Replica/pilot ingestion with explicit depth validation.
- Run one full stub batch through a Cursor-independent command.
- Run at least one live query only if credentials are available.
- Evaluate at least 50 verified semantic-index queries.
- Report coverage, retrieval status, expected node/view/zone hits where GT exists, runtime, tokens where available, and failure categories.

## Week 3 Scope

- Improve resume, retries, caching, deterministic reruns, and batch diagnostics.
- Expand toward 100 queries only when scene evidence supports verified expectations.
- Run repeated Replica/pilot batches and compare run consistency.
- Add one baseline adapter smoke comparison using the same canonical result schema.
- Strengthen reporting and failure analysis.

## Metrics

| Metric | Rule |
|---|---|
| Retrieval success | Positive query returns `found=true`; negative query returns `found=false`. |
| Expected view hit | Computed only when `expected_view_ids` is non-empty. |
| Expected node hit | Computed only when `expected_node_ids` is non-empty. |
| Expected zone hit | Computed only when `expected_zone_ids` is non-empty. |
| Not-found correctness | Computed only for `negative` queries. |
| Runtime | Direct wall-clock runtime from the result, never the legacy estimate. |
| Token usage | Provider usage when available; otherwise `null`. |
| Context work | Checked-view and visited-node counts from the trace. |
| 3D IoU | Computed only with reliable depth and reliable GT 3D boxes; otherwise `N/A`. |

## Current Evidence Limitations

The 50-query benchmark v1 is verified against the existing `default` ViewJSON and tree records, not against independent dataset annotations. It is suitable for evaluation-layer validation and semantic-index regression tests, not final paper accuracy.

The current `default` depth maps are constant-valued and cannot support 3D localization claims. Live and cached-live evidence remains unavailable until Phase 2 provider work and credentials/cache are present.
