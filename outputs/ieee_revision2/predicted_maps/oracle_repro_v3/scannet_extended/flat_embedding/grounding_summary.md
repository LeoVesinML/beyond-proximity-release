# BBQ-aligned grounding summary

- Benchmark: `scannet_bbq_grounding_extended_v2`
- Queries: 64
- Canonical / legacy-adapted results: 64 / 0

Missing, unavailable, and invalid predictions remain in GT-eligible 3D-box denominators as IoU 0.

| Metric | Value | Numerator / denominator |
|---|---:|---:|
| 3D bbox IoU | 0.3427 | 56 eligible |
| Acc@0.1 | 0.3393 | 19 / 56 |
| Acc@0.25 | 0.3393 | 19 / 56 |
| Acc@0.5 | 0.3393 | 19 / 56 |
| Recall@1 exact object ID | 0.3393 | 19 / 56 |
| Recall@3 exact object ID | 0.7857 | 44 / 56 |
| Recall@5 exact object ID | 0.8750 | 49 / 56 |
| MRR exact object ID | 0.5707 | 56 eligible |
| Verified-negative accuracy | 0.0000 | 0 / 8 |

## Segmentation

| Metric | Status | Value |
|---|---|---:|
| mAcc | N/A | N/A |
| mIoU | N/A | N/A |
| fmIoU | N/A | N/A |

## Failure taxonomy

| Failure | Count |
|---|---:|
| wrong object | 37 |
| wrong relation | 23 |
| false positive | 8 |
| invalid/unavailable prediction | 0 |
| bad bbox | 0 |
| missing GT | 0 |

## Efficiency

| Measure | Mean | Total | Records |
|---|---:|---:|---:|
| runtime_seconds | 1.2545 | 80.2892 | 64 |
| checked_nodes | 2.0000 | 128.0000 | 64 |
| checked_views | 4.8750 | 312.0000 | 64 |
| checked_objects | 49.0000 | 3136.0000 | 64 |
| context_chars | 8414.6250 | 538536.0000 | 64 |
| estimated_tokens | 2114.9844 | 135359.0000 | 64 |

Construction cost is reported separately because it is a one-time indexing/map cost.

- Construction cost status: `reported`
