# BBQ-aligned grounding summary

- Benchmark: `replica_bbq_aligned_official_gt_pilot_v2`
- Queries: 56
- Canonical / legacy-adapted results: 56 / 0

Missing, unavailable, and invalid predictions remain in GT-eligible 3D-box denominators as IoU 0.

| Metric | Value | Numerator / denominator |
|---|---:|---:|
| 3D bbox IoU | 1.0000 | 48 eligible |
| Acc@0.1 | 1.0000 | 48 / 48 |
| Acc@0.25 | 1.0000 | 48 / 48 |
| Acc@0.5 | 1.0000 | 48 / 48 |
| Recall@1 exact object ID | 1.0000 | 48 / 48 |
| Recall@3 exact object ID | 1.0000 | 48 / 48 |
| Recall@5 exact object ID | 1.0000 | 48 / 48 |
| MRR exact object ID | 1.0000 | 48 eligible |
| Verified-negative accuracy | 1.0000 | 8 / 8 |

## Segmentation

| Metric | Status | Value |
|---|---|---:|
| mAcc | N/A | N/A |
| mIoU | N/A | N/A |
| fmIoU | N/A | N/A |

## Failure taxonomy

| Failure | Count |
|---|---:|
| wrong object | 0 |
| wrong relation | 8 |
| false positive | 0 |
| invalid/unavailable prediction | 0 |
| bad bbox | 0 |
| missing GT | 0 |

## Efficiency

| Measure | Mean | Total | Records |
|---|---:|---:|---:|
| runtime_seconds | 0.0014 | 0.0792 | 56 |
| checked_nodes | 1.7143 | 96.0000 | 56 |
| checked_views | 1.0000 | 56.0000 | 56 |
| checked_objects | 12.1071 | 678.0000 | 56 |
| context_chars | 1180.9107 | 66131.0000 | 56 |
| estimated_tokens | 302.6786 | 16950.0000 | 56 |

Construction cost is reported separately because it is a one-time indexing/map cost.

- Construction cost status: `reported`
