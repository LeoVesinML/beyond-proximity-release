# BBQ-aligned grounding summary

- Benchmark: `scannet_bbq_aligned_referit3d_official_gt_pilot_v1`
- Queries: 48
- Canonical / legacy-adapted results: 48 / 0

Missing, unavailable, and invalid predictions remain in GT-eligible 3D-box denominators as IoU 0.

| Metric | Value | Numerator / denominator |
|---|---:|---:|
| 3D bbox IoU | 0.6061 | 48 eligible |
| Acc@0.1 | 0.6042 | 29 / 48 |
| Acc@0.25 | 0.6042 | 29 / 48 |
| Acc@0.5 | 0.6042 | 29 / 48 |
| Recall@1 exact object ID | 0.6042 | 29 / 48 |
| Recall@3 exact object ID | 0.8125 | 39 / 48 |
| Recall@5 exact object ID | 0.8125 | 39 / 48 |
| MRR exact object ID | 0.7049 | 48 eligible |
| Verified-negative accuracy | N/A | 0 / 0 |

## Segmentation

| Metric | Status | Value |
|---|---|---:|
| mAcc | N/A | N/A |
| mIoU | N/A | N/A |
| fmIoU | N/A | N/A |

## Failure taxonomy

| Failure | Count |
|---|---:|
| wrong object | 15 |
| wrong relation | 19 |
| false positive | 0 |
| invalid/unavailable prediction | 4 |
| bad bbox | 4 |
| missing GT | 0 |

## Efficiency

| Measure | Mean | Total | Records |
|---|---:|---:|---:|
| runtime_seconds | 0.0053 | 0.2565 | 48 |
| checked_nodes | 1.8333 | 88.0000 | 48 |
| checked_views | 3.2917 | 158.0000 | 48 |
| checked_objects | 21.2917 | 1022.0000 | 48 |
| context_chars | 3658.0000 | 175584.0000 | 48 |
| estimated_tokens | 927.5833 | 44524.0000 | 48 |

Construction cost is reported separately because it is a one-time indexing/map cost.

- Construction cost status: `reported`
