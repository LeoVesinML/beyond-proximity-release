# BBQ-aligned grounding summary

- Benchmark: `scannet_bbq_aligned_referit3d_official_gt_pilot_v1`
- Queries: 48
- Canonical / legacy-adapted results: 48 / 0

Missing, unavailable, and invalid predictions remain in GT-eligible 3D-box denominators as IoU 0.

| Metric | Value | Numerator / denominator |
|---|---:|---:|
| 3D bbox IoU | 0.5020 | 48 eligible |
| Acc@0.1 | 0.5000 | 24 / 48 |
| Acc@0.25 | 0.5000 | 24 / 48 |
| Acc@0.5 | 0.5000 | 24 / 48 |
| Recall@1 exact object ID | 0.5208 | 25 / 48 |
| Recall@3 exact object ID | 0.8333 | 40 / 48 |
| Recall@5 exact object ID | 0.8750 | 42 / 48 |
| MRR exact object ID | 0.6876 | 48 eligible |
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
| wrong object | 20 |
| wrong relation | 1 |
| false positive | 0 |
| invalid/unavailable prediction | 4 |
| bad bbox | 4 |
| missing GT | 0 |

## Efficiency

| Measure | Mean | Total | Records |
|---|---:|---:|---:|
| runtime_seconds | 0.0016 | 0.0761 | 48 |
| checked_nodes | 1.8333 | 88.0000 | 48 |
| checked_views | 2.9375 | 141.0000 | 48 |
| checked_objects | 22.7083 | 1090.0000 | 48 |
| context_chars | 3897.4375 | 187077.0000 | 48 |
| estimated_tokens | 987.4167 | 47396.0000 | 48 |

Construction cost is reported separately because it is a one-time indexing/map cost.

- Construction cost status: `reported`
