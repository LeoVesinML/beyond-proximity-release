# BBQ-aligned grounding summary

- Benchmark: `scannet_bbq_grounding_extended_v2`
- Queries: 64
- Canonical / legacy-adapted results: 64 / 0

Missing, unavailable, and invalid predictions remain in GT-eligible 3D-box denominators as IoU 0.

| Metric | Value | Numerator / denominator |
|---|---:|---:|
| 3D bbox IoU | 0.5910 | 56 eligible |
| Acc@0.1 | 0.5893 | 33 / 56 |
| Acc@0.25 | 0.5893 | 33 / 56 |
| Acc@0.5 | 0.5893 | 33 / 56 |
| Recall@1 exact object ID | 0.6071 | 34 / 56 |
| Recall@3 exact object ID | 0.7857 | 44 / 56 |
| Recall@5 exact object ID | 0.8214 | 46 / 56 |
| MRR exact object ID | 0.7046 | 56 eligible |
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
| wrong object | 15 |
| wrong relation | 19 |
| false positive | 0 |
| invalid/unavailable prediction | 8 |
| bad bbox | 8 |
| missing GT | 0 |

## Efficiency

| Measure | Mean | Total | Records |
|---|---:|---:|---:|
| runtime_seconds | 0.0014 | 0.0886 | 64 |
| checked_nodes | 1.5000 | 96.0000 | 64 |
| checked_views | 1.7344 | 111.0000 | 64 |
| checked_objects | 4.3750 | 280.0000 | 64 |
| context_chars | 755.3125 | 48340.0000 | 64 |
| estimated_tokens | 200.0625 | 12804.0000 | 64 |

Construction cost is reported separately because it is a one-time indexing/map cost.

- Construction cost status: `reported`
