# Raw RGB-D Hierarchy Benchmark

- RGB images: 97
- Depth maps: 97
- Camera poses: 97
- Construction model calls: 15
- Construction label tokens: 151
- Construction latency: 14.15 s
- Manual ViewJSON files read during construction: 0

| Method | hit@1 | hit@3 | MRR | Views checked | Query tokens |
|---|---:|---:|---:|---:|---:|
| flat_clip | 0.432 | 0.624 | 0.571 | 19.40 | 8.03 |
| manual_hierarchy_clip | 0.424 | 0.592 | 0.537 | 10.76 | 8.03 |
| raw_rgbd_hierarchy_clip | 0.368 | 0.528 | 0.480 | 10.00 | 8.03 |

- Macro pairwise F1 against manual zones: 0.487
- Macro Rand index against manual zones: 0.637

Construction uses raw RGB images, depth maps, and camera poses only. Manual zones and expected view IDs are loaded after construction for evaluation.
