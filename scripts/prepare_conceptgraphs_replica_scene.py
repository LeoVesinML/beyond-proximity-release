#!/usr/bin/env python3
"""Sample an official NICE-SLAM Replica RGB-D trajectory for ConceptGraphs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=10)
    args = parser.parse_args()
    if args.frame_count < 2:
        raise SystemExit("--frame-count must be at least 2")
    result_dir = args.source / "results"
    trajectory_path = args.source / "traj.txt"
    images = sorted(result_dir.glob("frame*.jpg"))
    depths = sorted(result_dir.glob("depth*.png"))
    trajectory = [line for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    usable = min(len(images), len(depths), len(trajectory))
    if usable < args.frame_count:
        raise SystemExit(
            f"Replica scene {args.source} has {usable} complete RGB-D poses; "
            f"need {args.frame_count}"
        )
    indices = sorted({round(index * (usable - 1) / (args.frame_count - 1)) for index in range(args.frame_count)})
    if len(indices) != args.frame_count:
        raise SystemExit("Replica sampling produced duplicate frame indices")

    scene_out = args.out / args.scene_id
    sampled_results = scene_out / "results"
    sampled_results.mkdir(parents=True, exist_ok=True)
    for pattern in ("frame*.jpg", "depth*.png"):
        for stale_path in sampled_results.glob(pattern):
            stale_path.unlink()
    manifest_frames = []
    sampled_poses = []
    for output_index, source_index in enumerate(indices):
        image_out = sampled_results / f"frame{output_index:06d}.jpg"
        depth_out = sampled_results / f"depth{output_index:06d}.png"
        shutil.copy2(images[source_index], image_out)
        shutil.copy2(depths[source_index], depth_out)
        sampled_poses.append(trajectory[source_index])
        manifest_frames.append({
            "view_id": f"frame{source_index:06d}",
            "sample_index": output_index,
            "source_index": source_index,
            "source_image": str(images[source_index]),
            "source_depth": str(depths[source_index]),
        })
    (scene_out / "traj.txt").write_text("\n".join(sampled_poses) + "\n", encoding="utf-8")
    manifest = {
        "adapter": "nice_slam_replica_to_conceptgraphs_sample_v1",
        "scene_id": args.scene_id,
        "source_scene": str(args.source),
        "source_frame_count": usable,
        "sampled_frame_count": len(indices),
        "sample_strategy": "evenly_spaced_inclusive_endpoints",
        "source_indices": indices,
        "pose_convention": "official NICE-SLAM Replica camera-to-world trajectory",
        "depth_scale": 6553.5,
        "frames": manifest_frames,
    }
    (scene_out / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {args.scene_id}: {len(indices)} / {usable} Replica frames")


if __name__ == "__main__":
    main()
