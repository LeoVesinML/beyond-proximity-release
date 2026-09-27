#!/usr/bin/env python3
"""Select camera views with a deterministic pose-space coverage-greedy fallback."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.import_replica_scene import display_path, write_json  # noqa: E402
from scripts.validate_scene_geometry import scene_inputs  # noqa: E402


def pose_features(frames: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    view_ids: list[str] = []
    positions: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for frame in frames:
        matrix = np.asarray(frame["transform_matrix"], dtype=float)
        view_ids.append(str(frame["view_id"]))
        positions.append(matrix[:3, 3])
        forward = -matrix[:3, 2]
        norm = float(np.linalg.norm(forward))
        directions.append(forward / norm if norm > 0 else np.array([0.0, 0.0, -1.0]))
    return view_ids, np.vstack(positions), np.vstack(directions)


def automatic_distance_threshold(positions: np.ndarray) -> float:
    if len(positions) <= 1:
        return 1.0
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    distances[distances == 0] = np.inf
    nearest = np.min(distances, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return 1.0
    threshold = float(np.median(finite) * 2.5)
    return threshold if threshold > 0 else 1.0


def coverage_sets(
    positions: np.ndarray,
    directions: np.ndarray,
    *,
    distance_threshold: float,
    angle_threshold_degrees: float,
) -> list[set[int]]:
    cosine_threshold = math.cos(math.radians(angle_threshold_degrees))
    sets: list[set[int]] = []
    for source in range(len(positions)):
        covered = {source}
        for target in range(len(positions)):
            distance = float(np.linalg.norm(positions[source] - positions[target]))
            facing_similarity = float(np.dot(directions[source], directions[target]))
            if distance <= distance_threshold and facing_similarity >= cosine_threshold:
                covered.add(target)
        sets.append(covered)
    return sets


def coverage_ratio(selected: list[int], sets: list[set[int]]) -> float:
    if not sets:
        return 0.0
    covered: set[int] = set()
    for index in selected:
        covered.update(sets[index])
    return round(len(covered) / len(sets), 6)


def greedy_indices(k: int, sets: list[set[int]]) -> list[int]:
    selected: list[int] = []
    covered: set[int] = set()
    remaining = set(range(len(sets)))
    while remaining and len(selected) < k:
        best = max(remaining, key=lambda index: (len(sets[index] - covered), -index))
        selected.append(best)
        covered.update(sets[best])
        remaining.remove(best)
    return selected


def random_indices(k: int, count: int, seed: int) -> list[int]:
    generator = random.Random(seed)
    return sorted(generator.sample(range(count), k))


def trajectory_indices(k: int, count: int) -> list[int]:
    if k >= count:
        return list(range(count))
    if k == 1:
        return [0]
    values = np.linspace(0, count - 1, num=k)
    selected = []
    for value in values:
        index = int(round(float(value)))
        if index not in selected:
            selected.append(index)
    for index in range(count):
        if len(selected) >= k:
            break
        if index not in selected:
            selected.append(index)
    return sorted(selected[:k])


def selection_record(indices: list[int], view_ids: list[str], sets: list[set[int]]) -> dict[str, Any]:
    covered: set[int] = set()
    for index in indices:
        covered.update(sets[index])
    return {
        "status": "complete",
        "selected_count": len(indices),
        "selected_view_ids": [view_ids[index] for index in indices],
        "covered_candidate_count": len(covered),
        "uncovered_candidate_count": len(sets) - len(covered),
        "coverage_ratio": coverage_ratio(indices, sets),
    }


def unavailable_selection_record(reason: str) -> dict[str, Any]:
    return {
        "status": "not_available",
        "reason": reason,
        "selected_count": 0,
        "selected_view_ids": [],
        "covered_candidate_count": 0,
        "uncovered_candidate_count": 0,
        "coverage_ratio": None,
    }


def select_views(
    scene_dir: Path,
    *,
    k_values: list[int],
    seed: int = 0,
    distance_threshold: float | None = None,
    angle_threshold_degrees: float = 75.0,
) -> dict[str, Any]:
    scene_dir = scene_dir.resolve()
    layout, frames, _ = scene_inputs(scene_dir)
    if any(value <= 0 for value in k_values):
        raise ValueError("K values must be positive")

    view_ids: list[str] = []
    sets: list[set[int]] = []
    threshold = distance_threshold
    if frames:
        view_ids, positions, directions = pose_features(frames)
        threshold = distance_threshold if distance_threshold is not None else automatic_distance_threshold(positions)
        sets = coverage_sets(
            positions,
            directions,
            distance_threshold=threshold,
            angle_threshold_degrees=angle_threshold_degrees,
        )

    rows = []
    warnings: list[str] = []
    unavailable_reason = "No camera poses or frame records are available for view selection."
    if not view_ids:
        warnings.append(unavailable_reason)
    for requested_k in k_values:
        effective_k = min(requested_k, len(view_ids))
        if effective_k < requested_k:
            warnings.append(
                f"Requested K={requested_k}, but only {len(view_ids)} candidate views exist; used K={effective_k}."
            )
        if effective_k == 0:
            greedy = unavailable_selection_record(unavailable_reason)
            random_selection = unavailable_selection_record(unavailable_reason)
            trajectory = unavailable_selection_record(unavailable_reason)
        else:
            greedy = selection_record(greedy_indices(effective_k, sets), view_ids, sets)
            random_selection = selection_record(
                random_indices(effective_k, len(view_ids), seed + requested_k),
                view_ids,
                sets,
            )
            trajectory = selection_record(trajectory_indices(effective_k, len(view_ids)), view_ids, sets)
        rows.append({
            "requested_k": requested_k,
            "effective_k": effective_k,
            "comparison_status": "complete" if effective_k > 0 else "not_available",
            "coverage_greedy": greedy,
            "random": random_selection,
            "trajectory": trajectory,
        })

    available_ratios = {
        strategy: [row[strategy]["coverage_ratio"] for row in rows if row[strategy]["coverage_ratio"] is not None]
        for strategy in ("coverage_greedy", "random", "trajectory")
    }
    return {
        "schema_version": "semanticsplat.view_selection.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_id": scene_dir.name,
        "scene_path": display_path(scene_dir),
        "layout": layout,
        "candidate_view_count": len(view_ids),
        "comparison_available": bool(view_ids),
        "coverage_proxy": {
            "name": "pose_space_distance_and_orientation",
            "status": "available" if view_ids else "not_available",
            "description": (
                "A candidate is covered when a selected camera is spatially nearby and has a similar forward direction. "
                "This is not NoField, voxel coverage, object coverage, or rendered-surface visibility."
            ),
            "distance_threshold": round(float(threshold), 6) if threshold is not None else None,
            "angle_threshold_degrees": angle_threshold_degrees,
        },
        "coverage_statistics": {
            "candidate_view_count": len(view_ids),
            "requested_k_values": list(k_values),
            "effective_k_values": [row["effective_k"] for row in rows],
            "comparison_run_count": sum(row["comparison_status"] == "complete" for row in rows),
            "maximum_coverage_ratio": {
                strategy: max(ratios) if ratios else None
                for strategy, ratios in available_ratios.items()
            },
        },
        "seed": seed,
        "results": rows,
        "warnings": sorted(set(warnings)),
    }


def write_batch_reports(
    scenes: list[Path],
    output_dir: Path,
    *,
    k_values: list[int],
    seed: int = 0,
    distance_threshold: float | None = None,
    angle_threshold_degrees: float = 75.0,
) -> list[tuple[Path, dict[str, Any]]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    scene_ids: set[str] = set()
    for scene in scenes:
        report = select_views(
            scene,
            k_values=k_values,
            seed=seed,
            distance_threshold=distance_threshold,
            angle_threshold_degrees=angle_threshold_degrees,
        )
        scene_id = str(report["scene_id"])
        if scene_id in scene_ids:
            raise ValueError(f"Duplicate scene ID in batch: {scene_id}")
        scene_ids.add(scene_id)
        output = output_dir / f"{scene_id}_view_selection.json"
        write_json(output, report)
        reports.append((output, report))
    return reports


def format_ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def print_report_summary(report: dict[str, Any]) -> None:
    print(
        f"scene={report['scene_id']} layout={report['layout']} "
        f"candidates={report['candidate_view_count']} comparison={report['comparison_available']}"
    )
    for row in report["results"]:
        print(
            f"K={row['requested_k']} effective={row['effective_k']} "
            f"greedy={format_ratio(row['coverage_greedy']['coverage_ratio'])} "
            f"random={format_ratio(row['random']['coverage_ratio'])} "
            f"trajectory={format_ratio(row['trajectory']['coverage_ratio'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare coverage-greedy, random, and trajectory view selection")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scene", type=Path)
    inputs.add_argument("--scenes", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="Single-scene report path")
    parser.add_argument(
        "--out",
        type=Path,
        help="Batch report directory; defaults to docs/validation/geometry",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance-threshold", type=float)
    parser.add_argument("--angle-threshold-degrees", type=float, default=75.0)
    args = parser.parse_args()
    if args.scene is not None:
        if args.out is not None:
            parser.error("--out is only valid with --scenes")
        report = select_views(
            args.scene,
            k_values=args.k,
            seed=args.seed,
            distance_threshold=args.distance_threshold,
            angle_threshold_degrees=args.angle_threshold_degrees,
        )
        if args.output:
            write_json(args.output, report)
            print(f"Wrote {args.output}")
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        print_report_summary(report)
        return

    if args.output is not None:
        parser.error("--output is only valid with --scene")
    output_dir = args.out or Path("docs/validation/geometry")
    reports = write_batch_reports(
        args.scenes,
        output_dir,
        k_values=args.k,
        seed=args.seed,
        distance_threshold=args.distance_threshold,
        angle_threshold_degrees=args.angle_threshold_degrees,
    )
    for output, report in reports:
        print(f"Wrote {output}")
        print_report_summary(report)
    print(
        f"summary scenes={len(reports)} "
        f"comparisons_available={sum(report['comparison_available'] for _, report in reports)}"
    )


if __name__ == "__main__":
    main()
