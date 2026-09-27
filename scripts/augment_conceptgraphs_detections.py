#!/usr/bin/env python3
"""Add missing caption metadata expected by pinned ConceptGraphs mapping."""
from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path


def augment_detection(path: Path) -> bool:
    with gzip.open(path, "rb") as handle:
        detection = pickle.load(handle)
    if not isinstance(detection, dict):
        raise SystemExit(f"Detection pickle is not a dict: {path}")

    class_ids = detection.get("class_id")
    classes = detection.get("classes")
    if class_ids is None or classes is None:
        raise SystemExit(f"Detection pickle lacks class_id/classes: {path}")

    labels: list[str] = []
    for index, class_id in enumerate(class_ids):
        class_name = str(classes[int(class_id)])
        labels.append(f"{class_name} {index}")

    changed = False
    if "detection_class_labels" not in detection:
        detection["detection_class_labels"] = labels
        changed = True
    if "captions" not in detection:
        detection["captions"] = [
            {"id": str(index), "name": label.rsplit(" ", 1)[0], "caption": None}
            for index, label in enumerate(labels)
        ]
        changed = True

    if changed:
        with gzip.open(path, "wb") as handle:
            pickle.dump(detection, handle)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    args = parser.parse_args()

    if args.detections.is_file():
        files = [args.detections]
    elif args.detections.is_dir():
        files = sorted(args.detections.glob("*.pkl.gz"))
    else:
        raise SystemExit(f"Missing detection path: {args.detections}")

    changed_count = 0
    for path in files:
        if augment_detection(path):
            changed_count += 1
    print(
        f"Augmented {changed_count}/{len(files)} ConceptGraphs detection file(s) "
        f"under {args.detections}"
    )


if __name__ == "__main__":
    main()
