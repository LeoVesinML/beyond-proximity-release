"""Save / load ViewJSON analysis files for a scene."""
import json
from pathlib import Path
from typing import Any, List, Optional

from backend.schemas.types import ViewJSON


def _views_dir(scene_dir: str) -> Path:
    return Path(scene_dir) / "views"


def save_view_analysis(scene_dir: str, view_id: str, view_json: ViewJSON) -> None:
    d = _views_dir(scene_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{view_id}.json"
    with open(path, "w") as fh:
        json.dump(view_json.model_dump(), fh, indent=2)


def load_view_analysis(scene_dir: str, view_id: str) -> Optional[ViewJSON]:
    path = _views_dir(scene_dir) / f"{view_id}.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return ViewJSON.model_validate(json.load(fh))


def view_analysis_exists(scene_dir: str, view_id: str) -> bool:
    return (_views_dir(scene_dir) / f"{view_id}.json").exists()


def list_view_ids(scene_dir: str) -> List[str]:
    d = _views_dir(scene_dir)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def list_view_summaries(scene_dir: str) -> List[dict]:
    """Return a summary record for every analyzed view.

    Includes the new semantic-boundary fields (facing, visible_landmarks)
    required by the semantic merge pass in spatial_graph.py.
    """
    result = []
    for vid in list_view_ids(scene_dir):
        path = _views_dir(scene_dir) / f"{vid}.json"
        try:
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        if "scene_summary" in raw:
            result.append(_legacy_summary(vid, raw))
        elif raw.get("schema_version") == "semanticsplat.captured_view_json.v1":
            result.append(_captured_summary(vid, raw))
    return result


def _legacy_summary(view_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    objects = raw.get("objects")
    landmarks = raw.get("visible_landmarks")
    return {
        "view_id": view_id,
        "summary": str(raw.get("scene_summary") or ""),
        "object_count": len(objects) if isinstance(objects, list) else 0,
        "room_type": str(raw.get("room_type") or "unknown"),
        "facing": str(raw.get("facing") or "unknown"),
        "visible_landmarks": (
            [str(value) for value in landmarks if value]
            if isinstance(landmarks, list)
            else []
        ),
    }


def _captured_summary(view_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    objects = raw.get("visible_objects")
    landmarks = raw.get("landmarks")
    regions = raw.get("visible_regions")
    semantic_text = " ".join(
        [
            str(raw.get("summary") or ""),
            *[
                str(region.get("label") or "")
                for region in regions
                if isinstance(region, dict)
            ],
        ]
    ).lower() if isinstance(regions, list) else str(raw.get("summary") or "").lower()
    room_rules = (
        ("outdoor", "outdoor"),
        ("street", "outdoor"),
        ("corridor", "corridor"),
        ("lounge", "lounge"),
        ("reception", "reception"),
        ("service", "service_area"),
        ("ballroom", "ballroom"),
        ("banquet", "ballroom"),
        ("conference", "ballroom"),
        ("stage", "stage"),
        ("theater", "stage"),
        ("auditorium", "stage"),
    )
    room_type = next(
        (value for keyword, value in room_rules if keyword in semantic_text),
        "unknown",
    )
    landmark_labels = (
        [
            str(item.get("label"))
            for item in landmarks
            if isinstance(item, dict) and item.get("label")
        ]
        if isinstance(landmarks, list)
        else []
    )
    return {
        "view_id": view_id,
        "summary": str(raw.get("summary") or ""),
        "object_count": len(objects) if isinstance(objects, list) else 0,
        "room_type": room_type,
        "facing": "unknown",
        "visible_landmarks": landmark_labels,
    }
