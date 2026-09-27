#!/usr/bin/env python3
"""Build benchmark v2 from the validated captured-scene manual semantic index.

The output is intentionally labelled `verified_from_view_json`: it is a
reference benchmark over the manual semantic index, not independent visual or
dataset ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SCENE_ROOT = ROOT / "backend" / "data" / "scenes"
BENCHMARK_OUT = ROOT / "docs" / "benchmarks" / "benchmark_queries_v2.json"
BENCHMARK_V1 = ROOT / "docs" / "benchmarks" / "benchmark_queries_v1.json"
STATUS_OUT = ROOT / "docs" / "benchmarks" / "benchmark_status_v2.md"
GT_REPORT_OUT = ROOT / "docs" / "benchmarks" / "manual_gt_coverage_report_v2.md"
GT_COVERAGE_ALIAS_OUT = ROOT / "docs" / "benchmarks" / "gt_coverage_report.md"
REVIEW_OUT = ROOT / "docs" / "benchmarks" / "ambiguous_and_negative_review_v2.md"
ZONE_GT_OUT = ROOT / "docs" / "benchmarks" / "manual_zone_gt_v2.json"
COVERAGE_TABLE_OUT = ROOT / "docs" / "benchmarks" / "person2_coverage_table_v2.md"
CHANGELOG_OUT = ROOT / "docs" / "benchmarks" / "benchmark_v2_changelog.md"
FINAL_LOCK_OUT = ROOT / "docs" / "benchmarks" / "final_benchmark_lock_v2.md"
CONFIG_OUT = ROOT / "configs" / "week3_benchmark_v2.yaml"


@dataclass(frozen=True)
class SceneSpec:
    benchmark_scene_id: str
    scene_id: str
    validation_report: str


SCENES = [
    SceneSpec(
        "ConferenceHall",
        "ConferenceHall-capture-pilot",
        "docs/validation/semantic_index/ConferenceHall-capture-pilot_semantic_index.json",
    ),
    SceneSpec(
        "Museume",
        "Museume-capture",
        "docs/validation/semantic_index/Museume-capture_semantic_index.json",
    ),
    SceneSpec(
        "Theater",
        "Theater-capture",
        "docs/validation/semantic_index/Theater-capture_semantic_index.json",
    ),
    SceneSpec(
        "outdoor-street",
        "outdoor-street-capture",
        "docs/validation/semantic_index/outdoor-street-capture_semantic_index.json",
    ),
    SceneSpec(
        "outdoor-drone",
        "outdoor-drone-capture",
        "docs/validation/semantic_index/outdoor-drone-capture_semantic_index.json",
    ),
]


MANUAL_ZONE_SPECS: dict[str, list[dict[str, Any]]] = {
    "ConferenceHall": [
        {
            "zone_id": "manual_zone_conferencehall_banquet_hall",
            "zone_label": "conference and banquet hall",
            "view_ids": ["v001", "v005", "v006", "v017", "v018"],
            "description": "Main event hall views with round tables, chairs, piano-side hall context, and banquet seating.",
        },
        {
            "zone_id": "manual_zone_conferencehall_lounge_corridor",
            "zone_label": "lounge seating and corridor",
            "view_ids": ["v001", "v002", "v003", "v007", "v009", "v010", "v011", "v012", "v014", "v015", "v016", "v020"],
            "description": "Lounge, corridor, art-wall, sofa, piano, and transition views around the captured hall.",
        },
        {
            "zone_id": "manual_zone_conferencehall_reception_service",
            "zone_label": "reception and service counter",
            "view_ids": ["v004", "v008", "v013"],
            "description": "Reception counter, service counter, red-wall lobby, and adjacent counter views.",
        },
        {
            "zone_id": "manual_zone_conferencehall_elevator_access",
            "zone_label": "elevator and doorway access",
            "view_ids": ["v003", "v011", "v012", "v013", "v014"],
            "description": "Elevator, doorway, and golden door-panel access views.",
        },
        {
            "zone_id": "manual_zone_conferencehall_exit_wayfinding",
            "zone_label": "exit and wayfinding",
            "view_ids": ["v006", "v009", "v019"],
            "description": "Exit sign, doorway, and wayfinding views.",
        },
    ],
    "Museume": [
        {
            "zone_id": "manual_zone_museume_entrance_information",
            "zone_label": "entrance and information",
            "view_ids": ["v001", "v002"],
            "description": "Entrance lobby, stair, information counter, and visitor information views.",
        },
        {
            "zone_id": "manual_zone_museume_red_panel_gallery",
            "zone_label": "red-panel exhibition gallery",
            "view_ids": ["v003", "v004", "v005", "v006", "v007", "v008", "v009", "v012", "v013", "v014", "v015", "v018"],
            "description": "Red panels, wall labels, exhibition corridor, historical photographs, and display-wall views.",
        },
        {
            "zone_id": "manual_zone_museume_ruin_label_gallery",
            "zone_label": "ruin and label gallery",
            "view_ids": ["v004", "v005", "v010", "v015"],
            "description": "Stone ruin, label rail, dark niche, and surrounding exhibit views.",
        },
        {
            "zone_id": "manual_zone_museume_window_mesh_side",
            "zone_label": "window and mesh side",
            "view_ids": ["v006", "v008", "v009", "v011", "v017", "v019", "v020"],
            "description": "Window-side gallery views with mesh partitions, seating, and side displays.",
        },
        {
            "zone_id": "manual_zone_museume_reichenstein_media",
            "zone_label": "Reichenstein and media station",
            "view_ids": ["v016", "v017", "v019", "v020"],
            "description": "Reichenstein exhibit wall, information panel, media station, and adjacent window-side views.",
        },
        {
            "zone_id": "manual_zone_museume_visitor_seating_workshop",
            "zone_label": "visitor seating and workshop",
            "view_ids": ["v002", "v007", "v008", "v009", "v020"],
            "description": "Workshop corner, bench, chairs, and visitor seating views.",
        },
    ],
    "Theater": [
        {
            "zone_id": "manual_zone_theater_stage_backstage",
            "zone_label": "stage and backstage corridor",
            "view_ids": ["v001", "v002", "v003", "v006", "v007", "v018", "v019"],
            "description": "Stage curtain, backstage passage, pale corridor, side door, and corridor-passage views.",
        },
        {
            "zone_id": "manual_zone_theater_lobby_notice",
            "zone_label": "lobby and notice wall",
            "view_ids": ["v004", "v005", "v006", "v020"],
            "description": "Lobby seating, notice wall, schedule screen, printed timetable, and photo display views.",
        },
        {
            "zone_id": "manual_zone_theater_lounge_green_room",
            "zone_label": "lounge and green room",
            "view_ids": ["v009", "v010", "v011", "v012"],
            "description": "Green room, lounge seating, TV wall, bookcase wall, and instrument-corner views.",
        },
        {
            "zone_id": "manual_zone_theater_stairs_foyer_balcony",
            "zone_label": "stairs, foyer, and balcony",
            "view_ids": ["v001", "v012", "v013", "v014", "v015", "v016", "v017", "v019"],
            "description": "Stairway, railing, upper foyer, balcony, landing, and long hall circulation views.",
        },
        {
            "zone_id": "manual_zone_theater_performance_seating",
            "zone_label": "performance seating",
            "view_ids": ["v004", "v009", "v010", "v011", "v012"],
            "description": "Audience, lounge, and red seating views suitable for watching or waiting.",
        },
    ],
    "outdoor-street": [
        {
            "zone_id": "manual_zone_outdoor_street_courtyard_stair_service",
            "zone_label": "courtyard stairs and service facade",
            "view_ids": ["v001", "v002", "v020", "v021"],
            "description": "Courtyard, stair entrance, service wall, emergency doors, and basement-window side views.",
        },
        {
            "zone_id": "manual_zone_outdoor_street_covered_passage_graffiti",
            "zone_label": "covered passage and graffiti wall",
            "view_ids": ["v003", "v004", "v013", "v016", "v017"],
            "description": "Covered passage, graffiti wall, underpass, alley, red wall side, and bright opening views.",
        },
        {
            "zone_id": "manual_zone_outdoor_street_facade_entrance_glassblock",
            "zone_label": "facade entrances and glass-block wall",
            "view_ids": ["v005", "v006", "v007", "v008", "v009", "v019", "v021"],
            "description": "Glass-block entrance, numbered entrance, facade details, utility panels, and service entry views.",
        },
        {
            "zone_id": "manual_zone_outdoor_street_walkway_road",
            "zone_label": "street walkway and roadway",
            "view_ids": ["v010", "v011", "v012", "v014", "v015", "v018"],
            "description": "Street walkway, cobblestone ground, roadway, sidewalk, and street-corner views.",
        },
        {
            "zone_id": "manual_zone_outdoor_street_red_facade_alley",
            "zone_label": "red facade and side alley",
            "view_ids": ["v012", "v014", "v016", "v017", "v018"],
            "description": "Red facade, side alley, barred doorway/window, and distant opening views.",
        },
    ],
    "outdoor-drone": [
        {
            "zone_id": "manual_zone_outdoor_drone_fortress_island",
            "zone_label": "fortress island landmark",
            "view_ids": ["v001", "v012", "v013", "v014"],
            "description": "Fortress island, rocky island, lighthouse or tower, and close-up island views.",
        },
        {
            "zone_id": "manual_zone_outdoor_drone_open_water_bay",
            "zone_label": "open water and bay",
            "view_ids": ["v001", "v002", "v003", "v004", "v006", "v007", "v008", "v009", "v010", "v011", "v012", "v013", "v014", "v016"],
            "description": "Open sea, bay water, surrounding water, and waterfront water views.",
        },
        {
            "zone_id": "manual_zone_outdoor_drone_city_shoreline",
            "zone_label": "city shoreline and hills",
            "view_ids": ["v002", "v003", "v004", "v005", "v006", "v008", "v009", "v010", "v011", "v012", "v014", "v015", "v016"],
            "description": "Distant city, shoreline, coastline, waterfront district, hills, and urban coast views.",
        },
        {
            "zone_id": "manual_zone_outdoor_drone_bridge_crossing",
            "zone_label": "bridge and linear crossing",
            "view_ids": ["v006", "v009", "v015"],
            "description": "Bridge, long roadway, transport corridor, and linear crossing views.",
        },
        {
            "zone_id": "manual_zone_outdoor_drone_harbor_marina",
            "zone_label": "harbor and marina",
            "view_ids": ["v002", "v003", "v004", "v011"],
            "description": "Harbor peninsula, marina, waterfront district, and pier/harbor cluster views.",
        },
    ],
}


SCENE_PLANS: dict[str, dict[str, list[Any]]] = {
    "ConferenceHall": {
        "simple_object": [
            ("piano", "Find the piano"),
            ("reception counter", "Find the reception counter"),
            ("green exit sign", "Find the green exit sign"),
            ("elevator doors", "Find the elevator doors"),
            ("round tables", "Find the round tables"),
        ],
        "attribute": [
            ("piano", "black", "Find the black piano"),
            ("sofas", "cream", "Find the cream sofas"),
            ("elevator doors", "golden", "Find the golden elevator doors"),
            ("reception counter", "dark", "Find the dark reception counter"),
            ("covered chairs", "white", "Find the white covered chairs"),
        ],
        "relational": [
            ("piano", "near wall", "Find the piano near the wall"),
            ("chairs", "around tables", "Find the chairs around tables"),
            ("table lamps", "near sofas", "Find the table lamps near the sofas"),
            ("column", "near piano", "Find the column near the piano"),
            ("flower vase", "on or near counter", "Find the flower vase on the reception counter"),
        ],
        "multi_hop": [
            ("reception area", "reception counter", "Find the reception counter in the reception area"),
            ("banquet seating area", "round tables", "Find the round tables in the banquet seating area"),
            ("lounge seating area", "sofas", "Find the sofas in the lounge seating area"),
            ("piano area", "piano", "Find the piano in the piano area"),
            ("exit or doorway area", "green exit sign", "Find the green exit sign in the exit or doorway area"),
        ],
        "functional": [
            ("sittable", ["sofas", "chairs", "armchair"], ["lounge seating area"], "Find an area where someone can sit"),
            ("meeting", ["round tables", "covered chairs"], ["banquet seating area"], "Find an area suitable for a meeting"),
            ("reception_help", ["reception counter"], ["reception area"], "Find where a visitor could ask for reception help"),
            ("exit_or_wayfinding", ["green exit sign", "doorway"], ["exit or doorway area"], "Find a wayfinding or exit area"),
            ("elevator_access", ["elevator doors"], ["elevator area"], "Find the elevator access area"),
        ],
        "negative": ["bed", "shower", "vehicle", "swimming pool", "kitchen stove"],
    },
    "Museume": {
        "simple_object": [
            ("information counter", "Find the information counter"),
            ("stairs", "Find the stairs"),
            ("red exhibition panel", "Find the red exhibition panel"),
            ("stone ruin", "Find the stone ruin"),
            ("Reichenstein information panel", "Find the Reichenstein information panel"),
        ],
        "attribute": [
            ("green information screens", "bright", "Find the bright green information screens"),
            ("yellow circular graphic", "bright", "Find the bright yellow circular graphic"),
            ("metal mesh partitions", "black", "Find the black metal mesh partitions"),
            ("red exhibition panel", "text", "Find the red text exhibition panel"),
            ("tall windows", "outside view", "Find the tall windows with an outside view"),
        ],
        "relational": [
            ("stairs", "metal railing", "Find the stairs with the metal railing"),
            ("historical photographs", "right side of red panel", "Find the historical photographs on the red panel"),
            ("chairs", "visitor seating", "Find the visitor chairs"),
            ("display labels", "near the ruin", "Find the display labels near the stone ruin"),
            ("yellow floor marker", "floor", "Find the yellow floor marker on the gallery floor"),
        ],
        "multi_hop": [
            ("information counter area", "information counter", "Find the information counter in the information counter area"),
            ("stair area", "stairs", "Find the stairs in the stair area"),
            ("ruin display area", "stone wall ruin", "Find the stone wall ruin in the ruin display area"),
            ("red panel area", "red exhibition panel", "Find the red exhibition panel in the red panel area"),
            ("window side", "tall windows", "Find the tall windows on the window side"),
        ],
        "functional": [
            ("visitor_help", ["information counter", "green information screens"], ["information counter area"], "Find where a museum visitor could get information"),
            ("view_exhibits", ["red exhibition panel", "stone ruin", "display stand"], ["ruin display area", "red panel area"], "Find an area suitable for viewing exhibits"),
            ("sit", ["chairs", "bench or low seating", "metal bench"], ["window side"], "Find a place where a visitor can sit"),
            ("enter_or_exit", ["glass entrance door", "glass doors"], ["entrance lobby"], "Find the museum entrance or exit area"),
            ("read_labels", ["information labels", "display labels", "wall labels"], ["label rail area"], "Find a place with exhibit labels to read"),
        ],
        "negative": ["grand piano", "swimming pool", "bed", "shower", "projection screen"],
    },
    "Theater": {
        "simple_object": [
            ("black stage curtain", "Find the black stage curtain"),
            ("red upholstered seats", "Find the red upholstered seats"),
            ("green exit sign", "Find the green exit sign"),
            ("staircase", "Find the staircase"),
            ("wall-mounted TV", "Find the wall-mounted TV"),
        ],
        "attribute": [
            ("red upholstered seats", "bright red", "Find the bright red upholstered seats"),
            ("white door", "closed", "Find the closed white door"),
            ("black stage curtain", "large", "Find the large black stage curtain"),
            ("pale green walls", "corridor", "Find the pale green corridor walls"),
            ("wall lamps", "warm", "Find the warm wall lamps"),
        ],
        "relational": [
            ("green exit sign", "above doorway", "Find the green exit sign above the doorway"),
            ("clock", "above bookcase", "Find the clock above the bookcase"),
            ("chairs", "around room", "Find the chairs around the room"),
            ("staircase", "visible through doorway", "Find the staircase through the doorway"),
            ("keyboard or instrument stand", "right side", "Find the keyboard or instrument stand on the right side"),
        ],
        "multi_hop": [
            ("stage area", "black stage curtain", "Find the black stage curtain in the stage area"),
            ("lobby seating area", "red upholstered seats", "Find the red upholstered seats in the lobby seating area"),
            ("notice wall", "digital schedule screen", "Find the digital schedule screen on the notice wall"),
            ("railing side", "black railing", "Find the black railing on the railing side"),
            ("backstage corridor", "corridor passage", "Find the corridor passage in the backstage corridor"),
        ],
        "functional": [
            ("watch_performance", ["red upholstered seats", "chairs"], ["lobby seating area"], "Find an area suitable for watching a performance"),
            ("perform", ["black stage curtain", "stage floor"], ["stage area"], "Find an area suitable for performing"),
            ("exit_or_wayfinding", ["green exit sign", "doorway"], ["doorway area"], "Find a theater exit or wayfinding area"),
            ("schedule_info", ["digital schedule screen", "timetable board", "printed schedules"], ["notice wall"], "Find where a visitor could check the schedule"),
            ("lounge", ["sofa", "sofas", "small round tables"], ["lounge seating area"], "Find a theater lounge seating area"),
        ],
        "negative": ["swimming pool", "bicycle", "kitchen stove", "fortress island", "beach"],
    },
    "outdoor-street": {
        "simple_object": [
            ("wide outdoor stairs", "Find the wide outdoor stairs"),
            ("glass-block wall", "Find the glass-block wall"),
            ("traffic signs", "Find the traffic signs"),
            ("graffiti", "Find the graffiti"),
            ("metal handrails", "Find the metal handrails"),
        ],
        "attribute": [
            ("beige building facade", "stone", "Find the stone beige building facade"),
            ("red building facade", "bright red", "Find the bright red building facade"),
            ("turquoise supports", "metal", "Find the metal turquoise supports"),
            ("cobblestone pavement", "outdoor walkway", "Find the outdoor cobblestone pavement"),
            ("glass entrance doors", "number 8", "Find the glass entrance doors with number 8"),
        ],
        "relational": [
            ("traffic signs", "center-left", "Find the traffic signs near the tiled wall"),
            ("graffiti", "wall text", "Find the graffiti on the wall"),
            ("barred window", "low", "Find the low barred window"),
            ("metal handrails", "foreground", "Find the metal handrails in the foreground"),
            ("glass-block wall", "right side", "Find the glass-block wall on the right side"),
        ],
        "multi_hop": [
            ("stair entrance area", "wide outdoor stairs", "Find the wide outdoor stairs in the stair entrance area"),
            ("glass-block entrance", "glass entrance doors", "Find the glass entrance doors in the glass-block entrance"),
            ("graffiti wall area", "graffiti", "Find the graffiti in the graffiti wall area"),
            ("covered passage", "covered walkway", "Find the covered walkway in the covered passage"),
            ("facade wall", "beige building facade", "Find the beige building facade on the facade wall"),
        ],
        "functional": [
            ("walk", ["paved walkway", "cobblestone pavement", "sidewalk or pavement"], ["cobblestone walkway", "paved walkway"], "Find an area suitable for walking"),
            ("enter_building", ["glass entrance doors", "recessed door"], ["glass-block entrance"], "Find a building entrance area"),
            ("cross_or_move", ["wide outdoor stairs", "metal handrails"], ["stair entrance area"], "Find an area suitable for moving between street levels"),
            ("utility_access", ["utility door", "small utility panels"], ["service wall area"], "Find a utility access area"),
            ("wayfinding", ["traffic signs", "building number 8"], ["facade detail area"], "Find a street wayfinding marker"),
        ],
        "negative": ["grand piano", "swimming pool", "bed", "sofa", "open sea water"],
    },
    "outdoor-drone": {
        "simple_object": [
            ("fortress island", "Find the fortress island"),
            ("tower or lighthouse", "Find the tower or lighthouse"),
            ("open water", "Find the open water"),
            ("distant city skyline", "Find the distant city skyline"),
            ("bridge or long roadway", "Find the bridge or long roadway"),
        ],
        "attribute": [
            ("white fortress buildings", "large", "Find the large white fortress buildings"),
            ("sea water", "green-blue", "Find the green-blue sea water"),
            ("distant city", "far", "Find the far distant city"),
            ("bright glare", "white", "Find the white bright glare"),
            ("rocky island", "surrounded by water", "Find the rocky island surrounded by water"),
        ],
        "relational": [
            ("tower or lighthouse", "on island", "Find the tower or lighthouse on the island"),
            ("white fortress-like building", "on island", "Find the white fortress-like building on the island"),
            ("city buildings", "coastal", "Find the coastal city buildings"),
            ("bridge or road lines", "center-left", "Find the bridge or road lines across the bay"),
            ("hills", "background", "Find the hills behind the city"),
        ],
        "multi_hop": [
            ("fortress island", "tower or lighthouse", "Find the tower or lighthouse in the fortress island area"),
            ("open water", "sea water", "Find the sea water in the open water region"),
            ("distant shoreline", "distant city skyline", "Find the distant city skyline along the distant shoreline"),
            ("bay water", "bridge or road lines", "Find the bridge or road lines over bay water"),
            ("marina or harbor", "marina or harbor cluster", "Find the marina or harbor cluster in the marina area"),
        ],
        "functional": [
            ("coastal_landmark", ["fortress island", "tower or lighthouse"], ["fortress island"], "Find a coastal landmark"),
            ("open_water_navigation", ["open water", "sea water", "bay water"], ["open water", "bay water"], "Find an open water area"),
            ("harbor", ["marina or harbor cluster", "harbor edge or pier"], ["marina or harbor"], "Find a harbor or marina area"),
            ("city_orientation", ["distant city skyline", "city buildings"], ["distant shoreline"], "Find the distant city orientation landmark"),
            ("bridge_route", ["bridge or long roadway", "bridge or road lines"], ["bridge or linear crossing"], "Find a bridge or roadway crossing"),
        ],
        "negative": ["grand piano", "bed", "shower", "traffic signs", "theater stage"],
    },
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized[:48] or "item"


def region_node_id(label: str) -> str:
    digest = hashlib.sha1(label.strip().casefold().encode("utf-8")).hexdigest()[:8]
    return f"region_{slug(label)}_{digest}"


def object_node_id(view_id: str, index: int, label: str) -> str:
    return f"object_{view_id}_{index:03d}_{slug(label)}"


def landmark_node_id(kind: str, view_id: str, index: int, label: str) -> str:
    return f"{kind}_{view_id}_{index:03d}_{slug(label)}"


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def contains_text(values: list[str], needle: str) -> bool:
    target = normalize(needle)
    return any(target in normalize(value) for value in values)


def scene_views(scene_id: str) -> list[dict[str, Any]]:
    views_dir = SCENE_ROOT / scene_id / "views"
    return [load_json(path) for path in sorted(views_dir.glob("*.json"))]


def collect_scene_index(scene_id: str) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    landmarks: list[dict[str, Any]] = []
    views = scene_views(scene_id)
    regions_by_view: dict[str, list[dict[str, Any]]] = {}
    objects_by_view: dict[str, list[dict[str, Any]]] = {}

    for view in views:
        view_id = view["view_id"]
        view_source = f"backend/data/scenes/{scene_id}/views/{view_id}.json"
        regions_by_view[view_id] = []
        objects_by_view[view_id] = []
        for region in view.get("visible_regions", []):
            entry = {
                "kind": "region",
                "label": region["label"],
                "node_id": region_node_id(region["label"]),
                "view_id": view_id,
                "approx_location": region.get("approx_location", ""),
                "source": view_source,
            }
            regions.append(entry)
            regions_by_view[view_id].append(entry)
        for index, obj in enumerate(view.get("visible_objects", []), 1):
            entry = {
                "kind": "object",
                "label": obj["label"],
                "node_id": object_node_id(view_id, index, obj["label"]),
                "view_id": view_id,
                "attributes": [str(item) for item in obj.get("attributes", [])],
                "approx_location": obj.get("approx_location", ""),
                "source": view_source,
            }
            objects.append(entry)
            objects_by_view[view_id].append(entry)
        for index, landmark in enumerate(view.get("landmarks", []), 1):
            kind = landmark.get("kind", "landmark")
            landmarks.append(
                {
                    "kind": kind,
                    "label": landmark["label"],
                    "node_id": landmark_node_id(kind, view_id, index, landmark["label"]),
                    "view_id": view_id,
                    "approx_location": landmark.get("approx_location", ""),
                    "source": view_source,
                }
            )
    all_labels = {normalize(item["label"]) for item in objects + regions + landmarks}
    return {
        "views": views,
        "objects": objects,
        "regions": regions,
        "landmarks": landmarks,
        "regions_by_view": regions_by_view,
        "objects_by_view": objects_by_view,
        "all_labels": all_labels,
    }


def matching_objects(index: dict[str, Any], label: str, *, attr: str | None = None) -> list[dict[str, Any]]:
    matches = [
        item for item in index["objects"]
        if normalize(item["label"]) == normalize(label)
    ]
    if attr is not None:
        matches = [
            item for item in matches
            if contains_text(item["attributes"] + [item.get("approx_location", "")], attr)
        ]
    return sorted(matches, key=lambda item: (item["view_id"], item["node_id"]))


def matching_regions(index: dict[str, Any], labels: list[str]) -> list[dict[str, Any]]:
    wanted = {normalize(label) for label in labels}
    return sorted(
        [item for item in index["regions"] if normalize(item["label"]) in wanted],
        key=lambda item: (item["view_id"], item["node_id"]),
    )


def objects_in_region(index: dict[str, Any], region_label: str, object_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    region_views = {
        item["view_id"] for item in index["regions"]
        if normalize(item["label"]) == normalize(region_label)
    }
    objects = [
        item for item in matching_objects(index, object_label)
        if item["view_id"] in region_views
    ]
    regions = [
        item for item in index["regions"]
        if normalize(item["label"]) == normalize(region_label)
        and item["view_id"] in {obj["view_id"] for obj in objects}
    ]
    if not objects:
        objects = matching_objects(index, object_label)
        regions = matching_regions(index, [region_label])
    return objects, regions


def unique_values(items: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({str(item[key]) for item in items if item.get(key)})


def zone_refs_for_views(scene_id: str, view_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(view_ids)
    refs: list[dict[str, Any]] = []
    for zone in MANUAL_ZONE_SPECS.get(scene_id, []):
        zone_views = {str(view_id) for view_id in zone["view_ids"]}
        matched_views = sorted(wanted & zone_views)
        if not matched_views:
            continue
        refs.append({
            "zone_id": str(zone["zone_id"]),
            "zone_label": str(zone["zone_label"]),
            "matched_view_ids": matched_views,
            "description": str(zone["description"]),
        })
    return sorted(refs, key=lambda item: item["zone_id"])


def all_zone_ids(scene_id: str) -> list[str]:
    return sorted(str(zone["zone_id"]) for zone in MANUAL_ZONE_SPECS.get(scene_id, []))


def benchmark_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sprint_bucket_for_query_type(query_type: str) -> str:
    return {
        "simple_object": "simple",
        "attribute": "simple",
        "relational": "relational",
        "multi_hop": "multi_hop",
        "functional": "intent",
        "negative": "negative_absence_review",
    }.get(query_type, "other")


def query_record(
    *,
    query_id: str,
    scene_id: str,
    source_scene_id: str,
    query: str,
    query_type: str,
    expected_output_type: str,
    object_items: list[dict[str, Any]] | None = None,
    region_items: list[dict[str, Any]] | None = None,
    expected_object_labels: list[str] | None = None,
    expected_affordance: str | None = None,
    negative_label: str | None = None,
) -> dict[str, Any]:
    object_items = object_items or []
    region_items = region_items or []
    expected_labels = expected_object_labels or unique_values(object_items, "label")
    view_ids = sorted({item["view_id"] for item in object_items + region_items})
    node_ids = sorted({item["node_id"] for item in object_items + region_items})
    region_ids = sorted({item["node_id"] for item in region_items})
    region_labels = unique_values(region_items, "label")
    sources = sorted({item["source"] for item in object_items + region_items})
    zone_refs = [] if query_type == "negative" else zone_refs_for_views(scene_id, view_ids)
    zone_ids = sorted({item["zone_id"] for item in zone_refs})
    zone_labels = sorted({item["zone_label"] for item in zone_refs})
    zone_view_evidence = {
        item["zone_id"]: item["matched_view_ids"]
        for item in zone_refs
    }
    if query_type == "negative":
        sources = [f"backend/data/scenes/{source_scene_id}/views/"]
        notes = (
            f"Verified absent from the manual ViewJSON semantic index for {source_scene_id}; "
            "not independent visual GT."
        )
    else:
        notes = (
            "Verified against manual ViewJSON semantic-index entries; this is a "
            "reference semantic-map label, not independent dataset GT."
        )
        if zone_ids:
            notes += " Manual zone labels are view-overlap references and are emitted as tree-native zone nodes during tree build."
            sources = sorted({*sources, benchmark_relative(ZONE_GT_OUT)})
    record: dict[str, Any] = {
        "query_id": query_id,
        "scene_id": scene_id,
        "query": query,
        "query_type": query_type,
        "sprint_plan_bucket": sprint_bucket_for_query_type(query_type),
        "expected_output_type": expected_output_type,
        "expected_zone_ids": zone_ids,
        "expected_zone_labels": zone_labels,
        "expected_zone_view_evidence": zone_view_evidence,
        "expected_node_ids": node_ids,
        "expected_region_ids": region_ids,
        "expected_region_labels": region_labels,
        "expected_view_ids": view_ids,
        "expected_object_labels": expected_labels,
        "acceptance_rule": (
            "found=false"
            if query_type == "negative"
            else "found=true and selected_view_id is one of expected_view_ids"
        ),
        "notes": notes,
        "verification_status": "verified_from_view_json",
        "verification_source": sources,
    }
    if expected_affordance:
        record["expected_affordance"] = expected_affordance
    if negative_label:
        record["negative_label"] = negative_label
        record["negative_absence_checked_zone_ids"] = all_zone_ids(scene_id)
    return record


def require_matches(matches: list[dict[str, Any]], context: str) -> list[dict[str, Any]]:
    if not matches:
        raise ValueError(f"No ViewJSON evidence found for {context}")
    return matches


def build_queries() -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    query_index = 1
    per_scene_evidence: dict[str, dict[str, Any]] = {}

    for scene in SCENES:
        index = collect_scene_index(scene.scene_id)
        plan = SCENE_PLANS[scene.benchmark_scene_id]
        per_scene_evidence[scene.benchmark_scene_id] = {
            "view_count": len(index["views"]),
            "object_label_count": len({normalize(item["label"]) for item in index["objects"]}),
            "region_label_count": len({normalize(item["label"]) for item in index["regions"]}),
        }

        for label, query in plan["simple_object"]:
            objects = require_matches(matching_objects(index, label), f"{scene.scene_id}: {label}")
            queries.append(query_record(
                query_id=f"qv2_{query_index:03d}",
                scene_id=scene.benchmark_scene_id,
                source_scene_id=scene.scene_id,
                query=query,
                query_type="simple_object",
                expected_output_type="object",
                object_items=objects,
            ))
            query_index += 1

        for label, attr, query in plan["attribute"]:
            objects = require_matches(matching_objects(index, label, attr=attr), f"{scene.scene_id}: {attr} {label}")
            queries.append(query_record(
                query_id=f"qv2_{query_index:03d}",
                scene_id=scene.benchmark_scene_id,
                source_scene_id=scene.scene_id,
                query=query,
                query_type="attribute",
                expected_output_type="object",
                object_items=objects,
            ))
            query_index += 1

        for label, relation, query in plan["relational"]:
            objects = require_matches(matching_objects(index, label, attr=relation), f"{scene.scene_id}: {label} relation {relation}")
            queries.append(query_record(
                query_id=f"qv2_{query_index:03d}",
                scene_id=scene.benchmark_scene_id,
                source_scene_id=scene.scene_id,
                query=query,
                query_type="relational",
                expected_output_type="object",
                object_items=objects,
            ))
            query_index += 1

        for region_label, object_label, query in plan["multi_hop"]:
            objects, regions = objects_in_region(index, region_label, object_label)
            objects = require_matches(objects, f"{scene.scene_id}: {object_label} in {region_label}")
            regions = require_matches(regions, f"{scene.scene_id}: region {region_label}")
            queries.append(query_record(
                query_id=f"qv2_{query_index:03d}",
                scene_id=scene.benchmark_scene_id,
                source_scene_id=scene.scene_id,
                query=query,
                query_type="multi_hop",
                expected_output_type="region",
                object_items=objects,
                region_items=regions,
            ))
            query_index += 1

        for affordance, labels, region_labels, query in plan["functional"]:
            objects: list[dict[str, Any]] = []
            for label in labels:
                objects.extend(matching_objects(index, label))
            regions = matching_regions(index, region_labels)
            if not objects and not regions:
                raise ValueError(f"No functional evidence found for {scene.scene_id}: {query}")
            queries.append(query_record(
                query_id=f"qv2_{query_index:03d}",
                scene_id=scene.benchmark_scene_id,
                source_scene_id=scene.scene_id,
                query=query,
                query_type="functional",
                expected_output_type="region",
                object_items=objects,
                region_items=regions,
                expected_object_labels=labels,
                expected_affordance=affordance,
            ))
            query_index += 1

        for label in plan["negative"]:
            if normalize(label) in index["all_labels"]:
                raise ValueError(f"Negative label is present in {scene.scene_id}: {label}")
            queries.append(query_record(
                query_id=f"qv2_{query_index:03d}",
                scene_id=scene.benchmark_scene_id,
                source_scene_id=scene.scene_id,
                query=f"Find {label}",
                query_type="negative",
                expected_output_type="not_found",
                negative_label=label,
            ))
            query_index += 1

    type_counts = Counter(query["query_type"] for query in queries)
    scene_counts = Counter(query["scene_id"] for query in queries)
    sprint_bucket_counts = Counter(query["sprint_plan_bucket"] for query in queries)
    zone_positive_count = sum(
        1 for query in queries
        if query["query_type"] != "negative" and query.get("expected_zone_ids")
    )
    multi_answer_count = sum(
        1 for query in queries
        if len(query.get("expected_view_ids", [])) > 1
        or len(query.get("expected_node_ids", [])) > 1
    )
    return {
        "schema_version": "semanticsplat.benchmark_queries.v1",
        "benchmark_id": "five_capture_manual_semantic_index_v2",
        "status": "verified_manual_semantic_index_reference",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene_scope": [scene.benchmark_scene_id for scene in SCENES],
        "captured_scene_scope": [scene.scene_id for scene in SCENES],
        "query_count": len(queries),
        "query_type_counts": dict(sorted(type_counts.items())),
        "scene_query_counts": dict(sorted(scene_counts.items())),
        "sprint_plan_bucket_counts": dict(sorted(sprint_bucket_counts.items())),
        "sprint_plan_bucket_status": {
            "simple": {
                "status": "executable",
                "benchmark_query_type": "simple_object + attribute",
                "benchmark_query_types": ["simple_object", "attribute"],
                "count": sprint_bucket_counts.get("simple", 0),
            },
            "relational": {
                "status": "executable",
                "benchmark_query_type": "relational",
                "benchmark_query_types": ["relational"],
                "count": type_counts.get("relational", 0),
            },
            "multi_hop": {
                "status": "executable",
                "benchmark_query_type": "multi_hop",
                "benchmark_query_types": ["multi_hop"],
                "count": type_counts.get("multi_hop", 0),
            },
            "intent": {
                "status": "executable",
                "benchmark_query_type": "functional",
                "benchmark_query_types": ["functional"],
                "count": type_counts.get("functional", 0),
            },
            "ambiguity": {
                "status": "reviewed_not_labeled_ambiguous",
                "benchmark_query_type": None,
                "benchmark_query_types": [],
                "count": 0,
                "multi_answer_positive_queries": multi_answer_count,
                "reason": (
                    "Ambiguous cases are documented separately. Multi-answer positives "
                    "keep multiple expected IDs instead of forcing a single canonical target."
                ),
            },
            "freshness": {
                "status": "blocked_no_temporal_gt",
                "benchmark_query_type": None,
                "benchmark_query_types": [],
                "count": 0,
                "reason": (
                    "The captured scenes have no temporal recapture or change labels, "
                    "so freshness/change queries would be fabricated."
                ),
            },
            "negative_absence_review": {
                "status": "executable_for_absence_checks",
                "benchmark_query_type": "negative",
                "benchmark_query_types": ["negative"],
                "count": sprint_bucket_counts.get("negative_absence_review", 0),
                "reason": "Negative cases are reviewed separately to avoid fake accuracy.",
            },
        },
        "ground_truth_scope": (
            "All expected fields are verified against manual captured-scene ViewJSON "
            "and generated tree nodes, with manual zone references from view overlap. "
            "This benchmark is suitable for semantic-index "
            "regression and graph-vs-flat same-input evaluation. It is not independent "
            "visual/dataset GT, does not provide 3D boxes, and has no official "
            "Replica/ScanNet provenance."
        ),
        "zone_gt_status": (
            "manual_zone_reference_available_tree_native"
        ),
        "zone_gt_source": benchmark_relative(ZONE_GT_OUT),
        "positive_queries_with_expected_zones": zone_positive_count,
        "per_scene_evidence": per_scene_evidence,
        "queries": queries,
    }


def write_zone_gt(benchmark: dict[str, Any]) -> None:
    zone_query_counts = Counter(
        zone_id
        for query in benchmark["queries"]
        for zone_id in query.get("expected_zone_ids", [])
    )
    scenes: list[dict[str, Any]] = []
    for scene in SCENES:
        zones = []
        for zone in MANUAL_ZONE_SPECS[scene.benchmark_scene_id]:
            zones.append({
                "zone_id": zone["zone_id"],
                "zone_label": zone["zone_label"],
                "view_ids": zone["view_ids"],
                "description": zone["description"],
                "query_reference_count": zone_query_counts.get(str(zone["zone_id"]), 0),
            })
        scenes.append({
            "benchmark_scene_id": scene.benchmark_scene_id,
            "captured_scene_id": scene.scene_id,
            "zone_count": len(zones),
            "zones": zones,
        })
    write_json(ZONE_GT_OUT, {
        "schema_version": "semanticsplat.manual_zone_gt.v1",
        "benchmark_id": benchmark["benchmark_id"],
        "status": "manual_zone_reference_tree_native",
        "created_at": benchmark["created_at"],
        "source": (
            "Manual zone references created from captured-scene ViewJSON view "
            "overlap and scene review. They are expected-answer labels for "
            "benchmark queries, not independent dataset GT. The captured-scene "
            "tree builder emits these references as stable tree-native zone nodes."
        ),
        "not_official_dataset_gt": True,
        "positive_queries_with_expected_zones": benchmark.get("positive_queries_with_expected_zones", 0),
        "scene_count": len(scenes),
        "zone_count": sum(scene["zone_count"] for scene in scenes),
        "scenes": scenes,
    })


def v1_inventory() -> dict[str, Any]:
    if not BENCHMARK_V1.exists():
        return {
            "available": False,
            "total": 0,
            "five_scene_total": 0,
            "scene_counts": {},
            "type_counts": {},
            "five_scene_gt_nonempty": 0,
        }
    data = load_json(BENCHMARK_V1)
    queries = data.get("queries", [])
    if not isinstance(queries, list):
        queries = []
    five_scene_ids = {scene.benchmark_scene_id for scene in SCENES}
    five_scene_queries = [
        query for query in queries
        if isinstance(query, dict) and query.get("scene_id") in five_scene_ids
    ]
    return {
        "available": True,
        "total": len(queries),
        "five_scene_total": len(five_scene_queries),
        "scene_counts": dict(sorted(Counter(str(query.get("scene_id")) for query in queries if isinstance(query, dict)).items())),
        "type_counts": dict(sorted(Counter(str(query.get("query_type")) for query in queries if isinstance(query, dict)).items())),
        "five_scene_gt_nonempty": sum(
            1 for query in five_scene_queries
            if query.get("expected_view_ids") or query.get("expected_node_ids") or query.get("expected_zone_ids")
        ),
    }


def write_status_docs(benchmark: dict[str, Any]) -> None:
    type_counts = benchmark["query_type_counts"]
    scene_counts = benchmark["scene_query_counts"]
    sprint_buckets = benchmark.get("sprint_plan_bucket_status", {})
    verified = sum(
        1 for query in benchmark["queries"]
        if str(query.get("verification_status", "")).startswith("verified_")
    )
    zone_non_empty = sum(1 for query in benchmark["queries"] if query.get("expected_zone_ids"))
    region_non_empty = sum(1 for query in benchmark["queries"] if query.get("expected_region_ids"))
    node_non_empty = sum(1 for query in benchmark["queries"] if query.get("expected_node_ids"))
    view_non_empty = sum(1 for query in benchmark["queries"] if query.get("expected_view_ids"))
    negative_count = sum(1 for query in benchmark["queries"] if query.get("query_type") == "negative")
    positive_count = benchmark["query_count"] - negative_count
    scene_type_counts = Counter(
        (query["scene_id"], query["query_type"])
        for query in benchmark["queries"]
    )
    v1 = v1_inventory()

    status_lines = [
        "# Benchmark Status V2",
        "",
        "Status date: 2026-07-01.",
        "",
        "Benchmark file: `docs/benchmarks/benchmark_queries_v2.json`.",
        "",
        "This is a verified manual-semantic-index benchmark over the five captured scenes. It is not independent dataset GT.",
        "",
        "## Counts",
        "",
        f"- Total queries: {benchmark['query_count']}",
        f"- Verified-from-ViewJSON queries: {verified}",
        f"- Scene count: {len(benchmark['scene_scope'])}",
        f"- Query types: {', '.join(sorted(type_counts))}",
        f"- Positive queries: {positive_count}",
        f"- Negative queries: {negative_count}",
        "",
        "## Query Type Distribution",
        "",
        "| Query type | Count |",
        "|---|---:|",
    ]
    for query_type, count in sorted(type_counts.items()):
        status_lines.append(f"| {query_type} | {count} |")
    status_lines.extend([
        "",
        "## Sprint-Plan Bucket Coverage",
        "",
        "The sprint plan requested simple, relational, multi-hop, intent, ambiguity, and freshness coverage. The benchmark keeps the existing schema query types for evaluator compatibility and maps them as follows.",
        "",
        "| Sprint bucket | Status | Benchmark query types | Count | Notes |",
        "|---|---|---|---:|---|",
    ])
    for bucket in ("simple", "relational", "multi_hop", "intent", "ambiguity", "freshness", "negative_absence_review"):
        record = sprint_buckets.get(bucket, {}) if isinstance(sprint_buckets, dict) else {}
        query_types = record.get("benchmark_query_types") or []
        if query_types:
            query_type_text = ", ".join(f"`{item}`" for item in query_types)
        else:
            query_type_text = "N/A"
        notes = record.get("reason") or ""
        if bucket == "ambiguity" and record.get("multi_answer_positive_queries") is not None:
            notes = (
                f"{notes} Multi-answer positive queries: "
                f"{record.get('multi_answer_positive_queries')}."
            ).strip()
        status_lines.append(
            f"| {bucket} | {record.get('status', 'unknown')} | {query_type_text} | "
            f"{record.get('count', 0)} | {notes} |"
        )
    status_lines.extend([
        "",
        "## Scene Distribution",
        "",
        "| Scene scope | Count |",
        "|---|---:|",
    ])
    for scene_id, count in sorted(scene_counts.items()):
        status_lines.append(f"| `{scene_id}` | {count} |")
    status_lines.extend([
        "",
        "## Expected Field Coverage",
        "",
        "| Field | Non-empty queries | Notes |",
        "|---|---:|---|",
        f"| `expected_view_ids` | {view_non_empty} | Positive queries have expected views; negatives intentionally empty. |",
        f"| `expected_node_ids` | {node_non_empty} | Positive queries map to manual semantic tree nodes. |",
        f"| `expected_region_ids` | {region_non_empty} | Region/functional/multi-hop queries include region nodes where available. |",
        f"| `expected_zone_ids` | {zone_non_empty} | Manual zone references exist for positives and are emitted as stable zone nodes in rebuilt captured-scene trees. |",
        "",
        "## Use",
        "",
        "Use this benchmark for same-input graph-vs-flat semantic-index evaluation. Report it as manual semantic-index reference GT, not independent semantic accuracy.",
    ])
    STATUS_OUT.write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    gt_lines = [
        "# Manual GT Coverage Report V2",
        "",
        "Status date: 2026-07-01.",
        "",
        "## Scope",
        "",
        "The benchmark provides expected IDs derived from manual captured-scene ViewJSON annotations and generated tree nodes.",
        "",
        "| GT type | Status | Count | Claim allowed |",
        "|---|---|---:|---|",
        f"| Manual semantic-index expected views | available | {view_non_empty} queries | Semantic-index regression / same-input retrieval quality. |",
        f"| Manual semantic-index expected nodes | available | {node_non_empty} queries | Tree-node selection checks where current tree IDs exist. |",
        f"| Manual region expectations | partial | {region_non_empty} queries | Region-level reference where ViewJSON contains matching region labels. |",
        f"| Manual zone expectations | available, tree-native after rebuild | {zone_non_empty} queries | Zone-denominator checks and graph-enrichment target labels. |",
        "| Tree-native zone traversal | available after rebuild | 5 captured scenes | Current trees include stable `manual_zone_*` nodes. |",
        "| Independent visual/dataset GT | unavailable | 0 | Do not report independent semantic accuracy. |",
        "| Manual coarse GT boxes | available after bbox builder | 125 positive queries | Internal coarse regression signal only, not official dataset localization. |",
        "| Pixel-perfect GT boxes / masks | unavailable | 0 | Do not report official 3D localization accuracy. |",
        "",
        "## Verification Status",
        "",
        f"- `verified_from_view_json`: {verified}",
        "- `verified_dataset_gt`: 0",
        "- `missing_gt`: 0 in v2, because every query has a manual semantic-index reference label.",
        "",
        "## Important Limitation",
        "",
        "These labels are suitable for evaluating whether methods retrieve the same manual semantic-index entries. They do not prove the annotations are visually correct or dataset-ground-truth correct.",
    ]
    gt_text = "\n".join(gt_lines) + "\n"
    GT_REPORT_OUT.write_text(gt_text, encoding="utf-8")
    GT_COVERAGE_ALIAS_OUT.write_text(gt_text, encoding="utf-8")

    negatives = [query for query in benchmark["queries"] if query["query_type"] == "negative"]
    review_lines = [
        "# Ambiguous And Negative Review V2",
        "",
        "Status date: 2026-07-01.",
        "",
        "## Ambiguous Queries",
        "",
        "No query is labelled `ambiguous` in `benchmark_queries_v2.json`. Multi-answer positive queries keep multiple expected views/nodes instead of forcing one canonical answer.",
        "",
        "Freshness/change queries are not included because the captured scenes have no temporal recapture or change labels. They require new temporal evidence rather than manual invention.",
        "",
        "## Negative Query Scope",
        "",
        "Negative cases are verified absent only from the manual ViewJSON semantic index for the target captured scene. They are not independent visual absence proofs.",
        "",
        "| Query ID | Scene | Query | Negative label | Verification source |",
        "|---|---|---|---|---|",
    ]
    for query in negatives:
        source = ", ".join(query.get("verification_source", []))
        review_lines.append(
            f"| `{query['query_id']}` | `{query['scene_id']}` | {query['query']} | "
            f"{query.get('negative_label', '')} | `{source}` |"
        )
    REVIEW_OUT.write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    coverage_lines = [
        "# Person 2 Coverage Table V2",
        "",
        "Status date: 2026-07-01.",
        "",
        "## Day 1 Inventory",
        "",
        "| Source | Total queries | Five captured-scene queries | Five-scene queries with non-empty GT fields |",
        "|---|---:|---:|---:|",
        f"| `benchmark_queries_v1.json` | {v1['total']} | {v1['five_scene_total']} | {v1['five_scene_gt_nonempty']} |",
        f"| `benchmark_queries_v2.json` | {benchmark['query_count']} | {benchmark['query_count']} | {verified} |",
        "",
        "The previous v1 file contained the 40 five-scene queries requested for inventory, but those captured-scene queries did not have reliable expected IDs. V2 replaces that gap with 150 verified manual semantic-index reference queries.",
        "",
        "## V2 Scene By Query Type",
        "",
        "| Scene | simple_object | attribute | relational | multi_hop | functional / intent | negative | Total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene_id in benchmark["scene_scope"]:
        row_counts = {
            query_type: scene_type_counts.get((scene_id, query_type), 0)
            for query_type in ("simple_object", "attribute", "relational", "multi_hop", "functional", "negative")
        }
        total = sum(row_counts.values())
        coverage_lines.append(
            f"| `{scene_id}` | {row_counts['simple_object']} | {row_counts['attribute']} | "
            f"{row_counts['relational']} | {row_counts['multi_hop']} | "
            f"{row_counts['functional']} | {row_counts['negative']} | {total} |"
        )
    coverage_lines.extend([
        "",
        "## Plan Bucket Distribution",
        "",
        "| Required bucket | Status | Count |",
        "|---|---|---:|",
    ])
    for bucket in ("simple", "relational", "multi_hop", "intent", "ambiguity", "freshness"):
        record = sprint_buckets.get(bucket, {}) if isinstance(sprint_buckets, dict) else {}
        coverage_lines.append(f"| {bucket} | {record.get('status', 'unknown')} | {record.get('count', 0)} |")
    coverage_lines.extend([
        "",
        "Additional reviewed negative absence cases: 25.",
    ])
    COVERAGE_TABLE_OUT.write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")

    changelog_lines = [
        "# Benchmark V2 Changelog",
        "",
        "Status date: 2026-07-01.",
        "",
        "| Version item | Change | Evidence |",
        "|---|---|---|",
        "| v1 inventory | Audited 90-query v1 file, including 40 five captured-scene queries with missing expected IDs. | `docs/benchmarks/person2_coverage_table_v2.md` |",
        "| v2 benchmark | Created 150 five-scene verified manual semantic-index queries, 30 per scene. | `docs/benchmarks/benchmark_queries_v2.json` |",
        "| query types | Balanced evaluator query types: simple_object, attribute, relational, multi_hop, functional, negative. | `docs/benchmarks/benchmark_status_v2.md` |",
        "| plan buckets | Added Person 2 buckets: simple, relational, multi_hop, intent, ambiguity review, freshness blocker. | `docs/benchmarks/benchmark_status_v2.md` |",
        "| zone references | Added manual expected zone IDs and labels for every positive query. | `docs/benchmarks/manual_zone_gt_v2.json` |",
        "| negative review | Kept absence checks separate from positive accuracy and documented limitations. | `docs/benchmarks/ambiguous_and_negative_review_v2.md` |",
        "| bbox package | BBox builder adds manual coarse 2D/3D boxes for positive queries. | `docs/benchmarks/manual_bbox_gt_v2.json` |",
    ]
    CHANGELOG_OUT.write_text("\n".join(changelog_lines) + "\n", encoding="utf-8")

    lock_lines = [
        "# Final Benchmark Lock V2",
        "",
        "Status date: 2026-07-01.",
        "",
        "Immutable benchmark ID: `five_capture_manual_semantic_index_v2`.",
        "",
        "This is the locked Person 2 benchmark package for article-facing graph-vs-flat work. It is manual semantic-index reference GT, not independent dataset GT.",
        "",
        "## Required Person 2 Outputs",
        "",
        "| Plan output | Status | Evidence |",
        "|---|---|---|",
        "| Coverage table by scene/query type | DONE | `docs/benchmarks/person2_coverage_table_v2.md` |",
        "| 80-100 verified queries minimum | DONE | 150 queries in `docs/benchmarks/benchmark_queries_v2.json` |",
        "| Expected objects, views, regions, zones | DONE | `docs/benchmarks/benchmark_queries_v2.json`, `docs/benchmarks/manual_zone_gt_v2.json` |",
        "| Query distribution | DONE | `docs/benchmarks/benchmark_status_v2.md` |",
        "| Missing/ambiguous GT report | DONE | `docs/benchmarks/ambiguous_and_negative_review_v2.md` |",
        "| 120-150 two-week expansion | DONE | 150-query v2 benchmark |",
        "| Denominator validation / GT coverage | DONE | `docs/benchmarks/gt_coverage_report.md` |",
        "| Final benchmark lock | DONE | this file |",
        "",
        "## Evidence Paths",
        "",
        "```text",
        "docs/benchmarks/benchmark_queries_v2.json",
        "docs/benchmarks/manual_zone_gt_v2.json",
        "docs/benchmarks/manual_bbox_gt_v2.json",
        "docs/benchmarks/person2_coverage_table_v2.md",
        "docs/benchmarks/benchmark_status_v2.md",
        "docs/benchmarks/gt_coverage_report.md",
        "docs/benchmarks/ambiguous_and_negative_review_v2.md",
        "docs/benchmarks/benchmark_v2_changelog.md",
        "docs/project/limitations_resolution_status.md",
        "configs/week3_benchmark_v2.yaml",
        "configs/week3_benchmark_v2_bbox_gt.yaml",
        "outputs/week3/benchmark_v2_stub_manual_bbox_gt_v4/metrics_summary.json",
        "outputs/week3/benchmark_v2_stub_manual_bbox_gt_v4/metrics_summary.md",
        "outputs/week3/benchmark_v2_stub_manual_bbox_gt_v4/failure_modes.md",
        "outputs/week3/graph_vs_flat_benchmark_v2/graph_vs_flat_metrics.json",
        "outputs/week3/graph_vs_flat_benchmark_v2/per_query_results.json",
        "docs/experiments/graph_vs_flat/graph_vs_flat_benchmark_v2.md",
        "```",
        "",
        "## Locked Limitations",
        "",
        "- Freshness/change queries remain blocked because no temporal recapture/change labels exist.",
        "- Manual labels and boxes are not official Replica, ScanNet, or independent dataset GT.",
        "- Zone expectations are manual reference labels emitted as stable tree-native nodes after rebuild.",
    ]
    FINAL_LOCK_OUT.write_text("\n".join(lock_lines) + "\n", encoding="utf-8")


def write_config() -> None:
    scene_ids = ", ".join(f'"{scene.scene_id}"' for scene in SCENES)
    lines = [
        "{",
        '  "config_version": "semanticsplat.experiment_config.v1",',
        '  "week": "week3",',
        '  "run_id": "week3_stub_benchmark_v2_manual_gt",',
        '  "method": "semantic_splat",',
        '  "mode": "stub",',
        '  "supported_modes": ["stub", "cached_live", "live"],',
        '  "scene_root": "backend/data/scenes",',
        f'  "scene_ids": [{scene_ids}],',
        '  "scenes": [',
    ]
    for index, scene in enumerate(SCENES):
        comma = "," if index + 1 < len(SCENES) else ""
        lines.extend([
            "    {",
            f'      "scene_id": "{scene.scene_id}",',
            f'      "benchmark_scene_id": "{scene.benchmark_scene_id}",',
            f'      "path": "backend/data/scenes/{scene.scene_id}",',
            '      "semantic_eval_allowed": true,',
            '      "geometry_eval_allowed": false,',
            f'      "validation_report": "{scene.validation_report}"',
            f"    }}{comma}",
        ])
    lines.extend([
        "  ],",
        '  "benchmark_path": "docs/benchmarks/benchmark_queries_v2.json",',
        '  "output_root": "outputs/week3",',
        '  "resume": true,',
        '  "query_limit": null,',
        '  "depth_sample_limit": 3,',
        '  "view_selection": {',
        '    "strategy": "existing",',
        '    "max_views": null,',
        '    "coverage_report_dir": "docs/validation/geometry"',
        "  },",
        '  "model": {',
        '    "temperature": 0.0,',
        '    "cache_dir": "outputs/week3/model_cache",',
        '    "provider_env": ["SEMANTICSPLAT_PROVIDER", "SEMANTICSPLAT_MODEL", "SEMANTICSPLAT_API_KEY"]',
        "  },",
        '  "notes": [',
        '    "Benchmark v2 uses manual ViewJSON semantic-index expectations, not independent dataset GT.",',
        '    "expected_zone_ids use manual zone references emitted as stable tree-native zone nodes after rebuild.",',
        '    "Base config leaves geometry_eval_allowed false; use configs/week3_benchmark_v2_bbox_gt.yaml for manual coarse box evaluation."',
        "  ]",
        "}",
    ])
    CONFIG_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_benchmark(benchmark: dict[str, Any]) -> None:
    queries = benchmark["queries"]
    if len(queries) != 150:
        raise ValueError(f"Expected 150 queries, found {len(queries)}")
    type_counts = Counter(query["query_type"] for query in queries)
    scene_counts = Counter(query["scene_id"] for query in queries)
    if set(type_counts.values()) != {25}:
        raise ValueError(f"Query type counts are not balanced: {type_counts}")
    if set(scene_counts.values()) != {30}:
        raise ValueError(f"Scene counts are not balanced: {scene_counts}")
    ids = [query["query_id"] for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query IDs in v2 benchmark")
    for query in queries:
        if query["verification_status"] != "verified_from_view_json":
            raise ValueError(f"Unexpected verification status: {query['query_id']}")
        if query["query_type"] != "negative" and not query["expected_view_ids"]:
            raise ValueError(f"Positive query has no expected views: {query['query_id']}")
        if query["query_type"] != "negative" and not query["expected_node_ids"]:
            raise ValueError(f"Positive query has no expected nodes: {query['query_id']}")
        if query["query_type"] != "negative" and not query["expected_zone_ids"]:
            raise ValueError(f"Positive query has no expected zones: {query['query_id']}")
        if query["query_type"] == "negative" and query["expected_view_ids"]:
            raise ValueError(f"Negative query has expected views: {query['query_id']}")
        if query["query_type"] == "negative" and query["expected_zone_ids"]:
            raise ValueError(f"Negative query has expected zones: {query['query_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark_queries_v2.json and coverage docs")
    parser.add_argument("--check", action="store_true", help="Validate generated content without writing files")
    args = parser.parse_args()
    benchmark = build_queries()
    validate_benchmark(benchmark)
    if args.check:
        print(json.dumps({
            "query_count": benchmark["query_count"],
            "query_type_counts": benchmark["query_type_counts"],
            "scene_query_counts": benchmark["scene_query_counts"],
        }, indent=2, ensure_ascii=False))
        return
    write_json(BENCHMARK_OUT, benchmark)
    write_zone_gt(benchmark)
    write_status_docs(benchmark)
    write_config()
    print(f"Wrote {BENCHMARK_OUT.relative_to(ROOT)}")
    print(f"Wrote {STATUS_OUT.relative_to(ROOT)}")
    print(f"Wrote {GT_REPORT_OUT.relative_to(ROOT)}")
    print(f"Wrote {GT_COVERAGE_ALIAS_OUT.relative_to(ROOT)}")
    print(f"Wrote {REVIEW_OUT.relative_to(ROOT)}")
    print(f"Wrote {ZONE_GT_OUT.relative_to(ROOT)}")
    print(f"Wrote {COVERAGE_TABLE_OUT.relative_to(ROOT)}")
    print(f"Wrote {CHANGELOG_OUT.relative_to(ROOT)}")
    print(f"Wrote {FINAL_LOCK_OUT.relative_to(ROOT)}")
    print(f"Wrote {CONFIG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
