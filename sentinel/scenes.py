"""Scene configurations: which zones, tripwires and rules apply to each camera.

A scene is plain JSON-serializable data, so it can come from a file, the API or the dashboard.
Coordinates are normalized (0..1) relative to the frame.
"""
from __future__ import annotations

import copy

from .rules import (
    CrowdRule, IntrusionRule, Line, LineCrossingRule, LoiteringRule, Rule, StationaryVehicleRule, Zone,
)

SCENES: dict[str, dict] = {
    "warehouse": {
        "title": "Warehouse: hazard zone",
        "video": "worker-zone-detection",
        "description": "Industrial floor. Workers must stay right of the taped line; the left side is a hazard zone.",
        "classes": ["person"],
        "analysis_fps": 10,
        "zones": [{"name": "Hazard zone", "points": [[0.0, 0.6], [0.46, 0.6], [0.23, 1.0], [0.0, 1.0]]}],
        "lines": [{"name": "Exit", "start": [0.9, 0.45], "end": [0.9, 0.85], "in_label": "in", "out_label": "out"}],
        "rules": [
            {"type": "intrusion", "zone": "Hazard zone", "classes": ["person"]},
            {"type": "loitering", "zone": "Hazard zone", "seconds": 4, "classes": ["person"]},
            {"type": "line_crossing", "line": "Exit", "classes": ["person"]},
        ],
    },
    "retail": {
        "title": "Store aisles: dwell time",
        "video": "store-aisle-detection",
        "description": "Retail store. Measure how long shoppers stay in each aisle and detect crowding.",
        "classes": ["person"],
        "analysis_fps": 10,
        "zones": [
            {"name": "Left aisle", "kind": "monitored",
             "points": [[0.0, 0.25], [0.31, 0.25], [0.31, 1.0], [0.0, 1.0]]},
            {"name": "Right aisle", "kind": "monitored",
             "points": [[0.56, 0.25], [1.0, 0.25], [1.0, 1.0], [0.56, 1.0]]},
        ],
        "lines": [],
        "rules": [
            {"type": "loitering", "zone": "Left aisle", "seconds": 15, "classes": ["person"], "severity": "info"},
            {"type": "loitering", "zone": "Right aisle", "seconds": 15, "classes": ["person"], "severity": "info"},
            {"type": "crowd", "max_count": 4, "classes": ["person"]},
        ],
    },
    "parking": {
        "title": "Parking lot: traffic counting",
        "video": "person-bicycle-car-detection",
        "description": "Outdoor lot. Count cars on the lane and bikes/pedestrians on the path; flag stopped cars.",
        "classes": ["person", "bicycle", "car", "motorcycle", "truck", "bus"],
        "analysis_fps": 12,
        "zones": [],
        "lines": [
            {"name": "Car lane", "start": [0.0, 0.6], "end": [0.45, 0.6], "in_label": "in", "out_label": "out"},
            {"name": "Path", "start": [0.85, 0.0], "end": [0.85, 0.5], "in_label": "west", "out_label": "east"},
        ],
        "rules": [
            {"type": "line_crossing", "line": "Car lane", "classes": ["car", "truck", "bus", "motorcycle"]},
            {"type": "line_crossing", "line": "Path", "classes": ["person", "bicycle"]},
            {"type": "stopped_vehicle", "seconds": 8},
        ],
    },
    "hallway": {
        "title": "Hallway: people counting",
        "video": "people-detection",
        "description": "Indoor hallway with two exits. Count people leaving through each side.",
        "classes": ["person"],
        "analysis_fps": 12,
        "zones": [],
        "lines": [
            {"name": "West exit", "start": [0.12, 0.45], "end": [0.12, 0.8], "in_label": "out", "out_label": "in"},
            {"name": "East exit", "start": [0.88, 0.45], "end": [0.88, 0.8], "in_label": "in", "out_label": "out"},
        ],
        "rules": [
            {"type": "line_crossing", "line": "West exit", "classes": ["person"]},
            {"type": "line_crossing", "line": "East exit", "classes": ["person"]},
            {"type": "crowd", "max_count": 2, "classes": ["person"], "min_seconds": 1},
        ],
    },
    "office": {
        "title": "Office: restricted door",
        "video": "one-by-one-person-detection",
        "description": "Small office. The door area on the right is restricted after hours.",
        "classes": ["person"],
        "analysis_fps": 10,
        "zones": [{"name": "Restricted door", "points": [[0.7, 0.72], [1.0, 0.72], [1.0, 1.0], [0.7, 1.0]]}],
        "lines": [],
        "rules": [
            {"type": "intrusion", "zone": "Restricted door", "classes": ["person"]},
            {"type": "loitering", "zone": "Restricted door", "seconds": 6, "classes": ["person"]},
        ],
    },
}


def get_scene(name: str) -> dict:
    return copy.deepcopy(SCENES[name])


def default_scene() -> dict:
    """A generic scene for uploaded videos: count people in frame, alert on crowds."""
    return {
        "title": "Custom video", "video": None, "description": "Uploaded video",
        "classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
        "analysis_fps": 10, "zones": [], "lines": [],
        "rules": [{"type": "crowd", "max_count": 5, "classes": ["person"]},
                  {"type": "stopped_vehicle", "seconds": 10}],
    }


def build(scene: dict) -> tuple[list[Zone], list[Line], list[Rule]]:
    """Turn a scene dict into Zone/Line objects and live Rule instances."""
    zones = {z["name"]: Zone(z["name"], [tuple(p) for p in z["points"]], z.get("kind", "restricted"))
             for z in scene.get("zones", [])}
    lines = {
        ln["name"]: Line(ln["name"], tuple(ln["start"]), tuple(ln["end"]),
                         ln.get("in_label", "in"), ln.get("out_label", "out"))
        for ln in scene.get("lines", [])
    }
    rules: list[Rule] = []
    for i, r in enumerate(scene.get("rules", [])):
        kind = r["type"]
        classes = r.get("classes")
        if kind == "intrusion":
            rules.append(IntrusionRule(f"intrusion:{r['zone']}", zones[r["zone"]], classes or ["person"],
                                       r.get("min_frames", 3), r.get("severity", "critical")))
        elif kind == "loitering":
            rules.append(LoiteringRule(f"loitering:{r['zone']}", zones[r["zone"]], r.get("seconds", 10),
                                       classes or ["person"], r.get("severity", "warning")))
        elif kind == "line_crossing":
            rules.append(LineCrossingRule(f"count:{r['line']}", lines[r["line"]], classes or ["person"]))
        elif kind == "crowd":
            zone = zones[r["zone"]] if r.get("zone") else None
            rules.append(CrowdRule(f"crowd:{r.get('zone', 'scene')}", r["max_count"], zone, classes or ["person"],
                                   r.get("min_seconds", 2.0), r.get("cooldown_s", 20.0),
                                   r.get("severity", "warning")))
        elif kind == "stopped_vehicle":
            zone = zones[r["zone"]] if r.get("zone") else None
            rules.append(StationaryVehicleRule(f"stopped:{i}", r.get("seconds", 8), zone))
        else:
            raise ValueError(f"Unknown rule type: {kind}")
    return list(zones.values()), list(lines.values()), rules
