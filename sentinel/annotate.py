"""Draw detections, tracks, zones, tripwires and a status overlay on frames."""
from __future__ import annotations

import cv2
import numpy as np

from .config import COCO_CLASSES

# BGR colors (fixed per class so the same class always has the same color)
CLASS_COLORS = {
    "person": (214, 120, 42),     # blue
    "bicycle": (52, 104, 235),    # orange
    "car": (122, 175, 27),        # teal/green
    "motorcycle": (0, 161, 237),  # yellow
    "truck": (164, 123, 232),     # pink
    "bus": (167, 58, 74),         # violet
}
DEFAULT_COLOR = (180, 180, 180)
ZONE_COLORS = {  # (idle, occupied) in BGR
    "restricted": ((59, 59, 208), (0, 0, 255)),      # red
    "monitored": ((214, 120, 42), (240, 160, 60)),   # blue
}
LINE_COLOR = (0, 200, 255)


def _px(points, w, h):
    return np.array([[int(x * w), int(y * h)] for x, y in points], np.int32)


def draw(frame: np.ndarray, tracks, zones, lines, rules, active_zones: set, stats: dict,
         show_trails: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(w / 1100, 0.85)
    thick = max(1, int(2 * scale))
    out = frame.copy()

    # Zones: translucent fill, stronger when someone is inside
    overlay = out.copy()
    def zone_color(z):
        idle, busy = ZONE_COLORS.get(z.kind, ZONE_COLORS["restricted"])
        return busy if z.name in active_zones else idle

    for z in zones:
        cv2.fillPoly(overlay, [_px(z.points, w, h)], zone_color(z))
    cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)
    for z in zones:
        pts = _px(z.points, w, h)
        color = zone_color(z)
        cv2.polylines(out, [pts], True, color, thick + (2 if z.name in active_zones else 0))
        x, y = pts.min(0)
        _label(out, z.name, (int(x) + 6, int(y) + int(24 * scale)), color, scale)

    # Tripwires with live counts
    counts = {r.line.name: r.counts for r in rules if r.type == "line_crossing"}
    for ln in lines:
        p1 = (int(ln.start[0] * w), int(ln.start[1] * h))
        p2 = (int(ln.end[0] * w), int(ln.end[1] * h))
        cv2.line(out, p1, p2, LINE_COLOR, thick + 1, cv2.LINE_AA)
        c = counts.get(ln.name, {})
        text = f"{ln.name}: " + "  ".join(f"{k} {v}" for k, v in c.items())
        _label(out, text, (min(p1[0], p2[0]) + 8, min(p1[1], p2[1]) - 8), (0, 140, 200), scale)

    # Tracks
    for t in tracks:
        name = COCO_CLASSES[t.cls]
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        x1, y1, x2, y2 = map(int, t.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        _label(out, f"{name} #{t.track_id} {t.score:.2f}", (x1, max(y1 - 6, 14)), color, scale)
        if show_trails and len(t.trail) > 1:
            pts = np.array(t.trail, np.int32)
            cv2.polylines(out, [pts], False, color, max(1, thick - 1), cv2.LINE_AA)

    # Status panel
    lines_txt = [f"FPS {stats.get('fps', 0):.1f}  |  {stats.get('time', '')}"]
    lines_txt.append("  ".join(f"{k}: {v}" for k, v in stats.get("counts", {}).items()) or "no objects")
    if stats.get("alerts"):
        lines_txt.append(f"ALERTS: {stats['alerts']}")
    pad = int(8 * scale)
    lh = int(22 * scale)
    box_w = int(max(len(s) for s in lines_txt) * 9.5 * scale) + 2 * pad
    cv2.rectangle(out, (0, 0), (box_w, lh * len(lines_txt) + 2 * pad), (20, 20, 20), -1)
    for i, s in enumerate(lines_txt):
        color = (80, 80, 255) if s.startswith("ALERTS") else (240, 240, 240)
        cv2.putText(out, s, (pad, pad + lh * (i + 1) - int(5 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, color, max(1, thick - 1), cv2.LINE_AA)
    return out


def _label(img, text, org, color, scale):
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale
    (tw, th), base = cv2.getTextSize(text, font, fs, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base - 1), color, -1)
    cv2.putText(img, text, (x, y - 2), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
