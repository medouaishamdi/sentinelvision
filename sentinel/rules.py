"""Zones, tripwires and the rules that turn tracks into security events.

All geometry is stored in *normalized* coordinates (0..1), so a scene configuration works
at any video resolution. Rules look at each track's anchor point (bottom-center of the box,
i.e. where the person or vehicle touches the ground).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import COCO_CLASSES

SEVERITY = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Zone:
    name: str
    points: list[tuple[float, float]]  # normalized polygon
    kind: str = "restricted"  # restricted (red) | monitored (blue, analytics only)

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon (normalized coordinates)."""
        inside = False
        pts = self.points
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
            j = i
        return inside


@dataclass
class Line:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    in_label: str = "in"  # crossing from the left side of start->end to the right side
    out_label: str = "out"

    def side(self, x: float, y: float) -> float:
        (x1, y1), (x2, y2) = self.start, self.end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def crossed(self, p_prev, p_now) -> str | None:
        """Direction label if the segment p_prev -> p_now crosses this line segment."""
        s1, s2 = self.side(*p_prev), self.side(*p_now)
        if s1 == 0 or s2 == 0 or s1 * s2 > 0:  # points exactly on the line are skipped by the rule
            return None
        # Also require the movement to intersect the tripwire *segment*, not its infinite extension
        (x1, y1), (x2, y2) = self.start, self.end
        a, b = p_prev, p_now
        d1 = (b[0] - a[0]) * (y1 - a[1]) - (b[1] - a[1]) * (x1 - a[0])
        d2 = (b[0] - a[0]) * (y2 - a[1]) - (b[1] - a[1]) * (x2 - a[0])
        if d1 * d2 > 0:
            return None
        return self.in_label if s1 < 0 < s2 else self.out_label


@dataclass
class Event:
    type: str
    severity: str
    message: str
    frame: int
    time_s: float
    track_id: int | None = None
    object_class: str | None = None
    zone: str | None = None
    box: list | None = None  # pixel xyxy, for snapshots
    snapshot: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrackState:
    """Per-track memory the rules need between frames."""

    last_point: tuple[float, float] | None = None
    zones_inside: dict = field(default_factory=dict)  # zone name -> frames inside (consecutive)
    zone_enter_time: dict = field(default_factory=dict)
    fired: set = field(default_factory=set)
    line_points: dict = field(default_factory=dict)  # line name -> previous anchor
    stationary_since: float | None = None
    last_still_point: tuple[float, float] | None = None


class Rule:
    type = "rule"

    def __init__(self, name: str, classes: list[str] | None = None, severity: str = "warning"):
        self.name = name
        self.classes = {COCO_CLASSES.index(c) for c in classes} if classes else None
        self.severity = severity

    def applies(self, cls: int) -> bool:
        return self.classes is None or cls in self.classes

    def check(self, ctx: "FrameContext") -> list[Event]:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        return {"type": self.type, "name": self.name, "severity": self.severity,
                "classes": sorted(COCO_CLASSES[c] for c in self.classes) if self.classes else "all"}


@dataclass
class FrameContext:
    frame: int
    time_s: float
    fps: float
    tracks: list  # confirmed visible tracks
    norm: callable  # pixel point -> normalized point
    states: dict  # track_id -> TrackState


class IntrusionRule(Rule):
    """Alert when an object enters a restricted zone (must stay `min_frames` to avoid flicker)."""

    type = "intrusion"

    def __init__(self, name: str, zone: Zone, classes=("person",), min_frames: int = 3, severity="critical"):
        super().__init__(name, list(classes), severity)
        self.zone = zone
        self.min_frames = min_frames

    def check(self, ctx):
        events = []
        for t in ctx.tracks:
            if not self.applies(t.cls):
                continue
            st = ctx.states[t.track_id]
            inside = self.zone.contains(*ctx.norm(t.anchor))
            n = st.zones_inside.get(self.zone.name, 0) + 1 if inside else 0
            st.zones_inside[self.zone.name] = n
            key = (self.type, self.name)
            if n >= self.min_frames and key not in st.fired:
                st.fired.add(key)
                events.append(Event(self.type, self.severity,
                                    f"{COCO_CLASSES[t.cls].capitalize()} #{t.track_id} entered '{self.zone.name}'",
                                    ctx.frame, ctx.time_s, t.track_id, COCO_CLASSES[t.cls], self.zone.name,
                                    t.box.tolist()))
            elif not inside and key in st.fired:
                st.fired.discard(key)  # leaving and re-entering raises a new alert
        return events


class LoiteringRule(Rule):
    """Alert when an object stays inside a zone longer than `seconds`."""

    type = "loitering"

    def __init__(self, name: str, zone: Zone, seconds: float = 10.0, classes=("person",), severity="warning"):
        super().__init__(name, list(classes), severity)
        self.zone = zone
        self.seconds = seconds

    def check(self, ctx):
        events = []
        for t in ctx.tracks:
            if not self.applies(t.cls):
                continue
            st = ctx.states[t.track_id]
            key = (self.type, self.name)
            if self.zone.contains(*ctx.norm(t.anchor)):
                st.zone_enter_time.setdefault(self.name, ctx.time_s)
                dwell = ctx.time_s - st.zone_enter_time[self.name]
                if dwell >= self.seconds and key not in st.fired:
                    st.fired.add(key)
                    events.append(Event(self.type, self.severity,
                                        f"{COCO_CLASSES[t.cls].capitalize()} #{t.track_id} has stayed "
                                        f"{dwell:.0f}s in '{self.zone.name}'",
                                        ctx.frame, ctx.time_s, t.track_id, COCO_CLASSES[t.cls],
                                        self.zone.name, t.box.tolist()))
            else:
                st.zone_enter_time.pop(self.name, None)
                st.fired.discard(key)
        return events


class LineCrossingRule(Rule):
    """Count objects crossing a tripwire in each direction."""

    type = "line_crossing"

    def __init__(self, name: str, line: Line, classes=("person",), severity="info"):
        super().__init__(name, list(classes), severity)
        self.line = line
        self.counts = {line.in_label: 0, line.out_label: 0}

    def check(self, ctx):
        events = []
        for t in ctx.tracks:
            if not self.applies(t.cls):
                continue
            st = ctx.states[t.track_id]
            p = ctx.norm(t.anchor)
            if self.line.side(*p) == 0:
                continue  # exactly on the line: wait for the next point to know the direction
            prev = st.line_points.get(self.name)
            st.line_points[self.name] = p
            if prev is None:
                continue
            direction = self.line.crossed(prev, p)
            if direction:
                self.counts[direction] += 1
                events.append(Event(self.type, self.severity,
                                    f"{COCO_CLASSES[t.cls].capitalize()} #{t.track_id} crossed "
                                    f"'{self.line.name}' ({direction})",
                                    ctx.frame, ctx.time_s, t.track_id, COCO_CLASSES[t.cls], self.line.name,
                                    t.box.tolist()))
        return events


class CrowdRule(Rule):
    """Alert when more than `max_count` objects are inside a zone (or the whole frame)."""

    type = "crowd"

    def __init__(self, name: str, max_count: int, zone: Zone | None = None, classes=("person",),
                 min_seconds: float = 2.0, cooldown_s: float = 20.0, severity="warning"):
        super().__init__(name, list(classes), severity)
        self.zone = zone
        self.max_count = max_count
        self.min_seconds = min_seconds
        self.cooldown_s = cooldown_s
        self._over_since: float | None = None
        self._last_alert = -1e9
        self.current = 0

    def check(self, ctx):
        inside = [t for t in ctx.tracks if self.applies(t.cls)
                  and (self.zone is None or self.zone.contains(*ctx.norm(t.anchor)))]
        self.current = len(inside)
        if self.current > self.max_count:
            self._over_since = self._over_since if self._over_since is not None else ctx.time_s
            if (ctx.time_s - self._over_since >= self.min_seconds
                    and ctx.time_s - self._last_alert >= self.cooldown_s):
                self._last_alert = ctx.time_s
                where = f"'{self.zone.name}'" if self.zone else "the scene"
                return [Event(self.type, self.severity,
                              f"{self.current} people in {where} (limit {self.max_count})",
                              ctx.frame, ctx.time_s, None, None, self.zone.name if self.zone else None)]
        else:
            self._over_since = None
        return []


class StationaryVehicleRule(Rule):
    """Alert when a vehicle stops (moves less than `tolerance` of the frame) for `seconds`."""

    type = "stopped_vehicle"

    def __init__(self, name: str, seconds: float = 8.0, zone: Zone | None = None,
                 classes=("car", "truck", "bus", "motorcycle"), tolerance: float = 0.02, severity="warning"):
        super().__init__(name, list(classes), severity)
        self.seconds = seconds
        self.zone = zone
        self.tolerance = tolerance

    def check(self, ctx):
        events = []
        for t in ctx.tracks:
            if not self.applies(t.cls):
                continue
            st = ctx.states[t.track_id]
            p = ctx.norm(t.anchor)
            if self.zone is not None and not self.zone.contains(*p):
                st.stationary_since = None
                continue
            if st.last_still_point is None or np.hypot(p[0] - st.last_still_point[0],
                                                       p[1] - st.last_still_point[1]) > self.tolerance:
                st.last_still_point, st.stationary_since = p, ctx.time_s
                st.fired.discard((self.type, self.name))
                continue
            key = (self.type, self.name)
            if ctx.time_s - st.stationary_since >= self.seconds and key not in st.fired:
                st.fired.add(key)
                events.append(Event(self.type, self.severity,
                                    f"{COCO_CLASSES[t.cls].capitalize()} #{t.track_id} stopped for "
                                    f"{ctx.time_s - st.stationary_since:.0f}s",
                                    ctx.frame, ctx.time_s, t.track_id, COCO_CLASSES[t.cls],
                                    self.zone.name if self.zone else None, t.box.tolist()))
        return events
