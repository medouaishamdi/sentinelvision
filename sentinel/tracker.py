"""Multi-object tracker in the style of ByteTrack.

- Each track has a constant-velocity Kalman filter on (cx, cy, aspect, height).
- Association is done in two passes with the Hungarian algorithm on IoU:
  1. high-confidence detections vs. all tracks
  2. low-confidence detections vs. tracks still unmatched (recovers briefly occluded objects)
- New tracks are 'tentative' until seen `min_hits` times (filters one-frame false positives),
  and are deleted after `max_age` missed frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detections


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def xyah_to_xyxy(s):
    cx, cy, a, h = s[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanBox:
    """Constant-velocity Kalman filter, state = [cx, cy, a, h, vcx, vcy, va, vh]."""

    std_pos, std_vel = 1 / 20, 1 / 160

    def __init__(self, box):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        m = xyxy_to_xyah(box)
        self.x = np.r_[m, np.zeros(4)]
        h = m[3]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        self.P = np.diag(np.square(std))

    def predict(self):
        h = self.x[3]
        q = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
             self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(np.square(q))

    def update(self, box):
        z = xyxy_to_xyah(box)
        h = self.x[3]
        R = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]))
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self):
        return xyah_to_xyxy(self.x)


@dataclass
class Track:
    track_id: int
    cls: int
    kf: KalmanBox
    score: float
    first_frame: int
    last_frame: int
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    trail: list = field(default_factory=list)  # recent anchor points (bottom-center)

    @property
    def box(self) -> np.ndarray:
        return self.kf.box

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-center of the box: where the object touches the ground (used by zone rules)."""
        b = self.box
        return float((b[0] + b[2]) / 2), float(b[3])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(np.clip(x[:, 2:] - x[:, :2], 0, None), axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class ByteTracker:
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.15, match_iou: float = 0.25,
                 min_hits: int = 3, max_age: int = 30, trail_length: int = 40):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.min_hits = min_hits
        self.max_age = max_age
        self.trail_length = trail_length
        self.tracks: list[Track] = []
        self.frame = 0
        self._next_id = 1
        self.unique_per_class: dict[int, int] = {}

    def _associate(self, tracks: list[Track], boxes: np.ndarray, classes: np.ndarray, threshold: float):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        iou = iou_matrix(np.array([t.box for t in tracks]), boxes)
        iou *= np.array([[t.cls == c for c in classes] for t in tracks])  # never match across classes
        rows, cols = linear_sum_assignment(-iou)
        matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= threshold]
        mt, md = {r for r, _ in matches}, {c for _, c in matches}
        return (matches, [i for i in range(len(tracks)) if i not in mt],
                [j for j in range(len(boxes)) if j not in md])

    def update(self, dets: Detections) -> list[Track]:
        """Advance one frame. Returns confirmed tracks visible in this frame."""
        self.frame += 1
        for t in self.tracks:
            t.kf.predict()

        high = dets.scores >= self.high_thresh
        low = (dets.scores >= self.low_thresh) & ~high
        hb, hc, hs = dets.boxes[high], dets.classes[high], dets.scores[high]
        lb, lc, ls = dets.boxes[low], dets.classes[low], dets.scores[low]

        # Pass 1: confident detections
        matches, unmatched_t, unmatched_d = self._associate(self.tracks, hb, hc, self.match_iou)
        for ti, di in matches:
            self._update_track(self.tracks[ti], hb[di], hs[di])
        # Pass 2: weak detections vs. remaining tracks (occlusion recovery)
        remaining = [self.tracks[i] for i in unmatched_t]
        matches2, still_unmatched, _ = self._associate(remaining, lb, lc, 0.5)
        for ti, di in matches2:
            self._update_track(remaining[ti], lb[di], ls[di])
        for i in still_unmatched:
            remaining[i].missed += 1
        # New tracks from unmatched confident detections
        for di in unmatched_d:
            # id 0 = tentative; a public id is assigned only once the track is confirmed
            track = Track(0, int(hc[di]), KalmanBox(hb[di]), float(hs[di]), self.frame, self.frame)
            self.tracks.append(track)
            if self.min_hits <= 1:
                self._confirm(track)
        # Remove dead tracks (tentative tracks die immediately when missed)
        self.tracks = [t for t in self.tracks
                       if t.missed <= (self.max_age if t.confirmed else 0)]
        for t in self.tracks:
            if t.missed == 0:
                t.trail.append(t.anchor)
                del t.trail[:-self.trail_length]
        return [t for t in self.tracks if t.confirmed and t.missed == 0]

    def _update_track(self, track: Track, box, score):
        track.kf.update(box)
        track.score = float(score)
        track.hits += 1
        track.missed = 0
        track.last_frame = self.frame
        if not track.confirmed and track.hits >= self.min_hits:
            self._confirm(track)

    def _confirm(self, track: Track):
        track.confirmed = True
        track.track_id = self._next_id
        self._next_id += 1
        self.unique_per_class[track.cls] = self.unique_per_class.get(track.cls, 0) + 1

    @property
    def total_confirmed(self) -> int:
        return self._next_id - 1
