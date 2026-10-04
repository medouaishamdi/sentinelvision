"""Unit tests: detector math, tracker, geometry and rules."""
import numpy as np
import pytest

from sentinel.detector import YoloDetector, letterbox, nms
from sentinel.evaluation import average_precision, box_iou, match_predictions
from sentinel.rules import (
    CrowdRule, FrameContext, IntrusionRule, Line, LineCrossingRule, LoiteringRule, StationaryVehicleRule,
    TrackState, Zone,
)
from sentinel.scenes import SCENES, build, get_scene
from sentinel.tracker import ByteTracker

from .conftest import make_dets


# ---------------------------------------------------------------- detector
def test_letterbox_keeps_aspect_ratio():
    img = np.zeros((432, 768, 3), np.uint8)
    out, scale, (px, py) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert scale == pytest.approx(640 / 768)
    assert px == 0 and py == pytest.approx((640 - 360) / 2, abs=1)


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert list(keep) == [0, 2]


def test_detector_on_real_image(sample_image):
    det = YoloDetector("yolo11n", conf_threshold=0.25)
    d = det.detect(sample_image)
    assert d.boxes.shape[1] == 4 and len(d) == len(d.scores) == len(d.classes)
    h, w = sample_image.shape[:2]
    assert (d.boxes[:, [0, 2]] <= w).all() and (d.boxes[:, [1, 3]] <= h).all()
    assert {"preprocess", "inference", "postprocess"} <= set(d.timings_ms)


def test_detector_class_filter(sample_image):
    d = YoloDetector("yolo11n", conf_threshold=0.1, classes=["person"]).detect(sample_image)
    assert set(d.classes.tolist()) <= {0}


# -------------------------------------------------------------- evaluation
def test_iou_and_matching():
    a = np.array([[0, 0, 10, 10]], float)
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)
    assert box_iou(a, np.array([[5, 0, 15, 10]], float))[0, 0] == pytest.approx(1 / 3)
    correct = match_predictions(a, np.array([0]), a, np.array([0]))
    assert correct.all()
    assert not match_predictions(a, np.array([1]), a, np.array([0])).any()  # wrong class


def test_average_precision_perfect():
    # 0.995, not 1.0: the 101-point COCO-style integration (same as the Ultralytics implementation)
    # loses the final step at recall = 1. Kept identical so our mAP is comparable to the official one.
    assert average_precision(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.995, abs=1e-3)
    assert average_precision(np.array([0.5, 1.0]), np.array([1.0, 0.5])) < 0.9


# ----------------------------------------------------------------- tracker
def test_tracker_keeps_id_while_object_moves():
    tr = ByteTracker(min_hits=2)
    ids = set()
    for i in range(10):
        tracks = tr.update(make_dets([[100 + 5 * i, 100, 150 + 5 * i, 200]]))
        ids |= {t.track_id for t in tracks}
    assert ids == {1}


def test_tracker_survives_short_occlusion_and_confirms():
    tr = ByteTracker(min_hits=3, max_age=5)
    box = [100, 100, 150, 200]
    for _ in range(4):
        tr.update(make_dets([box]))
    for _ in range(3):  # 3 frames without detection
        tr.update(make_dets([]))
    tracks = tr.update(make_dets([box]))
    assert [t.track_id for t in tracks] == [1]


def test_tracker_recovers_with_low_confidence_detection():
    tr = ByteTracker(min_hits=1)
    tr.update(make_dets([[100, 100, 150, 200]], [0.9]))
    tracks = tr.update(make_dets([[102, 100, 152, 200]], [0.2]))  # weak box: second association pass
    assert [t.track_id for t in tracks] == [1]


def test_tentative_tracks_are_dropped():
    tr = ByteTracker(min_hits=3)
    assert tr.update(make_dets([[0, 0, 10, 10]])) == []  # one-frame false positive
    tr.update(make_dets([]))
    assert tr.tracks == [] and tr.total_confirmed == 0


def test_tracker_never_matches_across_classes():
    tr = ByteTracker(min_hits=1)
    a = tr.update(make_dets([[0, 0, 50, 50]], classes=[0]))
    b = tr.update(make_dets([[0, 0, 50, 50]], classes=[2]))
    assert a[0].track_id != b[0].track_id


# ---------------------------------------------------------------- geometry
def test_zone_contains():
    z = Zone("z", [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)])
    assert z.contains(0.25, 0.25)
    assert not z.contains(0.75, 0.25)


def test_line_crossing_direction_and_segment():
    ln = Line("door", (0.5, 0.0), (0.5, 1.0), "in", "out")
    assert ln.crossed((0.6, 0.5), (0.4, 0.5)) is not None
    assert ln.crossed((0.6, 0.5), (0.4, 0.5)) != ln.crossed((0.4, 0.5), (0.6, 0.5))
    assert ln.crossed((0.4, 0.5), (0.45, 0.5)) is None  # same side
    short = Line("short", (0.5, 0.0), (0.5, 0.2))
    assert short.crossed((0.4, 0.8), (0.6, 0.8)) is None  # passes beyond the segment's end


# ------------------------------------------------------------------- rules
class FakeTrack:
    def __init__(self, tid, anchor, cls=0):
        self.track_id, self.cls = tid, cls
        self._a = anchor
        self.box = np.array([anchor[0] - 5, anchor[1] - 20, anchor[0] + 5, anchor[1]], float)

    @property
    def anchor(self):
        return self._a


def ctx(frame, t, tracks, states):
    return FrameContext(frame, t, 10, tracks, lambda p: (p[0] / 100, p[1] / 100), states)


def run_rule(rule, positions_per_frame, fps=10):
    states, events = {}, []
    for f, tracks in enumerate(positions_per_frame):
        for tr in tracks:
            states.setdefault(tr.track_id, TrackState())
        events += rule.check(ctx(f, f / fps, tracks, states))
    return events


ZONE = Zone("restricted", [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])


def test_intrusion_requires_persistence_and_fires_once():
    rule = IntrusionRule("i", ZONE, min_frames=3)
    frames = [[FakeTrack(1, (60 + i, 60))] for i in range(10)]
    events = run_rule(rule, frames)
    assert len(events) == 1 and events[0].frame == 2 and events[0].severity == "critical"


def test_intrusion_ignores_flicker_and_other_classes():
    rule = IntrusionRule("i", ZONE, min_frames=3)
    flicker = [[FakeTrack(1, (60, 60))], [FakeTrack(1, (10, 10))]] * 5
    assert run_rule(rule, flicker) == []
    car = [[FakeTrack(2, (60, 60), cls=2)] for _ in range(5)]
    assert run_rule(IntrusionRule("i", ZONE, min_frames=1), car) == []


def test_loitering_after_threshold():
    rule = LoiteringRule("l", ZONE, seconds=2.0)
    events = run_rule(rule, [[FakeTrack(1, (70, 70))] for _ in range(40)])
    assert len(events) == 1 and events[0].time_s == pytest.approx(2.0)


def test_line_crossing_counts_both_directions():
    rule = LineCrossingRule("c", Line("mid", (0.5, 0.0), (0.5, 1.0), "in", "out"))
    right = [[FakeTrack(1, (40 + 5 * i, 50))] for i in range(5)]
    left = [[FakeTrack(2, (60 - 5 * i, 50))] for i in range(5)]
    run_rule(rule, right + left)
    assert sum(rule.counts.values()) == 2 and set(rule.counts.values()) == {1}


def test_crowd_rule_with_duration_and_cooldown():
    rule = CrowdRule("crowd", max_count=2, min_seconds=1.0, cooldown_s=100)
    crowd = [[FakeTrack(i, (10 * i, 50)) for i in range(1, 4)] for _ in range(50)]
    events = run_rule(rule, crowd)
    assert len(events) == 1 and events[0].time_s == pytest.approx(1.0)


def test_stopped_vehicle():
    rule = StationaryVehicleRule("s", seconds=2.0)
    parked = [[FakeTrack(1, (50, 50), cls=2)] for _ in range(30)]
    assert len(run_rule(rule, parked)) == 1
    moving = [[FakeTrack(2, (5 * i, 50), cls=2)] for i in range(30)]
    assert run_rule(StationaryVehicleRule("s", seconds=2.0), moving) == []


@pytest.mark.parametrize("name", list(SCENES))
def test_all_scenes_build(name):
    zones, lines, rules = build(get_scene(name))
    assert rules
