"""The surveillance pipeline: video -> detect -> track -> rules -> events + annotated video.

    python -m sentinel.pipeline --scene warehouse
    python -m sentinel.pipeline --source path/to/video.mp4 --out result.mp4
    python -m sentinel.pipeline --scene hallway --webcam 0 --show     # live camera (local machine)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import annotate
from .config import COCO_CLASSES, DEFAULT_MODEL, RUNS_DIR, VIDEOS_DIR
from .detector import YoloDetector
from .events import EventStore
from .rules import FrameContext, TrackState
from .scenes import build, default_scene, get_scene
from .tracker import ByteTracker

MAX_OUTPUT_WIDTH = 1280


@dataclass
class FrameResult:
    index: int
    time_s: float
    image: np.ndarray  # annotated BGR frame
    counts: dict
    events: list
    timings_ms: dict
    fps: float
    anchors: list = field(default_factory=list)  # (track_id, class, normalized anchor)


@dataclass
class RunReport:
    run_id: int | None
    scene: str
    source: str
    model: str
    summary: dict
    events: list = field(default_factory=list)
    timeseries: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    output_video: str | None = None

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "scene": self.scene, "source": self.source, "model": self.model,
                "summary": self.summary, "events": self.events, "timeseries": self.timeseries,
                "tracks": self.tracks, "output_video": self.output_video}


class VideoWriter:
    """H.264 MP4 writer (browser-playable) using the ffmpeg binary bundled with imageio-ffmpeg."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        import imageio_ffmpeg

        self.size = (size[0] // 2 * 2, size[1] // 2 * 2)  # H.264 needs even dimensions
        self._gen = imageio_ffmpeg.write_frames(
            str(path), self.size, fps=fps, codec="libx264", pix_fmt_out="yuv420p",
            output_params=["-preset", "veryfast", "-crf", "26", "-movflags", "+faststart"],
            macro_block_size=1, ffmpeg_log_level="error",
        )
        self._gen.send(None)

    def write(self, bgr: np.ndarray) -> None:
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            bgr = cv2.resize(bgr, self.size)
        self._gen.send(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    def close(self) -> None:
        self._gen.close()


class SurveillancePipeline:
    def __init__(self, scene: dict | str | None = None, model: str = DEFAULT_MODEL,
                 conf_threshold: float = 0.15, store: EventStore | None = None):
        self.scene_name = scene if isinstance(scene, str) else (scene or {}).get("title", "custom")
        self.scene = get_scene(scene) if isinstance(scene, str) else (scene or default_scene())
        self.model = model
        # Low detector threshold: the tracker decides (ByteTrack uses weak boxes to keep tracks alive)
        self.detector = YoloDetector(model, conf_threshold=conf_threshold, classes=self.scene["classes"])
        self.store = store

    def frames(self, source, max_seconds: float | None = None, run_dir: Path | None = None,
               realtime: bool = False) -> Iterator[FrameResult]:
        """Process a video file or camera index frame by frame (generator)."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not np.isfinite(src_fps) or src_fps <= 0 or src_fps > 240:
            src_fps = 25.0
        is_camera = isinstance(source, int)
        target_fps = float(self.scene.get("analysis_fps", 10))
        step = 1 if is_camera else max(1, round(src_fps / target_fps))
        zones, lines, rules = build(self.scene)
        tracker = ByteTracker(max_age=int(2 * target_fps))
        states: dict[int, TrackState] = defaultdict(TrackState)
        index = -1
        fps_ema = None
        wall_start = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index += 1
                if index % step:
                    continue
                t_s = (time.perf_counter() - wall_start) if is_camera else index / src_fps
                if max_seconds is not None and t_s > max_seconds:
                    break
                t0 = time.perf_counter()
                dets = self.detector.detect(frame)
                t1 = time.perf_counter()
                tracks = tracker.update(dets)
                t2 = time.perf_counter()
                h, w = frame.shape[:2]
                ctx = FrameContext(index, t_s, src_fps / step, tracks, lambda p: (p[0] / w, p[1] / h), states)
                events = [e for rule in rules for e in rule.check(ctx)]
                t3 = time.perf_counter()

                counts = Counter(COCO_CLASSES[t.cls] for t in tracks)
                active_zones = {z.name for z in zones for t in tracks if z.contains(*ctx.norm(t.anchor))}
                elapsed = time.perf_counter() - t0
                inst_fps = 1.0 / max(elapsed, 1e-6)
                fps_ema = inst_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * inst_fps
                image = annotate.draw(frame, tracks, zones, lines, rules, active_zones,
                                      {"fps": fps_ema, "time": f"{t_s:6.1f}s", "counts": dict(counts),
                                       "alerts": ", ".join(e.type for e in events if e.severity != "info")})
                t4 = time.perf_counter()
                if run_dir is not None:
                    for e in events:
                        e.snapshot = _save_snapshot(image, e, run_dir)
                timings = {**{k: round(v, 2) for k, v in dets.timings_ms.items()},
                           "tracking": round((t2 - t1) * 1e3, 2), "rules": round((t3 - t2) * 1e3, 2),
                           "drawing": round((t4 - t3) * 1e3, 2), "total": round((t4 - t0) * 1e3, 2)}
                anchors = [(t.track_id, COCO_CLASSES[t.cls], ctx.norm(t.anchor)) for t in tracks]
                yield FrameResult(index, t_s, image, dict(counts), events, timings, fps_ema, anchors)
                if realtime and not is_camera:
                    # Play at the video's natural speed (simulated live camera)
                    target = t_s - (time.perf_counter() - wall_start)
                    if target > 0:
                        time.sleep(target)
            self._last_rules = rules
            self._last_tracker = tracker
        finally:
            cap.release()

    def run(self, source, output_path: str | Path | None = None, max_seconds: float | None = None,
            on_progress: Callable[[float], None] | None = None, run_dir: Path | None = None) -> RunReport:
        """Process a whole video, write the annotated MP4 and store events. Returns a RunReport."""
        source_label = Path(str(source)).name
        run_id = self.store.new_run(self.scene_name, source_label, self.model) if self.store else None
        if run_dir is None:
            run_dir = Path(output_path).parent if output_path else (
                RUNS_DIR / (f"run_{run_id}" if run_id else f"run_{int(time.time() * 1000)}"))
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_path) if output_path else run_dir / "annotated.mp4"

        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        step = max(1, round(src_fps / float(self.scene.get("analysis_fps", 10))))

        writer = None
        all_events, timeseries, timings = [], [], defaultdict(list)
        paths: dict[int, dict] = {}
        peak = Counter()
        processed = 0
        wall = time.perf_counter()
        try:
            for fr in self.frames(source, max_seconds=max_seconds, run_dir=run_dir):
                if writer is None:
                    h, w = fr.image.shape[:2]
                    s = min(1.0, MAX_OUTPUT_WIDTH / w)
                    writer = VideoWriter(output_path, src_fps / step, (int(w * s), int(h * s)))
                writer.write(fr.image)
                processed += 1
                for k, v in fr.timings_ms.items():
                    timings[k].append(v)
                for cls, n in fr.counts.items():
                    peak[cls] = max(peak[cls], n)
                timeseries.append({"time_s": round(fr.time_s, 2), **fr.counts})
                for tid, cls, (x, y) in fr.anchors:
                    p = paths.setdefault(tid, {"track_id": tid, "class": cls, "start_s": round(fr.time_s, 2),
                                               "points": []})
                    p["end_s"] = round(fr.time_s, 2)
                    p["points"].append([round(x, 4), round(y, 4)])
                all_events.extend(fr.events)
                if self.store and fr.events:
                    self.store.add_events(run_id, fr.events)
                if on_progress and total_frames:
                    on_progress(min(fr.index / total_frames, 1.0))
        finally:
            if writer:
                writer.close()

        rules = getattr(self, "_last_rules", [])
        wall_s = time.perf_counter() - wall
        summary = {
            "frames_processed": processed,
            "video_seconds": round(timeseries[-1]["time_s"], 1) if timeseries else 0,
            "processing_seconds": round(wall_s, 1),
            "throughput_fps": round(processed / wall_s, 1) if wall_s else 0,
            "realtime_factor": round((timeseries[-1]["time_s"] if timeseries else 0) / wall_s, 2) if wall_s else 0,
            "latency_ms": {k: round(float(np.median(v)), 1) for k, v in timings.items()},
            "peak_simultaneous": dict(peak),
            "unique_objects": self._unique_counts(),
            "events_by_type": dict(Counter(e.type for e in all_events)),
            "alerts": sum(1 for e in all_events if e.severity != "info"),
            "line_counts": {r.line.name: dict(r.counts) for r in rules if r.type == "line_crossing"},
            "rules": [r.describe() for r in rules],
        }
        if self.store:
            self.store.finish_run(run_id, summary)
        summary["dwell_seconds"] = {
            str(p["track_id"]): round(p["end_s"] - p["start_s"], 1) for p in paths.values()
        }
        report = RunReport(run_id, self.scene_name, source_label, self.model, summary,
                           [e.to_dict() for e in all_events], timeseries, list(paths.values()), str(output_path))
        (run_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))
        return report

    def _unique_counts(self) -> dict:
        tracker = getattr(self, "_last_tracker", None)
        if tracker is None:
            return {}
        return {COCO_CLASSES[c]: n for c, n in tracker.unique_per_class.items()}


def _save_snapshot(image: np.ndarray, event, run_dir: Path) -> str:
    path = run_dir / f"event_{event.frame:06d}_{event.type}_{event.track_id or 0}.jpg"
    h, w = image.shape[:2]
    if event.box:
        x1, y1, x2, y2 = event.box
        mx, my = (x2 - x1) * 0.6, (y2 - y1) * 0.4
        crop = image[int(max(0, y1 - my)):int(min(h, y2 + my)), int(max(0, x1 - mx)):int(min(w, x2 + mx))]
        if crop.size == 0:
            crop = image
    else:
        crop = image
    scale = min(1.0, 480 / max(crop.shape[:2]))
    crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="SentinelVision surveillance pipeline")
    parser.add_argument("--scene", default=None, help="Scene preset (warehouse, retail, parking, hallway, office)")
    parser.add_argument("--source", default=None, help="Video file path (defaults to the scene's sample video)")
    parser.add_argument("--webcam", type=int, default=None, help="Camera index for a live feed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=None, help="Output MP4 path")
    parser.add_argument("--show", action="store_true", help="Show a live window (local machine only)")
    args = parser.parse_args()

    scene = args.scene
    pipe = SurveillancePipeline(scene, args.model, store=EventStore())
    source = args.webcam if args.webcam is not None else (
        args.source or str(VIDEOS_DIR / f"{get_scene(scene)['video']}.mp4"))

    if args.show or args.webcam is not None:
        for fr in pipe.frames(source, realtime=True):
            for e in fr.events:
                print(f"[{fr.time_s:7.1f}s] {e.severity.upper():8} {e.message}")
            cv2.imshow("SentinelVision (press q to quit)", fr.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        return
    report = pipe.run(source, args.out, on_progress=None)
    for e in report.events:
        print(f"[{e['time_s']:7.1f}s] {e['severity'].upper():8} {e['message']}")
    print(json.dumps({k: v for k, v in report.summary.items() if k != "rules"}, indent=2))
    print(f"Annotated video: {report.output_video}")


if __name__ == "__main__":
    main()
