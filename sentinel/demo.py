"""Pre-compute the demo results shown by the dashboard (one run per scene).

    python -m sentinel.demo
"""
from __future__ import annotations

import json
import shutil

import cv2
import numpy as np

from .config import PROJECT_ROOT
from .data import video_path
from .pipeline import SurveillancePipeline
from .scenes import SCENES

DEMO_DIR = PROJECT_ROOT / "data" / "demo"


def build(scenes=None) -> None:
    for name in scenes or SCENES:
        out = DEMO_DIR / name
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        report = SurveillancePipeline(name).run(str(video_path(name)), out / "annotated.mp4", run_dir=out)
        data = report.to_dict()
        # store paths relative to the project so the demo works on any machine
        data["output_video"] = "annotated.mp4"
        for e in data["events"]:
            if e.get("snapshot"):
                e["snapshot"] = e["snapshot"].split("/")[-1]
        (out / "report.json").write_text(json.dumps(data, default=str))
        save_background(str(video_path(name)), out / "background.jpg")
        s = report.summary
        print(f"{name:<10} {s['video_seconds']:>5}s video in {s['processing_seconds']:>5}s "
              f"({s['realtime_factor']}x real time) | events: {s['events_by_type']}")


def save_background(source: str, path, samples: int = 30) -> None:
    """Empty-scene backdrop for heatmaps: the sampled frame where the detector sees the fewest objects."""
    from .detector import YoloDetector

    det = YoloDetector("yolo11n", conf_threshold=0.3)
    cap = cv2.VideoCapture(source)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best, best_count = None, None
    for i in np.linspace(0, max(n - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        count = len(det.detect(frame))
        if best_count is None or count < best_count:
            best, best_count = frame, count
        if count == 0:
            break
    cap.release()
    if best is not None:
        scale = min(1.0, 1280 / best.shape[1])
        best = cv2.resize(best, (int(best.shape[1] * scale), int(best.shape[0] * scale)))
        cv2.imwrite(str(path), best, [cv2.IMWRITE_JPEG_QUALITY, 85])


def load(name: str) -> dict | None:
    path = DEMO_DIR / name / "report.json"
    return json.loads(path.read_text()) if path.exists() else None


if __name__ == "__main__":
    build()
