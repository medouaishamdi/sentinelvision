"""SentinelVision REST API.

Run from the project root:
    uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs  ·  Live stream: http://127.0.0.1:8000/stream/warehouse
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from sentinel import __version__
from sentinel.annotate import CLASS_COLORS, DEFAULT_COLOR
from sentinel.config import COCO_CLASSES, DEFAULT_MODEL, MODELS, REPORTS_DIR, RUNS_DIR
from sentinel.data import video_path
from sentinel.detector import YoloDetector
from sentinel.events import EventStore
from sentinel.pipeline import SurveillancePipeline
from sentinel.scenes import SCENES, default_scene, get_scene

MAX_VIDEO_MB = 200
MAX_IMAGE_MB = 10

app = FastAPI(
    title="SentinelVision API",
    description="Real-time object detection, tracking and smart-surveillance rules (intrusion, loitering, "
                "line crossing, crowding, stopped vehicles) with an event log.",
    version=__version__,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = EventStore()
_detectors: dict[str, YoloDetector] = {}
_jobs: dict[int, dict] = {}
_lock = threading.Lock()


def detector(model: str) -> YoloDetector:
    if model not in MODELS:
        raise HTTPException(422, f"Unknown model '{model}'. Choose one of {list(MODELS)}")
    with _lock:
        if model not in _detectors:
            _detectors[model] = YoloDetector(model)
        return _detectors[model]


# ------------------------------------------------------------------ meta
@app.get("/", tags=["meta"])
def root():
    return {"name": "SentinelVision API", "version": __version__, "docs": "/docs",
            "live_stream_example": "/stream/warehouse"}


@app.get("/health", tags=["meta"])
def health():
    return {"status": "healthy", "models_available": [m for m, p in MODELS.items() if p.exists()]}


@app.get("/model-info", tags=["meta"])
def model_info():
    bench = REPORTS_DIR / "detector_benchmark.json"
    return {"default_model": DEFAULT_MODEL, "runtime": "ONNX Runtime (CPU)",
            "benchmark": json.loads(bench.read_text()) if bench.exists() else None}


# ------------------------------------------------------------- detection
@app.post("/detect/image", tags=["detection"])
async def detect_image(file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL),
                       confidence: float = Form(0.35, ge=0.05, le=0.95),
                       annotated: bool = Form(False)):
    """Detect objects in one image. Returns JSON, or a JPEG with boxes if `annotated=true`."""
    data = await file.read()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image larger than {MAX_IMAGE_MB} MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "Could not decode the image (use JPG or PNG)")
    dets = detector(model).detect(image, conf_threshold=confidence)
    if annotated:
        for b, s, c in zip(dets.boxes, dets.scores, dets.classes):
            color = CLASS_COLORS.get(COCO_CLASSES[c], DEFAULT_COLOR)
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{COCO_CLASSES[c]} {s:.2f}", (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        ok, jpg = cv2.imencode(".jpg", image)
        return Response(jpg.tobytes(), media_type="image/jpeg")
    return {"model": model, "count": len(dets), "detections": dets.to_list(),
            "timings_ms": {k: round(v, 1) for k, v in dets.timings_ms.items()}}


# ---------------------------------------------------------------- scenes
@app.get("/scenes", tags=["scenes"])
def scenes():
    return [{"name": k, "title": v["title"], "description": v["description"]} for k, v in SCENES.items()]


@app.get("/scenes/{name}", tags=["scenes"])
def scene(name: str):
    if name not in SCENES:
        raise HTTPException(404, f"Scene '{name}' not found")
    return get_scene(name)


# ------------------------------------------------------------------ jobs
def _run_job(run_id: int, scene, source: str, model: str, max_seconds, cleanup: Path | None):
    job = _jobs[run_id]
    try:
        pipe = SurveillancePipeline(scene, model, store=None)
        out = RUNS_DIR / f"job_{run_id}"
        out.mkdir(parents=True, exist_ok=True)
        report = pipe.run(source, out / "annotated.mp4", max_seconds=max_seconds,
                          on_progress=lambda p: job.update(progress=round(p, 3)))
        store.add_events(run_id, [_event_obj(e) for e in report.events])
        store.finish_run(run_id, report.summary)
        (out / "report.json").write_text(json.dumps(report.to_dict(), default=str))
        job.update(status="done", progress=1.0, video=str(out / "annotated.mp4"), report=str(out / "report.json"))
    except Exception as exc:  # report failures to the client
        store.finish_run(run_id, {"error": str(exc)}, status="failed")
        job.update(status="failed", error=str(exc))
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


def _event_obj(d: dict):
    from sentinel.rules import Event

    return Event(**{k: d.get(k) for k in Event.__dataclass_fields__})


@app.post("/jobs", tags=["jobs"], status_code=202)
async def create_job(background: BackgroundTasks,
                     scene: str = Form("warehouse", description="Scene preset name, or 'custom'"),
                     scene_config: str | None = Form(None, description="Optional full scene JSON"),
                     model: str = Form(DEFAULT_MODEL),
                     max_seconds: float | None = Form(None, gt=0),
                     file: UploadFile | None = File(None, description="Video file (optional for presets)")):
    """Start analyzing a video in the background. Poll GET /jobs/{id} for progress."""
    if model not in MODELS:
        raise HTTPException(422, f"Unknown model '{model}'")
    cfg = json.loads(scene_config) if scene_config else None
    if cfg is None and scene != "custom" and scene not in SCENES:
        raise HTTPException(404, f"Scene '{scene}' not found")
    tmpdir = None
    if file is not None:
        tmpdir = Path(tempfile.mkdtemp())
        source = tmpdir / Path(file.filename or "video.mp4").name
        size = 0
        with open(source, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_VIDEO_MB * 1024 * 1024:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    raise HTTPException(413, f"Video larger than {MAX_VIDEO_MB} MB")
                f.write(chunk)
        if not cv2.VideoCapture(str(source)).isOpened():
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(422, "Could not open the video file")
    elif scene in SCENES:
        source = video_path(scene)
    else:
        raise HTTPException(422, "Upload a video file for a custom scene")
    scene_def = cfg or (get_scene(scene) if scene in SCENES else default_scene())
    run_id = store.new_run(scene, Path(source).name, model)
    _jobs[run_id] = {"run_id": run_id, "status": "running", "progress": 0.0}
    background.add_task(_run_job, run_id, scene_def, str(source), model, max_seconds, tmpdir)
    return {"job_id": run_id, "status": "running", "poll": f"/jobs/{run_id}"}


@app.get("/jobs/{job_id}", tags=["jobs"])
def job_status(job_id: int):
    run = store.run(job_id)
    if run is None:
        raise HTTPException(404, "Job not found")
    live = _jobs.get(job_id, {})
    return {"job_id": job_id, "status": live.get("status", run["status"]),
            "progress": live.get("progress", 1.0 if run["status"] == "done" else 0.0),
            "error": live.get("error"), "summary": run["summary"], "scene": run["scene"], "model": run["model"]}


@app.get("/jobs/{job_id}/video", tags=["jobs"])
def job_video(job_id: int):
    path = RUNS_DIR / f"job_{job_id}" / "annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Video not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"sentinel_job_{job_id}.mp4")


@app.get("/jobs/{job_id}/report", tags=["jobs"])
def job_report(job_id: int):
    path = RUNS_DIR / f"job_{job_id}" / "report.json"
    if not path.exists():
        raise HTTPException(404, "Report not ready")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- events
@app.get("/events", tags=["events"])
def events(job_id: int | None = None, type: str | None = None,
           severity: str | None = Query(None, pattern="^(info|warning|critical)$"),
           limit: int = Query(100, ge=1, le=1000)):
    """Query the event log across all jobs."""
    return store.events(job_id, type, severity, limit)


@app.get("/runs", tags=["events"])
def runs(limit: int = Query(20, ge=1, le=200)):
    return store.runs(limit)


# ---------------------------------------------------------------- stream
@app.get("/stream/{scene_name}", tags=["live"])
def stream(scene_name: str, model: str = DEFAULT_MODEL):
    """Live MJPEG stream: the scene's video processed in real time, like a camera feed.
    Open it directly in a browser tab."""
    if scene_name not in SCENES:
        raise HTTPException(404, f"Scene '{scene_name}' not found")
    pipe = SurveillancePipeline(scene_name, model)
    source = str(video_path(scene_name))

    def frames():
        for fr in pipe.frames(source, realtime=True):
            ok, jpg = cv2.imencode(".jpg", fr.image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")
