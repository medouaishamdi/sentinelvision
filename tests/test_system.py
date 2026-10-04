"""End-to-end tests on a real sample video + API."""
import json

import pytest
from fastapi.testclient import TestClient

from sentinel.data import video_path
from sentinel.events import EventStore
from sentinel.pipeline import SurveillancePipeline


@pytest.fixture(scope="module")
def hallway_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    store = EventStore(out / "events.db")
    pipe = SurveillancePipeline("hallway", store=store)
    report = pipe.run(str(video_path("hallway")), out / "annotated.mp4", run_dir=out)
    return report, store, out


def test_hallway_counts_match_ground_truth(hallway_report):
    """7 people walk through the hallway: 4 leave through the west exit, 3 through the east exit
    (counted manually from the annotated video and the trajectory map)."""
    report, _, _ = hallway_report
    counts = report.summary["line_counts"]
    assert counts["West exit"]["out"] == 4
    assert counts["East exit"]["out"] == 3
    assert report.summary["unique_objects"]["person"] == 7


def test_run_outputs(hallway_report):
    report, store, out = hallway_report
    assert (out / "annotated.mp4").stat().st_size > 100_000
    assert json.loads((out / "report.json").read_text())["summary"]["frames_processed"] > 500
    assert report.summary["realtime_factor"] > 0
    assert report.tracks and all(len(t["points"]) > 0 for t in report.tracks)
    runs = store.runs()
    assert runs[0]["status"] == "done"
    assert len(store.events(runs[0]["id"])) == len(report.events)


@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


def test_api_health_and_scenes(client):
    assert client.get("/health").json()["status"] == "healthy"
    names = {s["name"] for s in client.get("/scenes").json()}
    assert {"warehouse", "retail", "parking", "hallway", "office"} <= names
    assert client.get("/scenes/nope").status_code == 404


def test_api_detect_image(client, sample_image):
    import cv2

    ok, jpg = cv2.imencode(".jpg", sample_image)
    r = client.post("/detect/image", files={"file": ("x.jpg", jpg.tobytes(), "image/jpeg")})
    assert r.status_code == 200 and "detections" in r.json()
    r = client.post("/detect/image", files={"file": ("x.jpg", jpg.tobytes(), "image/jpeg")},
                    data={"annotated": "true"})
    assert r.headers["content-type"] == "image/jpeg"
    bad = client.post("/detect/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 422


def test_api_job_lifecycle(client):
    r = client.post("/jobs", data={"scene": "parking", "max_seconds": "8"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    status = client.get(f"/jobs/{job_id}").json()  # TestClient runs background tasks before returning
    assert status["status"] == "done", status
    assert client.get(f"/jobs/{job_id}/video").status_code == 200
    assert client.get(f"/jobs/{job_id}/report").json()["summary"]["frames_processed"] > 0
    assert client.get("/jobs/999999").status_code == 404
