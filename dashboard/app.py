"""SentinelVision dashboard.

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinel.config import MODELS, REPORTS_DIR  # noqa: E402
from sentinel.data import video_path  # noqa: E402
from sentinel.demo import DEMO_DIR, load as load_demo, save_background  # noqa: E402
from sentinel.pipeline import SurveillancePipeline  # noqa: E402
from sentinel.scenes import SCENES, get_scene  # noqa: E402

st.set_page_config(page_title="SentinelVision", page_icon="🛡️", layout="wide")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
EVENT_LABEL = {"intrusion": "Intrusion", "loitering": "Loitering / dwell", "line_crossing": "Line crossing",
               "crowd": "Crowd", "stopped_vehicle": "Stopped vehicle"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1300px;}
      div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.18);
          border-radius: 10px; padding: 12px 14px;}
      .event {border-left: 4px solid; padding: 6px 10px; margin-bottom: 6px; border-radius: 4px;
          background: rgba(127,127,127,0.06); font-size: 0.92rem;}
      .event.critical {border-color: #d03b3b;} .event.warning {border-color: #ec835a;} .event.info {border-color: #2a78d6;}
      .muted {opacity: 0.7; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", title_font=dict(size=15),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
    fig.update_xaxes(gridcolor="rgba(127,127,127,0.2)", zerolinecolor="rgba(127,127,127,0.3)")
    fig.update_yaxes(gridcolor="rgba(127,127,127,0.2)", zerolinecolor="rgba(127,127,127,0.3)")
    return fig


def event_html(e: dict) -> str:
    sev = e["severity"]
    return (f'<div class="event {sev}">{SEVERITY_ICON[sev]} <b>{e["time_s"]:.1f}s</b> · '
            f'{EVENT_LABEL.get(e["type"], e["type"])}: {e["message"]}</div>')


def heatmap(background: np.ndarray, tracks: list[dict], draw_paths: bool = True) -> np.ndarray:
    """Where objects spend time: density of track anchor points over the scene."""
    h, w = background.shape[:2]
    grid = np.zeros((h // 4, w // 4), np.float32)
    for t in tracks:
        for x, y in t["points"]:
            gx, gy = min(int(x * grid.shape[1]), grid.shape[1] - 1), min(int(y * grid.shape[0]), grid.shape[0] - 1)
            grid[gy, gx] += 1
    grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=6)
    if grid.max() > 0:
        grid /= grid.max()
    heat = cv2.applyColorMap((grid * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (w, h))
    alpha = cv2.resize(np.clip(np.sqrt(grid) * 1.2, 0, 0.7), (w, h))[..., None]
    out = (background * (1 - alpha) + heat * alpha).astype(np.uint8)
    if draw_paths:
        for t in tracks:
            pts = (np.array(t["points"]) * [w, h]).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(out, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def show_report(report: dict, folder: Path, background: Path | None):
    s = report["summary"]
    k = st.columns(5)
    k[0].metric("Alerts", s["alerts"], help="Warning + critical events")
    k[1].metric("Unique objects", sum(s["unique_objects"].values()),
                help=", ".join(f"{c}: {n}" for c, n in s["unique_objects"].items()))
    k[2].metric("Peak at once", max(s["peak_simultaneous"].values(), default=0))
    k[3].metric("Speed vs. real time", f"{s['realtime_factor']}×", help=f"{s['throughput_fps']} frames/s processed")
    k[4].metric("Latency / frame", f"{s['latency_ms'].get('total', 0):.0f} ms")

    left, right = st.columns([3, 2])
    with left:
        video = folder / Path(report["output_video"]).name
        if video.exists():
            st.video(str(video))
        if s.get("line_counts"):
            st.markdown("**Line counts:** " + " · ".join(
                f"{name}: " + ", ".join(f"{d} {n}" for d, n in c.items()) for name, c in s["line_counts"].items()))
    with right:
        st.markdown("**Event log**")
        events = report["events"]
        if not events:
            st.info("No events: nothing matched the rules.")
        box = st.container(height=380)
        for e in events:
            box.markdown(event_html(e), unsafe_allow_html=True)

    ts = pd.DataFrame(report["timeseries"]).fillna(0)
    if not ts.empty:
        fig = go.Figure()
        classes = [c for c in ts.columns if c != "time_s"]
        for i, c in enumerate(classes):
            fig.add_trace(go.Scatter(x=ts["time_s"], y=ts[c], mode="lines", name=c, line=dict(width=2, color=SERIES[i % 6]),
                                     hovertemplate=f"%{{x:.1f}}s: %{{y}} {c}<extra></extra>"))
        alerts = [e for e in report["events"] if e["severity"] != "info"]
        if alerts:
            fig.add_trace(go.Scatter(
                x=[e["time_s"] for e in alerts], y=[0] * len(alerts), mode="markers", name="alert",
                marker=dict(symbol="triangle-up", size=12, color="#d03b3b"),
                text=[e["message"] for e in alerts], hovertemplate="%{x:.1f}s: %{text}<extra></extra>"))
        fig.update_xaxes(title="Video time (s)")
        fig.update_yaxes(title="Objects in view", rangemode="tozero")
        st.plotly_chart(style_fig(fig.update_layout(title="Activity over time"), 280), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if background is not None and background.exists() and report.get("tracks"):
            st.markdown("**Heatmap: where people spend time** (white lines = individual paths)")
            st.image(heatmap(cv2.imread(str(background)), report["tracks"]), use_container_width=True)
    with c2:
        snaps = [e for e in report["events"] if e.get("snapshot") and e["severity"] != "info"][:6] or \
                [e for e in report["events"] if e.get("snapshot")][:6]
        if snaps:
            st.markdown("**Event snapshots**")
            cols = st.columns(3)
            for i, e in enumerate(snaps):
                p = folder / Path(e["snapshot"]).name
                if p.exists():
                    cols[i % 3].image(str(p), caption=f"{e['time_s']:.0f}s · {EVENT_LABEL.get(e['type'])}",
                                      use_container_width=True)
        if s.get("dwell_seconds"):
            dwell = pd.Series(s["dwell_seconds"], dtype=float)
            st.caption(f"Time on camera per tracked object: median {dwell.median():.0f}s, "
                       f"longest {dwell.max():.0f}s")


with st.sidebar:
    st.title("🛡️ SentinelVision")
    st.caption("Real-time object detection & smart surveillance")
    st.divider()
    st.caption("YOLO11 (ONNX Runtime, CPU) · ByteTrack-style tracker · rule engine")
    st.caption("Videos: Intel IoT DevKit sample videos (CC BY 4.0)")

tab_live, tab_analyze, tab_upload, tab_events, tab_perf, tab_how = st.tabs(
    ["Live monitor", "Scene analysis", "Analyze your video", "Event log", "Performance", "How it works"]
)
SCENE_TITLES = {v["title"]: k for k, v in SCENES.items()}

# ============================================================ LIVE
with tab_live:
    st.header("Live monitor")
    st.caption("Plays a camera feed at real-time speed through the full pipeline: detection, tracking, rules.")
    c1, c2, c3 = st.columns([2, 1, 1])
    live_scene = SCENE_TITLES[c1.selectbox("Camera", list(SCENE_TITLES), key="live_scene")]
    live_model = c2.selectbox("Model", list(MODELS), key="live_model")
    duration = c3.slider("Seconds to watch", 10, 120, 30, 5)
    st.caption(get_scene(live_scene)["description"])
    if st.button("▶ Start live feed", type="primary"):
        frame_box, side = st.columns([3, 1])
        img_slot = frame_box.empty()
        with side:
            fps_slot, count_slot = st.empty(), st.empty()
            st.markdown("**Live events**")
            feed = st.container(height=360)
        pipe = SurveillancePipeline(live_scene, live_model)
        n_events = 0
        for fr in pipe.frames(str(video_path(live_scene)), max_seconds=duration, realtime=True):
            img_slot.image(cv2.cvtColor(fr.image, cv2.COLOR_BGR2RGB), use_container_width=True)
            fps_slot.metric("Processing FPS", f"{fr.fps:.1f}")
            count_slot.metric("Objects in view", sum(fr.counts.values()))
            for e in fr.events:
                n_events += 1
                feed.markdown(event_html(e.to_dict()), unsafe_allow_html=True)
        st.success(f"Feed ended: {n_events} events.")

# ========================================================= ANALYZE
with tab_analyze:
    st.header("Scene analysis")
    title = st.selectbox("Scene", list(SCENE_TITLES), key="demo_scene")
    name = SCENE_TITLES[title]
    scene = get_scene(name)
    st.caption(scene["description"])
    with st.expander("Rules for this scene"):
        st.json({"zones": scene["zones"], "lines": scene["lines"], "rules": scene["rules"]})
    report = load_demo(name)
    if report is None:
        st.warning("No pre-computed result for this scene. Run `python -m sentinel.demo`.")
    else:
        show_report(report, DEMO_DIR / name, DEMO_DIR / name / "background.jpg")

# ========================================================== UPLOAD
with tab_upload:
    st.header("Analyze your own video")
    st.caption("Upload a short clip (max 200 MB, analysis limited to the first 60 s). "
               "Set a restricted zone and a counting line in percent of the frame.")
    up = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Restricted zone** (rectangle, % of frame)")
        use_zone = st.toggle("Enable zone", value=True)
        zx = st.slider("Zone left → right", 0, 100, (60, 95))
        zy = st.slider("Zone top → bottom", 0, 100, (50, 100))
        loiter = st.slider("Loitering alert after (s)", 3, 60, 10)
    with c2:
        st.markdown("**Counting line** (horizontal)")
        use_line = st.toggle("Enable counting line", value=True)
        ly = st.slider("Line height (% from top)", 5, 95, 60)
        crowd = st.slider("Crowd alert above (people)", 1, 30, 5)
        classes = st.multiselect("Objects to track", ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
                                 default=["person", "car"])
        model = st.selectbox("Model", list(MODELS), key="up_model")
    custom = {
        "title": "custom", "description": "Uploaded video", "classes": classes or ["person"], "analysis_fps": 10,
        "zones": [], "lines": [], "rules": [{"type": "crowd", "max_count": crowd, "classes": ["person"]}],
    }
    if use_zone:
        pts = [[zx[0] / 100, zy[0] / 100], [zx[1] / 100, zy[0] / 100], [zx[1] / 100, zy[1] / 100], [zx[0] / 100, zy[1] / 100]]
        custom["zones"].append({"name": "Restricted zone", "points": pts})
        custom["rules"] += [{"type": "intrusion", "zone": "Restricted zone", "classes": ["person"]},
                            {"type": "loitering", "zone": "Restricted zone", "seconds": loiter, "classes": ["person"]}]
    if use_line:
        custom["lines"].append({"name": "Line", "start": [0.0, ly / 100], "end": [1.0, ly / 100],
                                "in_label": "down", "out_label": "up"})
        custom["rules"].append({"type": "line_crossing", "line": "Line", "classes": classes or ["person"]})

    if up is not None:
        tmp = Path(tempfile.mkdtemp())
        src = tmp / up.name
        src.write_bytes(up.getvalue())
        cap = cv2.VideoCapture(str(src))
        ok, first = cap.read()
        cap.release()
        if not ok:
            st.error("Could not read this video.")
        else:
            from sentinel.annotate import draw
            from sentinel.scenes import build

            zones, lines, rules = build(custom)
            st.image(cv2.cvtColor(draw(first, [], zones, lines, rules, set(), {"fps": 0, "counts": {}}),
                                  cv2.COLOR_BGR2RGB), caption="Preview of your zone and line", width=640)
            if st.button("Run analysis", type="primary"):
                bar = st.progress(0.0, text="Analyzing...")
                out = tmp / "result"
                t0 = time.time()
                rep = SurveillancePipeline(custom, model).run(
                    str(src), out / "annotated.mp4", max_seconds=60, run_dir=out,
                    on_progress=lambda p: bar.progress(min(p * 1.0, 1.0), text=f"Analyzing... {p:.0%}"))
                bar.progress(1.0, text=f"Done in {time.time() - t0:.0f}s")
                save_background(str(src), out / "background.jpg")
                st.session_state["last_upload"] = (rep.to_dict(), str(out))
    if "last_upload" in st.session_state:
        rep, folder = st.session_state["last_upload"]
        show_report(rep, Path(folder), Path(folder) / "background.jpg")

# ========================================================== EVENTS
with tab_events:
    st.header("Event log")
    st.caption("All events from the scene analyses. The API also stores every job in SQLite (GET /events).")
    rows = []
    for n in SCENES:
        r = load_demo(n)
        for e in (r or {}).get("events", []):
            rows.append({"scene": SCENES[n]["title"], **e})
    if rows:
        df = pd.DataFrame(rows)
        f1, f2, f3 = st.columns(3)
        sev = f1.multiselect("Severity", ["critical", "warning", "info"], default=["critical", "warning"])
        typ = f2.multiselect("Type", sorted(df["type"].unique()))
        sc = f3.multiselect("Scene", sorted(df["scene"].unique()))
        view = df[df["severity"].isin(sev)] if sev else df
        if typ:
            view = view[view["type"].isin(typ)]
        if sc:
            view = view[view["scene"].isin(sc)]
        m = st.columns(4)
        m[0].metric("Events", len(view))
        m[1].metric("Critical", int((view["severity"] == "critical").sum()))
        m[2].metric("Warnings", int((view["severity"] == "warning").sum()))
        m[3].metric("Info", int((view["severity"] == "info").sum()))
        view = view.assign(severity=view["severity"].map(lambda s: f"{SEVERITY_ICON[s]} {s}"),
                           type=view["type"].map(lambda t: EVENT_LABEL.get(t, t)))
        st.dataframe(view[["scene", "time_s", "severity", "type", "message", "track_id", "zone"]],
                     hide_index=True, use_container_width=True,
                     column_config={"time_s": st.column_config.NumberColumn("Time (s)", format="%.1f"),
                                    "track_id": "Track", "message": "Message", "scene": "Scene",
                                    "severity": "Severity", "type": "Type", "zone": "Zone / line"})
        by_type = df.groupby(["type", "severity"]).size().reset_index(name="n")
        fig = go.Figure(go.Bar(x=[EVENT_LABEL.get(t, t) for t in by_type["type"]], y=by_type["n"],
                               marker_color=[{"critical": "#d03b3b", "warning": "#ec835a", "info": "#2a78d6"}[s]
                                             for s in by_type["severity"]],
                               text=by_type["n"], textposition="outside",
                               hovertemplate="%{x}: %{y}<extra></extra>"))
        st.plotly_chart(style_fig(fig.update_layout(title="Events by type (color = severity)"), 280),
                        use_container_width=True)

# ===================================================== PERFORMANCE
with tab_perf:
    st.header("Performance")
    bench_path = REPORTS_DIR / "detector_benchmark.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        rows = []
        for m, b in bench.items():
            rows.append({"Model": m, "mAP50 (COCO128)": b["coco128"]["map50"], "mAP50-95": b["coco128"]["map50_95"],
                         "Person AP50": b["coco128"]["per_class_ap50"].get("person"),
                         **{f"FPS @ {res}": v["fps"] for res, v in b["speed_cpu"].items()}})
        st.subheader("Detector: accuracy vs. speed (CPU)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(c, format="%.3f")
                                    for c in ["mAP50 (COCO128)", "mAP50-95", "Person AP50"]})
        fig = go.Figure()
        for i, (m, b) in enumerate(bench.items()):
            fig.add_trace(go.Bar(name=m, x=list(b["speed_cpu"]), y=[v["fps"] for v in b["speed_cpu"].values()],
                                 marker_color=SERIES[i], text=[f"{v['fps']:.0f}" for v in b["speed_cpu"].values()],
                                 textposition="outside", hovertemplate="%{x}: %{y:.1f} FPS<extra></extra>"))
        fig.add_hline(y=10, line_dash="dash", line_color="rgba(127,127,127,0.6)",
                      annotation_text="10 FPS analysis rate", annotation_position="top left")
        fig.update_yaxes(title="Frames per second")
        st.plotly_chart(style_fig(fig.update_layout(title="Detector throughput by input resolution",
                                                    barmode="group"), 300), use_container_width=True)
    st.subheader("Full pipeline per scene (yolo11n)")
    rows = []
    for n in SCENES:
        r = load_demo(n)
        if r:
            s = r["summary"]
            rows.append({"Scene": SCENES[n]["title"], "Video (s)": s["video_seconds"],
                         "Processing (s)": s["processing_seconds"], "× real time": s["realtime_factor"],
                         **{k: v for k, v in s["latency_ms"].items() if k != "total"}, "total": s["latency_ms"]["total"]})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        stages = ["preprocess", "inference", "postprocess", "tracking", "rules", "drawing"]
        fig = go.Figure()
        for i, stage in enumerate(stages):
            fig.add_trace(go.Bar(name=stage, y=df["Scene"], x=df[stage], orientation="h",
                                 marker_color=SERIES[i % 6], hovertemplate=f"{stage}: %{{x:.1f}} ms<extra></extra>"))
        fig.update_xaxes(title="Median latency per frame (ms)")
        st.plotly_chart(style_fig(fig.update_layout(title="Where the time goes", barmode="stack"), 300),
                        use_container_width=True)
        st.caption("Inference dominates; tracking and rules cost under 1 ms per frame.")

# =============================================================== HOW
with tab_how:
    st.header("How it works")
    st.markdown(
        """
**Pipeline** (every frame at the scene's analysis rate, e.g. 10 FPS):

1. **Detection**: YOLO11 exported to **ONNX** and run with ONNX Runtime on CPU (no PyTorch needed).
   Letterbox resizing, output decoding and class-aware non-maximum suppression are implemented by hand,
   and validated by reproducing the official COCO128 mAP.
2. **Tracking**: ByteTrack-style multi-object tracker: a Kalman filter predicts each object's motion,
   the Hungarian algorithm matches detections to tracks in two passes (confident, then weak detections
   to keep briefly occluded people), and new tracks must be confirmed over 3 frames.
3. **Rules**, applied to each track's ground point (bottom-center of the box):
   - **Intrusion**: entering a restricted zone
   - **Loitering / dwell**: staying in a zone longer than N seconds
   - **Line crossing**: directional counting on tripwires
   - **Crowd**: more than N people for a few seconds
   - **Stopped vehicle**: a car not moving for N seconds
4. **Events** are logged to SQLite with a snapshot image, and the annotated video is encoded as H.264.

Zones and lines use **normalized coordinates** (0-1), so the same configuration works at any resolution.
        """
    )
    st.info("Privacy by design: no face recognition, no identity. Track IDs are anonymous numbers that reset "
            "when a person leaves the frame. Snapshots are only taken for events.")
