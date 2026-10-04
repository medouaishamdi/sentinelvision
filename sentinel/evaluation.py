"""Detection accuracy (mAP) and speed benchmarks.

    python -m sentinel.evaluation           # downloads COCO128 (7 MB) on first run
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .config import COCO_CLASSES, MODELS, PROJECT_ROOT, REPORTS_DIR
from .detector import YoloDetector

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO128_DIR = PROJECT_ROOT / "data" / "coco128"
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def download_coco128() -> Path:
    if not (COCO128_DIR / "images").exists():
        print("Downloading COCO128 (7 MB)...")
        data = urllib.request.urlopen(COCO128_URL, timeout=120).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(COCO128_DIR.parent)
    return COCO128_DIR


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_labels(label_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros(0, int)
    rows = np.loadtxt(label_path, ndmin=2)
    cls = rows[:, 0].astype(int)
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), cls


def match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls) -> np.ndarray:
    """(n_pred, 10) bool: is each prediction a true positive at each IoU threshold."""
    correct = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct
    iou = box_iou(gt_boxes, pred_boxes) * (gt_cls[:, None] == pred_cls[None, :])
    for t, thr in enumerate(IOU_THRESHOLDS):
        gi, pi = np.nonzero(iou >= thr)
        if not len(gi):
            continue
        pairs = np.stack([gi, pi, iou[gi, pi]], 1)
        pairs = pairs[pairs[:, 2].argsort()[::-1]]  # best IoU first, one-to-one matching
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
        correct[pairs[:, 1].astype(int), t] = True
    return correct


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def compute_map(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, gt_cls: np.ndarray) -> dict:
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    per_class = {}
    for c in np.unique(gt_cls):
        n_gt = int((gt_cls == c).sum())
        mask = pred_cls == c
        if not mask.any():
            per_class[int(c)] = np.zeros(len(IOU_THRESHOLDS))
            continue
        fpc = (1 - tp[mask]).cumsum(0)
        tpc = tp[mask].cumsum(0)
        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc)
        per_class[int(c)] = np.array([average_precision(recall[:, j], precision[:, j])
                                      for j in range(len(IOU_THRESHOLDS))])
    ap = np.stack(list(per_class.values()))
    return {
        "map50": float(ap[:, 0].mean()),
        "map50_95": float(ap.mean()),
        "per_class_ap50": {COCO_CLASSES[c]: round(float(v[0]), 4) for c, v in per_class.items()},
    }


def evaluate_coco128(model: str, conf: float = 0.001, iou: float = 0.7) -> dict:
    root = download_coco128()
    det = YoloDetector(model, conf_threshold=conf, iou_threshold=iou)
    tps, confs, pcls, gcls = [], [], [], []
    for img_path in sorted((root / "images" / "train2017").glob("*.jpg")):
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        gt_boxes, gt_cls = load_labels(root / "labels" / "train2017" / f"{img_path.stem}.txt", w, h)
        d = det.detect(image)
        tps.append(match_predictions(d.boxes, d.classes, gt_boxes, gt_cls))
        confs.append(d.scores)
        pcls.append(d.classes)
        gcls.append(gt_cls)
    result = compute_map(np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), np.concatenate(gcls))
    result["n_images"] = 128
    return result


def benchmark_speed(model: str, sizes=((768, 432), (1280, 720), (1920, 1080)), runs: int = 30) -> dict:
    """Median end-to-end detection latency on CPU per input resolution."""
    det = YoloDetector(model)
    out = {}
    for w, h in sizes:
        img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
        for _ in range(3):
            det.detect(img)
        stages = {"preprocess": [], "inference": [], "postprocess": []}
        totals = []
        for _ in range(runs):
            t = time.perf_counter()
            d = det.detect(img)
            totals.append((time.perf_counter() - t) * 1e3)
            for k in stages:
                stages[k].append(d.timings_ms[k])
        med = float(np.median(totals))
        out[f"{w}x{h}"] = {"total_ms": round(med, 1), "fps": round(1000 / med, 1),
                           **{k: round(float(np.median(v)), 1) for k, v in stages.items()}}
    return out


def main() -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)
    results = {}
    for model in MODELS:
        print(f"Evaluating {model} on COCO128...")
        acc = evaluate_coco128(model)
        speed = benchmark_speed(model)
        results[model] = {"coco128": acc, "speed_cpu": speed}
        print(f"  mAP50 {acc['map50']:.3f} | mAP50-95 {acc['map50_95']:.3f} | person AP50 "
              f"{acc['per_class_ap50'].get('person', 0):.3f} | 1280x720: {speed['1280x720']['fps']} FPS")
    (REPORTS_DIR / "detector_benchmark.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
