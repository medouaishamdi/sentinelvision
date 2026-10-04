"""YOLO object detector running on ONNX Runtime (no PyTorch needed at runtime).

Pre-processing (letterbox), output decoding and non-maximum suppression are implemented
here explicitly, so the whole pipeline is transparent and dependency-light.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

from .config import COCO_CLASSES, DEFAULT_MODEL, MODELS


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy in original image pixels
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,) int class ids
    timings_ms: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scores)

    def filter(self, mask: np.ndarray) -> "Detections":
        return Detections(self.boxes[mask], self.scores[mask], self.classes[mask], self.timings_ms)

    def to_list(self) -> list[dict]:
        return [
            {"class": COCO_CLASSES[int(c)], "confidence": round(float(s), 4),
             "box": [round(float(v), 1) for v in b]}
            for b, s, c in zip(self.boxes, self.scores, self.classes)
        ]


def letterbox(image: np.ndarray, size: int = 640, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to a square. Returns padded image, scale, (pad_x, pad_y)."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else image
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Plain greedy non-maximum suppression. Returns kept indices sorted by score."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=int)


class YoloDetector:
    def __init__(self, model: str = DEFAULT_MODEL, conf_threshold: float = 0.35,
                 iou_threshold: float = 0.5, classes: list[str] | None = None, threads: int | None = None):
        path = MODELS[model] if model in MODELS else model
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(self.session.get_inputs()[0].shape[2])
        self.model_name = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_ids = None if classes is None else np.array([COCO_CLASSES.index(c) for c in classes])

    def detect(self, image: np.ndarray, conf_threshold: float | None = None, max_det: int = 300) -> Detections:
        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        t0 = time.perf_counter()
        padded, scale, (px, py) = letterbox(image, self.size)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0][0]  # (4 + 80, 8400)
        t2 = time.perf_counter()

        preds = out.T  # (8400, 84): cx, cy, w, h, 80 class scores
        class_scores = preds[:, 4:]
        if self.class_ids is not None:
            mask_cls = np.zeros(class_scores.shape[1], bool)
            mask_cls[self.class_ids] = True
            class_scores = np.where(mask_cls, class_scores, 0)
        cls = class_scores.argmax(1)
        scores = class_scores[np.arange(len(cls)), cls]
        keep = scores >= conf
        preds, scores, cls = preds[keep], scores[keep], cls[keep]

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        # Class-aware NMS: offset boxes per class so different classes never suppress each other
        offset = cls[:, None] * 4096.0
        kept = nms(boxes + offset, scores, self.iou_threshold)[:max_det]
        boxes, scores, cls = boxes[kept], scores[kept], cls[kept]

        # Undo letterbox -> original pixels
        boxes = (boxes - np.array([px, py, px, py])) / scale
        ih, iw = image.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, iw)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ih)
        t3 = time.perf_counter()
        timings = {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}
        return Detections(boxes.astype(np.float32), scores.astype(np.float32), cls.astype(int), timings)
