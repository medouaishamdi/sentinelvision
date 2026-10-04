import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentinel.detector import Detections  # noqa: E402


def make_dets(boxes, scores=None, classes=None) -> Detections:
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    scores = np.array(scores if scores is not None else [0.9] * n, dtype=np.float32)
    classes = np.array(classes if classes is not None else [0] * n, dtype=int)
    return Detections(boxes, scores, classes, {})


@pytest.fixture
def sample_image():
    """A real photo from COCO128 if available, else a synthetic frame."""
    import cv2

    candidates = sorted((ROOT / "data" / "coco128" / "images" / "train2017").glob("*.jpg"))
    if candidates:
        return cv2.imread(str(candidates[0]))
    return np.full((480, 640, 3), 127, np.uint8)
