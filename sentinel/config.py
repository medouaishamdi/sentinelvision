"""Paths and defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
DB_PATH = PROJECT_ROOT / "data" / "events.db"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_MODEL = "yolo11n"
MODELS = {
    "yolo11n": MODELS_DIR / "yolo11n.onnx",  # fastest, best for CPU real time
    "yolo11s": MODELS_DIR / "yolo11s.onnx",  # more accurate, ~2.5x slower
}

SAMPLE_VIDEOS_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/{name}.mp4"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# Classes a surveillance system usually cares about
SURVEILLANCE_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase"]
