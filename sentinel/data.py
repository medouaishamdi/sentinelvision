"""Download the sample surveillance videos (Intel IoT DevKit sample-videos, CC BY 4.0).

    python -m sentinel.data
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from .config import SAMPLE_VIDEOS_URL, VIDEOS_DIR
from .scenes import SCENES


def video_path(scene: str, download: bool = True) -> Path:
    name = SCENES[scene]["video"]
    path = VIDEOS_DIR / f"{name}.mp4"
    if download and not path.exists():
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        url = SAMPLE_VIDEOS_URL.format(name=name)
        print(f"Downloading {url} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def download_all() -> None:
    for scene in SCENES:
        print(scene, "->", video_path(scene))


if __name__ == "__main__":
    download_all()
