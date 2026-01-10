# ocr_vl/io_utils.py
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import List
import requests
from .config import IMAGE_EXTS, TMP_PREFIX
from .schemas import ImageInput


def is_http_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def list_images_in_dir(dir_path: str) -> List[ImageInput]:
    p = Path(dir_path)
    if not p.is_dir():
        raise FileNotFoundError(f"dir_path is not a directory: {dir_path}")

    images: List[ImageInput] = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            images.append(ImageInput(id=f.stem, path=str(f)))
    return images


def download_to_local(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    suffix = Path(url).suffix or ".png"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=TMP_PREFIX)
    os.close(fd)

    with open(tmp_path, "wb") as f:
        f.write(resp.content)

    return tmp_path


def ensure_local_path(path: str) -> str:
    if is_http_url(path):
        return download_to_local(path)
    return path
