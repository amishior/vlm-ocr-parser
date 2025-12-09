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
    """
    遍历目录，返回所有图片文件的 ImageInput 列表。
    """
    p = Path(dir_path)
    if not p.is_dir():
        raise FileNotFoundError(f"dir_path is not a directory: {dir_path}")

    images: List[ImageInput] = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            images.append(ImageInput(id=f.stem, path=str(f)))
    return images


def download_to_local(url: str) -> str:
    """
    简单 HTTP 下载到临时文件。
    生产环境可改为 OSS SDK / 内网网关。
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    suffix = Path(url).suffix or ".png"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=TMP_PREFIX)
    os.close(fd)

    with open(tmp_path, "wb") as f:
        f.write(resp.content)

    return tmp_path


def ensure_local_path(path: str) -> str:
    """
    入参可以是本地路径或 HTTP(S) URL：
    - 本地路径：原样返回
    - URL：下载到临时文件后返回本地路径
    """
    if is_http_url(path):
        return download_to_local(path)
    return path
