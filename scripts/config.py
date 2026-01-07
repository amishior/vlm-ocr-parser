# ocr_vl/config.py
from pathlib import Path

# 支持的图片后缀
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# 临时下载文件前缀（下载 OSS / HTTP 时用）
TMP_PREFIX = "paddleocr_vl_"

# 默认 Markdown 合并时的图片分隔符
MERGE_SEPARATOR = "\n\n---\n\n"

# 可以根据环境变量扩展，例如：
# import os
# OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "")
