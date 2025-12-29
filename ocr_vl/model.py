# ocr_vl/model.py
from __future__ import annotations

from typing import Optional

import paddle
from paddleocr import PaddleOCRVL

_pipeline: Optional[PaddleOCRVL] = None


def init_pipeline() -> PaddleOCRVL:
    """
    初始化并缓存 PaddleOCR-VL 模型，进程级单例。
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        device = "gpu:0"
    else:
        device = "cpu"

    print(f"[Model] Initializing PaddleOCRVL on device: {device}")

    pl = PaddleOCRVL(
        device=device,
        use_layout_detection=True,
        format_block_content=True,
    )

    print("[Model] PaddleOCR-VL 初始化完成")
    _pipeline = pl
    return _pipeline
