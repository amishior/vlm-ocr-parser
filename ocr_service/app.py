# app.py
from __future__ import annotations
import asyncio
from fastapi import FastAPI, HTTPException
from ocr_vl.model import init_pipeline
from ocr_vl.schemas import BatchOCRRequest, BatchOCRResponse
from ocr_vl.service import build_image_list_from_request, run_batch_ocr_sync

app = FastAPI(title="PaddleOCR-VL Markdown OCR Service")

# 全局锁，避免并发访问 PaddleOCRVL 的潜在问题
_pipeline_lock = asyncio.Lock()


@app.on_event("startup")
async def on_startup():
    """
    服务启动时预先加载模型，避免首个请求冷启动。
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, init_pipeline)
    print("[Server] PaddleOCR-VL loaded on startup")


@app.post("/parse", response_model=BatchOCRResponse)
async def parse_images(req: BatchOCRRequest):
    """
    批量解析接口：
    - 支持 dir_path（本地目录）
    - 支持 images 列表（本地/HTTP/OSS）
    """
    image_list = build_image_list_from_request(req)
    if not image_list:
        raise HTTPException(status_code=400, detail="No images provided")

    async with _pipeline_lock:
        loop = asyncio.get_running_loop()
        result: BatchOCRResponse = await loop.run_in_executor(
            None,
            run_batch_ocr_sync,
            image_list,
            req.batch_id,
        )

    return result
