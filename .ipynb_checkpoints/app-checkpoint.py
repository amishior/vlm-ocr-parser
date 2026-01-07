# app.py
from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException

from scripts.model import init_pipeline
from scripts.schemas import BatchOCRRequest, BatchOCRResponse
from scripts.service import build_image_list_from_request, run_batch_ocr_sync

app = FastAPI(title="PaddleOCR-VL Markdown OCR Service")

_pipeline_lock = asyncio.Lock()


@app.on_event("startup")
async def on_startup():

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, init_pipeline)
    print("[Server] PaddleOCR-VL loaded on startup")


@app.post("/parse", response_model=BatchOCRResponse)
async def parse_images(req: BatchOCRRequest):

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
