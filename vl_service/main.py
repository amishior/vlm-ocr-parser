# main.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.pipeline import run_vl_pipeline


class VLRequest(BaseModel):
    type_name: str
    product_name: str
    batch_id: str
    file_list: List[str]
    oss_name: Optional[str] = None


app = FastAPI(title="VL Markdown 解析服务（OCR + 去重 + LLM 质检）")


@app.post("/vl_parse", response_model=Dict[str, Any])
async def vl_parse(req: VLRequest):
    try:
        batch_input = req.dict()
        result = await run_vl_pipeline(batch_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}
