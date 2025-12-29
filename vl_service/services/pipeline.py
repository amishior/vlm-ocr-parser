# services/pipeline.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, List
import asyncio

from .ocr_client import call_ocr_service_with_engine
from .dedup_service import deduplicate_batch_strict
from .markdown_qc import get_cleaned_markdown_from_batch_async


async def run_vl_pipeline(batch_input: Dict[str, Any]) -> Dict[str, Any]:

    file_list: List[str] = batch_input["file_list"]
    batch_id: str = batch_input["batch_id"]
    ocr_engine: str | None = batch_input.get("ocr_engine")

    # ---- 1. 调用 OCR 服务（同步 requests / OpenAI 放到线程池里） ----
    loop = asyncio.get_running_loop()

    # 用 lambda 包一层，把 ocr_engine 传进去
    ocr_batch = await loop.run_in_executor(
        None,
        lambda: call_ocr_service_with_engine(
            batch_id=batch_id,
            file_list=file_list,
            ocr_engine=ocr_engine,
        ),
    )

    # ---- 2. 合并额外元字段 (type_name / product_name / oss_name) ----
    batch = dict(ocr_batch)
    for key in ["type_name", "product_name", "oss_name"]:
        if key in batch_input:
            batch[key] = batch_input[key]

    batch["batch_id"] = batch_id

    # ---- 3. 去重 ----
    deduped_batch = deduplicate_batch_strict(batch)

    # ---- 4. LLM 质检（增加 cleaned_markdown）----
    final_batch = await get_cleaned_markdown_from_batch_async(deduped_batch)

    return final_batch
