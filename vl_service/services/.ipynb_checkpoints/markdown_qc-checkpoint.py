# services/markdown_qc.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Any
import asyncio
from openai import OpenAI
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_QC_MODEL_NAME,
    QC_MAX_LEN,
    QC_MIN_LEN,
)

# ========= 客户端 =========
client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)


# ========= 工具：递归切片 =========

def _split_markdown_recursively(
    text: str,
    max_len: int = QC_MAX_LEN,
    min_len: int = QC_MIN_LEN,
) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_len:
        return [text]

    window = text[:max_len]

    def find_split_index(s: str, token: str) -> int:
        idx = s.rfind(token)
        return idx

    # 1) 优先按 "\n## " 切
    idx = find_split_index(window, "\n## ")
    if idx < min_len:
        idx = -1

    # 2) 其次按 "\n### "
    if idx == -1:
        idx = find_split_index(window, "\n### ")
        if idx < min_len:
            idx = -1

    # 3) 再不行按普通换行
    if idx == -1:
        idx = find_split_index(window, "\n\n")
        if idx < min_len:
            idx = -1

    # 4) 都找不到，只能硬切
    if idx == -1:
        idx = max_len

    left = text[:idx].strip()
    right = text[idx:].strip()

    chunks: List[str] = []
    if left:
        chunks.extend(_split_markdown_recursively(left, max_len, min_len))
    if right:
        chunks.extend(_split_markdown_recursively(right, max_len, min_len))
    return chunks


# ========= 工具：质检单个 chunk =========

async def _llm_qc_single_chunk(chunk_markdown: str) -> str:
    system_prompt = (
        "你是一名排版质检助手。现在给你一段由 OCR 得到的保险合同 Markdown 文本，"
        "请在**不改变原文含义**的前提下做以下优化：\n"
        "1. 修正常见 OCR 错别字（例如逗号后直接接“否则本公司”时中英文标点混用等）。\n"
        "2. 修正明显的序号错标，例如同一文段中序号过大、过小、冗余、错漏等。\n"
        "3. 优化 markdown 版式：\n"
        "   - 标题结构（# / ## / ###）保持合理、清晰；\n"
        "   - 段落之间加入必要的空行；\n"
        "   - 有序/无序列表使用标准 markdown 语法；\n"
        "   - URL 前后适当加空格，避免粘连；\n"
        "4. 不要删减条款内容，不要新编内容。\n"
        "5. 不要包裹 ```markdown 或 ``` 代码块，只输出纯 markdown。"
    )

    user_prompt = (
        "请对下面这段 markdown 文本进行排版和格式优化，"
        "保持语义不变，只在格式、标点、换行、标题层级等方面做改进：\n\n"
        "---------------- 原文开始 ----------------\n"
        f"{chunk_markdown}\n"
        "---------------- 原文结束 ----------------\n"
        "严禁输出任何其他无关内容。"
    )

    resp = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=LLM_QC_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"enable_thinking": False},
        ),
    )

    md = resp.choices[0].message.content or ""
    md = md.strip()
    if md.startswith("```"):
        md = md.strip("`").strip()
    return md


# ========= 并发质检多个 chunk =========

async def _qc_markdown_async(merged_markdown: str) -> Dict[str, Any]:
    merged_markdown = merged_markdown.strip()
    if not merged_markdown:
        return {
            "clean_markdown": "",
            "raw_chunks": [],
            "clean_chunks": [],
        }

    raw_chunks = _split_markdown_recursively(
        merged_markdown,
        max_len=QC_MAX_LEN,
        min_len=QC_MIN_LEN,
    )

    tasks = [
        asyncio.create_task(_llm_qc_single_chunk(ch))
        for ch in raw_chunks
    ]
    clean_chunks = await asyncio.gather(*tasks)

    clean_markdown = "\n\n".join(ch.strip() for ch in clean_chunks if ch.strip())

    return {
        "clean_markdown": clean_markdown,
        "raw_chunks": raw_chunks,
        "clean_chunks": clean_chunks,
    }


# ========= 对外：给 batch 增加 cleaned_markdown =========

async def get_cleaned_markdown_from_batch_async(
    batch: Dict[str, Any]
) -> Dict[str, Any]:
    merged_md = batch.get("merged_markdown", "") or ""

    qc_result = await _qc_markdown_async(merged_md)
    clean_md = qc_result["clean_markdown"]

    new_batch = dict(batch)
    new_batch["cleaned_markdown"] = clean_md
    return new_batch
