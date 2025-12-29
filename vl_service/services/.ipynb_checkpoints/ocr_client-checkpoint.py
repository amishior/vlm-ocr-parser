# services/ocr_client.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Any
import base64
import requests

from openai import OpenAI

from config import (
    OCR_PARSE_ENDPOINT,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    QWEN_VL_MODEL_NAME,
    OCR_ENGINE_DEFAULT,
)

# ========== Qwen3-VL 客户端 ==========

_qwen_client: OpenAI | None = None


def get_qwen_client() -> OpenAI:
    global _qwen_client
    if _qwen_client is None:
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法调用 qwen3-vl-plus")
        _qwen_client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )
    return _qwen_client


# ========== 通用工具 ==========


def build_images_payload(file_list: List[str]) -> List[Dict[str, str]]:
    """
    将 file_list 转为 PaddleOCR-VL 服务需要的 images 结构
    """
    return [
        {"id": f"page_{i + 1}", "path": p}
        for i, p in enumerate(file_list)
    ]


def _load_image_bytes(path: str) -> bytes:
    """
    支持本地路径 & HTTP(S) URL，返回二进制内容。
    """
    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path, timeout=30)
        resp.raise_for_status()
        return resp.content
    else:
        with open(path, "rb") as f:
            return f.read()


def _encode_image_to_base64(path: str) -> str:
    data = _load_image_bytes(path)
    return base64.b64encode(data).decode("utf-8")


# ========== PaddleOCR-VL ==========

def call_paddle_ocr_service(batch_id: str, file_list: List[str]) -> Dict[str, Any]:

    payload = {
        "batch_id": batch_id,
        "images": build_images_payload(file_list),
    }

    try:
        resp = requests.post(OCR_PARSE_ENDPOINT, json=payload, timeout=300)
    except Exception as e:
        raise RuntimeError(f"调用 PaddleOCR-VL 服务失败: {type(e).__name__}: {e}")

    if resp.status_code != 200:
        raise RuntimeError(
            f"PaddleOCR-VL 服务返回错误 HTTP {resp.status_code}: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"PaddleOCR-VL 服务返回非 JSON：{type(e).__name__}: {e}")

    return data


# ========== Qwen3-VL-Plus OCR ==========

def _qwen_vl_ocr_single_image(path: str) -> str:

    client = get_qwen_client()
    img_b64 = _encode_image_to_base64(path)

    prompt = """
你是一个专业的 OCR 解析助手，请严格按以下要求输出：

1. 读取图片中的全部文字内容（正文为主，可忽略电量、时间等手机状态栏信息）。
2. 使用 **标准 Markdown** 格式输出：
   - 标题用 `#` / `##` / `###` 表示；
   - 段落之间用空行分隔；
   - 列表项用 `1. ...`、`- ...` 等常规 markdown 语法；
   - 如识别到表格，可用 markdown 表格表示。
3. 不要补充图片中不存在的内容，不要改变条款含义，不要乱改顺序。
4. 不要输出 ```markdown 或 ``` 代码块。
5. 输出必须是 **纯 markdown 文本**，不要加任何说明性文字。
""".strip()

    resp = client.chat.completions.create(
        model=QWEN_VL_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        extra_body={"enable_thinking": False},
    )

    md = resp.choices[0].message.content or ""
    md = md.strip()
    if md.startswith("```"):
        md = md.strip("`").strip()
    return md


def call_qwen_vl_ocr_service(batch_id: str, file_list: List[str]) -> Dict[str, Any]:

    images: List[Dict[str, Any]] = []

    for i, p in enumerate(file_list):
        page_id = f"page_{i + 1}"
        print(f"[Qwen-VL OCR] 解析 {page_id}: {p}")
        try:
            md = _qwen_vl_ocr_single_image(p)
            err = None
        except Exception as e:
            md = ""
            err = f"{type(e).__name__}: {e}"
            print(f"[Qwen-VL OCR] {page_id} 失败: {err}")

        images.append(
            {
                "id": page_id,
                "path": p,
                "markdown": md,
                "error": err,
                "removed_segments": [],
                "segment_type": "text",
            }
        )

    any_ok = any((img.get("markdown") or "").strip() for img in images)
    status = "success" if any_ok else "failed"

    merged_markdown = "\n\n".join(
        img["markdown"] for img in images if img.get("markdown")
    )

    return {
        "batch_id": batch_id,
        "image_count": len(images),
        "status": status,
        "images": images,
        "merged_markdown": merged_markdown,
    }


# ========== 统一入口 + 引擎选择 + fallback ==========

def call_ocr_service_with_engine(
    batch_id: str,
    file_list: List[str],
    ocr_engine: str | None = None,
) -> Dict[str, Any]:

    engine = (ocr_engine or OCR_ENGINE_DEFAULT or "qwen3-vl-plus").lower()
    print(f"[OCR] 请求引擎 = {engine}, batch_id={batch_id}")

    # ---- 1) Qwen-VL 直接使用 ----
    if engine == "qwen3-vl-plus":
        return call_qwen_vl_ocr_service(batch_id, file_list)

    # ---- 2) PaddleOCR-VL + fallback ----
    if engine == "paddleocr-vl":
        try:
            batch = call_paddle_ocr_service(batch_id, file_list)
            imgs = batch.get("images") or []
            any_text = any((img.get("markdown") or "").strip() for img in imgs)

            if batch.get("status") != "success" or not any_text:
                raise RuntimeError(
                    f"PaddleOCR-VL 返回空结果，status={batch.get('status')}"
                )

            print("[OCR] PaddleOCR-VL 成功，无需 fallback")
            return batch

        except Exception as e:
            print(f"[OCR] PaddleOCR-VL 失败，fallback 到 Qwen3-VL-Plus: {e}")
            return call_qwen_vl_ocr_service(batch_id, file_list)

    # ---- 3) 未知引擎名：退回默认 Qwen-VL ----
    print(f"[OCR] 未知引擎 {engine}，使用默认 qwen3-vl-plus")
    return call_qwen_vl_ocr_service(batch_id, file_list)
