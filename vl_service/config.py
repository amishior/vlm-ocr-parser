# config.py
from __future__ import annotations
import os

# =====================================================
# 1. OCR 服务配置（PaddleOCR-VL FastAPI）
# =====================================================

OCR_SERVICE_BASE_URL = os.getenv("OCR_SERVICE_BASE_URL", "http://127.0.0.1:8000")
OCR_PARSE_ENDPOINT = f"{OCR_SERVICE_BASE_URL}/parse"


# =====================================================
# 2. DashScope / OpenAI 兼容接口配置（Qwen 系列）
#    - 同时给：Qwen3-VL OCR + LLM 质检 使用
# =====================================================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ========= 2.1 Qwen-VL OCR 模型 =========
QWEN_VL_MODEL_NAME = os.getenv("QWEN_VL_MODEL_NAME", "qwen3-vl-plus")
OCR_ENGINE_DEFAULT = os.getenv("OCR_ENGINE_DEFAULT", "qwen3-vl-plus")


# =====================================================
# 3. LLM 质检配置（Markdown 排版优化）
# =====================================================

LLM_API_KEY = os.getenv("LLM_API_KEY", DASHSCOPE_API_KEY)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", DASHSCOPE_BASE_URL)

LLM_QC_MODEL_NAME = os.getenv("QC_MODEL_NAME", "qwen3-max")

QC_MAX_LEN = int(os.getenv("QC_MAX_LEN", "4000"))
QC_MIN_LEN = int(os.getenv("QC_MIN_LEN", "1500"))
