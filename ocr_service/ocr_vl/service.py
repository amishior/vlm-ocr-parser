# ocr_vl/service.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from .config import MERGE_SEPARATOR
from .io_utils import ensure_local_path, list_images_in_dir
from .model import init_pipeline
from .schemas import BatchOCRRequest, BatchOCRResponse, ImageInput, ImageResult


def _extract_markdown_from_result(res) -> str:
    """
    从 PaddleOCRVL 单页结果对象中抽取 Markdown 字符串。
    做一点版本兼容兜底。
    """
    md_text = None

    # 1) 首选：res.markdown["markdown_texts"]
    if hasattr(res, "markdown") and res.markdown:
        if isinstance(res.markdown, dict):
            md_text = res.markdown.get("markdown_texts")

    # 2) 次选：res.json["markdown"]["markdown_texts"]
    if not md_text and hasattr(res, "json") and isinstance(res.json, dict):
        md_text = (
            res.json
            .get("markdown", {})
            .get("markdown_texts")
        )

    # 3) 兜底：把 text block 拼成纯文本 Markdown
    if not md_text and hasattr(res, "json") and isinstance(res.json, dict):
        parsing_res_list = res.json.get("parsing_res_list", [])
        texts = []
        for block in parsing_res_list:
            if block.get("block_label") == "text":
                content = block.get("block_content", "")
                if content:
                    texts.append(content)
        md_text = "\n\n".join(texts)

    return md_text or ""


def build_image_list_from_request(req: BatchOCRRequest) -> List[ImageInput]:
    """
    根据请求里的 dir_path + images 组合出最终的图片列表。
    """
    image_list: List[ImageInput] = []

    # 来自目录
    if req.dir_path:
        dir_images = list_images_in_dir(req.dir_path)
        image_list.extend(dir_images)

    # 来自显式列表
    image_list.extend(req.images)

    return image_list


def run_batch_ocr_sync(images: List[ImageInput],
                       batch_id: Optional[str] = None) -> BatchOCRResponse:
    """
    同步批量 OCR 核心逻辑：
    - 逐图调用 PaddleOCRVL
    - 抽取 markdown
    - 组装图片级结果 + 合并 markdown
    """
    pl = init_pipeline()
    image_results: List[ImageResult] = []

    for img in images:
        img_id = img.id or Path(img.path).stem
        original_path = img.path

        try:
            local_path = ensure_local_path(img.path)
            local_file = Path(local_path)
            if not local_file.is_file():
                raise FileNotFoundError(f"Image not found: {local_path}")

            outputs = pl.predict(input=str(local_file))

            md_parts: List[str] = []
            for res in outputs:
                md = _extract_markdown_from_result(res)
                if md:
                    md_parts.append(md)

            markdown = "\n\n".join(md_parts) if md_parts else ""

            image_results.append(
                ImageResult(
                    id=img_id,
                    path=original_path,
                    markdown=markdown,
                    error=None,
                )
            )

        except Exception as e:
            image_results.append(
                ImageResult(
                    id=img_id,
                    path=original_path,
                    markdown=None,
                    error=str(e),
                )
            )

    # 合并 Markdown，只拼接成功的图
    merged_parts: List[str] = []
    for ir in image_results:
        if ir.markdown:
            merged_parts.append(f"## {ir.id}\n\n{ir.markdown}")

    merged_markdown = MERGE_SEPARATOR.join(merged_parts)

    # 状态汇总
    status = "success"
    if any(ir.error for ir in image_results):
        status = "partial_success"
    if all(ir.error for ir in image_results):
        status = "failed"

    return BatchOCRResponse(
        batch_id=batch_id,
        image_count=len(image_results),
        status=status,
        images=image_results,
        merged_markdown=merged_markdown,
    )
