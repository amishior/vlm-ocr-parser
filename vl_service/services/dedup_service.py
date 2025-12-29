# services/dedup_service.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Any
from copy import deepcopy
import re


# ============== 文本规范化工具 ==============

def normalize_for_dedup(text: str) -> str:
    if not text:
        return ""

    t = text.strip().lower()

    rep_map = {
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "，": ",",
        "。": ".",
        "、": ",",
        "；": ";",
        "：": ":",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "！": "!",
        "？": "?",
        "—": "-",
        "－": "-",
        "·": ".",
    }
    for k, v in rep_map.items():
        t = t.replace(k, v)

    t = re.sub(r"\s+", "", t)

    t = re.sub(r"^[,.;:!?()\"'\-]+", "", t)
    t = re.sub(r"[,.;:!?()\"'\-]+$", "", t)

    return t


def is_heading_block(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()

    if stripped.startswith("#"):
        return True

    if re.match(r"^\d+(\.\d+)*[\.、]?\s*[\u4e00-\u9fff]", stripped):
        if len(stripped) <= 40:
            return True

    return False


def split_markdown_to_blocks(md: str) -> List[str]:
    if not md:
        return []

    raw_blocks = re.split(r"\n\s*\n", md)
    blocks = [b.strip("\n") for b in raw_blocks]
    return [b for b in blocks if b.strip()]


# ============== 辅助：记录删除信息 ==============

def _record_removed(images: List[Dict[str, Any]], blk: Dict[str, Any], reason: str) -> None:
    page_idx = blk["page_idx"]
    if page_idx < 0 or page_idx >= len(images):
        return
    img = images[page_idx]
    if "removed_segments" not in img or img["removed_segments"] is None:
        img["removed_segments"] = []

    img["removed_segments"].append(
        {
            "segment_type": "text" if "<table" not in blk["text"].lower() else "table",
            "source_image_id": img.get("id"),
            "source_image_path": img.get("path"),
            "reason": reason,
            "content": blk["text"],
        }
    )


# ============== 主去重逻辑 ==============

def deduplicate_batch_strict(batch: Dict[str, Any]) -> Dict[str, Any]:

    result = deepcopy(batch)
    images = result.get("images") or []

    all_blocks: List[Dict[str, Any]] = []

    for page_idx, img in enumerate(images):
        md = img.get("markdown") or ""
        blocks_text = split_markdown_to_blocks(md)

        blocks_objs: List[Dict[str, Any]] = []
        for seg_idx, txt in enumerate(blocks_text):
            blk = {
                "page_idx": page_idx,
                "seg_idx": seg_idx,
                "text": txt,
                "norm": normalize_for_dedup(txt),
                "is_heading": is_heading_block(txt),
                "keep": True,
                "removed_reason": None,
            }
            blocks_objs.append(blk)
            all_blocks.append(blk)

        img["_blocks"] = blocks_objs
        img["removed_segments"] = img.get("removed_segments") or []

    MIN_LEN_EXACT = 20
    MIN_LEN_OVERLAP = 30
    OVERLAP_RATIO = 0.7

    for i, blk in enumerate(all_blocks):
        if not blk["keep"]:
            continue
        norm_i = blk["norm"]
        if not norm_i:
            blk["keep"] = False
            blk["removed_reason"] = "empty_after_normalize"
            _record_removed(images, blk, "empty_after_normalize")
            continue

        for j in range(i):
            prev = all_blocks[j]
            if not prev["keep"]:
                continue
            norm_j = prev["norm"]
            if not norm_j:
                continue

            # ---- 标题块：只做“完全相等去重” ----
            if blk["is_heading"] and prev["is_heading"]:
                if norm_i == norm_j:
                    blk["keep"] = False
                    blk["removed_reason"] = "duplicate_heading"
                    _record_removed(images, blk, "duplicate_heading")
                    break
                else:
                    continue

            # ---- 普通文本块：精细去重 ----
            len_i = len(norm_i)
            len_j = len(norm_j)

            # 1) 完全相等：删掉后出现的
            if norm_i == norm_j and len_i >= MIN_LEN_EXACT:
                blk["keep"] = False
                blk["removed_reason"] = "duplicate_block_exact"
                _record_removed(images, blk, "duplicate_block_exact")
                break

            # 2) 前缀 / 子串重叠：删掉“更短”的那一个
            short, long = (norm_i, norm_j) if len_i <= len_j else (norm_j, norm_i)

            if len(short) >= MIN_LEN_OVERLAP:
                if short in long:
                    ratio = len(short) / len(long)
                    if ratio >= OVERLAP_RATIO:
                        if len_i < len_j:
                            blk["keep"] = False
                            blk["removed_reason"] = "shorter_overlap_with_previous"
                            _record_removed(images, blk, "shorter_overlap_with_previous")
                        else:
                            prev["keep"] = False
                            prev["removed_reason"] = "shorter_overlap_with_later"
                            _record_removed(images, prev, "shorter_overlap_with_later")
                        break

    doc_has_table = False
    for img in images:
        blocks = img.pop("_blocks", [])
        kept_blocks = [b for b in sorted(blocks, key=lambda x: x["seg_idx"]) if b["keep"]]
        new_md = "\n\n".join(b["text"] for b in kept_blocks)
        img["markdown"] = new_md

        if "<table" in new_md.lower():
            img["segment_type"] = "table"
            doc_has_table = True
        else:
            img["segment_type"] = "text"

    merged_markdown = "\n\n".join(
        img["markdown"] for img in images if img.get("markdown")
    )
    result["merged_markdown"] = merged_markdown
    result["segment_type"] = (
        "table" if "<table" in merged_markdown.lower() or doc_has_table else "text"
    )

    all_removed: List[Dict[str, Any]] = []
    for img in images:
        for r in img.get("removed_segments") or []:
            all_removed.append(r)
    result["removed_segments"] = all_removed

    result["image_count"] = len(images)

    return result
