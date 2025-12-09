# ocr_vl/schemas.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="图片 ID，可选；不传则用 path 的文件名"
    )
    path: str = Field(
        description="图片路径：本地文件路径或 HTTP(S)/OSS URL"
    )


class BatchOCRRequest(BaseModel):
    batch_id: Optional[str] = Field(
        default=None,
        description="本次批处理任务 ID，如产品 ID，可选"
    )
    dir_path: Optional[str] = Field(
        default=None,
        description="本地目录路径，可选；会扫描目录下所有图片"
    )
    images: List[ImageInput] = Field(
        default_factory=list,
        description="显式图片列表，可选；dir_path 与 images 可同时存在"
    )


class ImageResult(BaseModel):
    id: str
    path: str
    markdown: Optional[str] = None
    error: Optional[str] = None


class BatchOCRResponse(BaseModel):
    batch_id: Optional[str]
    image_count: int
    status: str
    images: List[ImageResult]
    merged_markdown: str
