"""
统一封装 PaddleOCR-VL 的初始化和调用逻辑，
带 GPU / CPU 自动选择与错误回退。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import paddle
from paddleocr import PaddleOCRVL


PathLike = Union[str, Path]


@dataclass
class ParserConfig:
    """
    OCR-VL 解析器的配置。
    """

    prefer_gpu: bool = True
    use_layout_detection: bool = True
    format_block_content: bool = True
    output_dir: Path = Path("outputs")  # 用于保存 markdown/json 等结果


def _choose_device(prefer_gpu: bool = True) -> str:
    """
    根据当前环境自动选择 device。
    - 有 CUDA 并且 GPU 可以正常初始化 -> 'gpu'
    - 否则 -> 'cpu'
    """
    if prefer_gpu and paddle.device.is_compiled_with_cuda():
        try:
            paddle.set_device("gpu")
            _ = paddle.randn([1])
            print("[vlm-ocr-parser] 使用 GPU 推理")
            return "gpu"
        except Exception as e:
            print(
                "[vlm-ocr-parser] GPU 初始化失败，将回退到 CPU，原因：",
                repr(e),
            )

    paddle.set_device("cpu")
    print("[vlm-ocr-parser] 使用 CPU 推理")
    return "cpu"


class VLMOcrParser:
    """
    封装 PaddleOCRVL：
    - 自动选择 GPU / CPU，遇到 Unsupported GPU architecture 时自动回退
    - 提供简单的图片 / PDF / 目录解析接口
    """

    def __init__(self, config: ParserConfig | None = None):
        if config is None:
            config = ParserConfig()
        self.config = config

        # 确保输出目录存在
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = _choose_device(prefer_gpu=config.prefer_gpu)
        self.pipeline = self._create_pipeline()

    # ------------------------------------------------------------------ #
    # 初始化 & 兜底逻辑
    # ------------------------------------------------------------------ #
    def _create_pipeline(self) -> PaddleOCRVL:
        """
        实际构建 PaddleOCRVL Pipeline。
        遇到 Unsupported GPU architecture 时，会自动改用 CPU 重试一次。
        """
        try:
            return PaddleOCRVL(
                device=self.device,
                use_layout_detection=self.config.use_layout_detection,
                format_block_content=self.config.format_block_content,
            )
        except RuntimeError as e:
            msg = str(e)
            if "Unsupported GPU architecture" in msg and self.device != "cpu":
                print(
                    "[vlm-ocr-parser] 捕获到 Unsupported GPU architecture，"
                    "强制切换到 CPU 重新初始化……"
                )
                paddle.set_device("cpu")
                self.device = "cpu"
                return PaddleOCRVL(
                    device="cpu",
                    use_layout_detection=self.config.use_layout_detection,
                    format_block_content=self.config.format_block_content,
                )
            raise

    # ------------------------------------------------------------------ #
    # 对外解析接口
    # ------------------------------------------------------------------ #
    def parse(
        self,
        input_path: PathLike,
        save_markdown: bool = True,
        save_json: bool = False,
    ):
        """
        解析单个图片 / PDF / 远程 URL。

        参数
        ----
        input_path: 本地路径或 URL
        save_markdown: 是否将结果保存为 markdown 文件
        save_json: 是否将结果保存为 json 文件
        """
        input_path_str = str(input_path)
        print(f"[vlm-ocr-parser] 开始解析: {input_path_str}")

        results = self.pipeline.predict(input_path_str)

        if not (save_markdown or save_json):
            # 直接返回原始结果对象，给上层自行处理
            return results

        for res in results:
            # PaddleOCR-VL 的结果对象自带 save_to_markdown / save_to_json
            # 这里统一把文件写入到 config.output_dir 目录下
            if save_markdown:
                res.save_to_markdown(save_path=str(self.config.output_dir))
            if save_json:
                res.save_to_json(save_path=str(self.config.output_dir))

        print(f"[vlm-ocr-parser] 解析完成，结果已保存到: {self.config.output_dir}")
        return results

    def parse_dir(
        self,
        input_dir: PathLike,
        patterns: Sequence[str] = (".png", ".jpg", ".jpeg", ".pdf"),
        save_markdown: bool = True,
        save_json: bool = False,
    ):
        """
        批量解析一个目录下的所有图片 / PDF 文件。
        """
        input_dir = Path(input_dir)
        files: List[Path] = []

        for ext in patterns:
            files.extend(input_dir.rglob(f"*{ext}"))

        files = sorted(set(files))
        print(f"[vlm-ocr-parser] 在 {input_dir} 中发现 {len(files)} 个文件需要解析")

        all_results = {}
        for f in files:
            all_results[str(f)] = self.parse(
                f,
                save_markdown=save_markdown,
                save_json=save_json,
            )

        return all_results


# ---------------------------------------------------------------------- #
# 默认构造器，给 example / notebook 快速使用
# ---------------------------------------------------------------------- #
def create_default_parser(
    prefer_gpu: bool = True,
    output_dir: PathLike = "outputs",
) -> VLMOcrParser:
    """
    提供一个简单的默认构造器，方便在示例代码/Notebook 中使用。
    """
    cfg = ParserConfig(
        prefer_gpu=prefer_gpu,
        use_layout_detection=True,
        format_block_content=True,
        output_dir=Path(output_dir),
    )
    return VLMOcrParser(cfg)
