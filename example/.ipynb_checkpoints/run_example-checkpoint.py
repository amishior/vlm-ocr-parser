"""
简单的命令行示例：

python -m example.run_example \
    --input tests/demo.png \
    --output outputs
"""

import argparse
from pathlib import Path

from vlm_ocr_parser import create_default_parser


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL 文档解析示例（带 GPU/CPU 自动回退）"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="待解析的图片 / PDF / 目录（本地路径或 URL）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="结果输出目录（默认: outputs）",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU（忽略 GPU）",
    )
    parser.add_argument(
        "--batch-dir",
        action="store_true",
        help="将 --input 作为目录，批量解析其中的文件",
    )
    return parser.parse_args()


def main():
    args = build_args()

    prefer_gpu = not args.cpu
    parser = create_default_parser(
        prefer_gpu=prefer_gpu,
        output_dir=args.output,
    )

    input_path = Path(args.input)

    if args.batch_dir and input_path.is_dir():
        parser.parse_dir(input_path)
    else:
        parser.parse(input_path)

    print("done ✅")


if __name__ == "__main__":
    main()
