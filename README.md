# vlm-ocr-parser

[中文](#中文) | [English](#english)

---

## 中文

`vlm-ocr-parser` 是一个基于 **PaddleOCR-VL** 的轻量级文档解析封装，目标是让你用尽可能少的工程成本，把「图片 / PDF / 文件夹」快速解析成可用于 **RAG / 检索 / 对比** 的结构化结果。

### ✨ 特性

- **统一的 Python 接口**：图片 / PDF / 目录一键解析  
- **自动设备选择**：优先尝试 GPU；遇到 `Unsupported GPU architecture` 等问题时自动回退到 CPU  
- **结果持久化**：支持保存为 **Markdown / JSON**，方便后续进入 RAG、差异比对、质检等流程  

> 本仓库是对官方 PaddleOCR-VL pipeline 的一层薄封装，方便你快速集成到自身项目中。  
> （以上能力点来自仓库现有 README 描述）  

### 📦 安装

建议 Python 3.10+，新建虚拟环境后：

```bash
pip install -r requirements.txt
````

> 如你需要 GPU 推理，请确保本机 CUDA / 驱动 / Paddle 版本与环境匹配；否则会自动回退到 CPU（性能会慢一些）。

### 🚀 快速开始（示例）

仓库提供了 `example/` 目录以及示例图片 `1.png ~ 9.png`，可以用它们快速验证。

推荐用法（思路示例）：

1. 安装依赖
2. 运行示例/服务（见下方「运行方式」）
3. 查看输出的 Markdown / JSON 文件

```python
# TODO: 把这里替换成实际对外暴露的入口
# from ocr_vl import PaddleOCRVLWrapper

# parser = PaddleOCRVLWrapper(device="auto")
# result = parser.parse("1.png")           # or parse_pdf("xx.pdf") / parse_dir("some_dir")
# parser.save(result, out_dir="outputs/", formats=["md", "json"])
```

### 🧩 运行方式


* **库式调用（Python import）**：适合接入自己的系统/RAG流水线
* **服务式调用（HTTP API）**：适合给别的同事/下游系统直接调用


* `ocr_vl/`：OCR/VL 解析封装核心模块
* `vl_service/`：VL 推理或对外服务相关
* `ocr_service/`：OCR 服务（路由/接口/封装）
* `app.py`：可能是一个启动入口（例如 FastAPI/Gradio/CLI）

### 📁 项目结构

```text
vlm-ocr-parser/
  example/          # 示例
  ocr_vl/           # OCR/VL 解析封装核心
  ocr_service/      # OCR 服务相关
  vl_service/       # VL 服务相关
  app.py            # 可能的启动入口
  requirements.txt  # 依赖
  1.png ... 9.png   # 示例图片
  README.md
```

### 📝 输出格式建议（对接 RAG / 对比 / 质检更省心）

建议你在输出 JSON 时尽量稳定包含这些字段（对下游很友好）：

* `meta`: 文件名、页码、耗时、device(gpu/cpu)、模型版本
* `blocks`: 每个 block 的 `type`（text/table/title/figure...）、`text`、`bbox`（可选）
* `md`: 拼接后的 Markdown（可选）
* `warnings`: 例如自动回退、页方向纠正、低置信度区域（可选）

### 🛠️ 常见问题（可逐步补充）

* **GPU 不可用 / 架构不支持**：会自动回退到 CPU，但速度变慢
* **长文档输出过长**：建议分页输出 / 分块保存，避免一次生成超长文本
* **表格不稳定**：可考虑结合版面检测或后处理（对齐、合并、校验）

### 📄 License

如需开源发布，建议补一个 LICENSE（MIT/Apache-2.0 等）。

---

## English

`vlm-ocr-parser` is a lightweight wrapper built on top of **PaddleOCR-VL**, designed to parse **images / PDFs / directories** into structured outputs that are easy to plug into **RAG / retrieval / diff & QC pipelines**.

### ✨ Key Features

* **Unified Python interface** for image / PDF / directory parsing
* **Auto device selection**: prefer GPU, fallback to CPU when GPU is not supported (e.g., `Unsupported GPU architecture`)
* **Persistence**: export results to **Markdown / JSON** for downstream usage

> The above feature bullets are aligned with the existing README in this repository.

### 📦 Installation

Python 3.10+ recommended:

```bash
pip install -r requirements.txt
```

### 🚀 Quick Start

This repo includes `example/` and sample images (`1.png` ~ `9.png`) so you can quickly verify the pipeline end-to-end.

Suggested usage pattern:

```python
# TODO: Replace with your actual public API
# from ocr_vl import PaddleOCRVLWrapper

# parser = PaddleOCRVLWrapper(device="auto")
# result = parser.parse("1.png")            # or parse_pdf("xx.pdf") / parse_dir("some_dir")
# parser.save(result, out_dir="outputs/", formats=["md", "json"])
```

### 🧩 Recommended Entrypoints

To make adoption easier, it’s best to provide:

* **Library-style API** (import & call in Python)
* **Service-style API** (HTTP endpoints)

Based on the current repository layout, you likely have:

* `ocr_vl/`: core parsing wrapper
* `vl_service/`: VLM inference / service utilities
* `ocr_service/`: OCR service layer
* `app.py`: app/server entrypoint

### 📁 Project Structure

```text
vlm-ocr-parser/
  example/
  ocr_vl/
  ocr_service/
  vl_service/
  app.py
  requirements.txt
  1.png ... 9.png
  README.md
```

### 📝 Output Schema (Best Practice)

For stable downstream integration, consider exporting JSON with:

* `meta`: filename, page index, latency, device, model version
* `blocks`: list of `{type, text, bbox(optional)}`
* `md`: merged markdown (optional)
* `warnings`: fallback / rotation / low-confidence hints (optional)

### 📄 License

Consider adding a LICENSE file (MIT/Apache-2.0).

[1]: https://github.com/amishior/vlm-ocr-parser.git "GitHub - amishior/vlm-ocr-parser"
