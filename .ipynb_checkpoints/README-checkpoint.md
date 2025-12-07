# vlm-ocr-parser

基于 [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) 的轻量级文档解析封装，
提供：

- **统一的 Python 接口**：图片 / PDF / 目录一键解析；
- **自动设备选择**：优先尝试 GPU，遇到 `Unsupported GPU architecture`
  等错误时自动回退到 CPU；
- **结果持久化**：支持将解析结果保存为 Markdown / JSON 文件，方便后续走 RAG 等流程。

> 本仓库只是对官方 PaddleOCR-VL pipeline 的一层薄封装，方便快速集成到自身项目中。

---

## 1. 安装

建议使用 Python 3.10+，新建虚拟环境后：

```bash
pip install -r requirements.txt
```
