"""Marker PDF 解析工具。

对应架构 3.7 文档处理类。基于 marker 库（本地 Python，非 Docker）。
marker 未安装时降级到 PyMuPDF（fitz），再不可用时降级到打桩。
"""

from __future__ import annotations

import time
from typing import Any

from .base import BaseTool, ToolResult, ToolSchema


class PdfMarkerTool(BaseTool):
    """PDF 解析工具，优先 Marker，降级 PyMuPDF。"""

    schema = ToolSchema(
        name="parse_pdf",
        description="解析 PDF 提取正文与元数据（Marker 优先，PyMuPDF 降级）",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "PDF 文件路径"},
                "max_pages": {"type": "integer", "description": "最大解析页数", "default": 20},
            },
            "required": ["file_path"],
        },
    )

    def __init__(self) -> None:
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """探测可用 PDF 后端：marker > pymupdf > none。"""
        try:
            from marker.converters.pdf import PdfConverter  # noqa: F401

            return "marker"
        except ImportError:
            pass
        try:
            import fitz  # noqa: F401  PyMuPDF

            return "pymupdf"
        except ImportError:
            return "none"

    def call(self, **kwargs: Any) -> ToolResult:
        start = time.time()
        file_path = kwargs.get("file_path", "")
        max_pages = int(kwargs.get("max_pages", 20))
        if not file_path:
            return ToolResult(success=False, error="缺少 file_path", duration_ms=(time.time() - start) * 1000)

        if self._backend == "marker":
            return self._parse_with_marker(file_path, max_pages, start)
        if self._backend == "pymupdf":
            return self._parse_with_pymupdf(file_path, max_pages, start)
        return ToolResult(
            success=False,
            error="PDF 解析不可用（marker 与 pymupdf 均未安装，请 pip install marker-py 或 pymupdf）",
            duration_ms=(time.time() - start) * 1000,
        )

    def _parse_with_marker(self, file_path: str, max_pages: int, start: float) -> ToolResult:
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            converter = PdfConverter(artifact_path="default", config={"max_pages": max_pages})
            rendered = converter(file_path)
            text, _, _ = text_from_rendered(rendered)
            meta = self._extract_meta_pymupdf(file_path)
            return ToolResult(
                success=True,
                content=[{"type": "text", "text": text, "metadata": meta, "backend": "marker"}],
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            # marker 失败降级到 pymupdf
            return self._parse_with_pymupdf(file_path, max_pages, start, fallback_err=str(e))

    def _parse_with_pymupdf(self, file_path: str, max_pages: int, start: float, fallback_err: str = "") -> ToolResult:
        try:
            import fitz

            doc = fitz.open(file_path)
            pages_to_read = min(len(doc), max_pages)
            text_parts = [doc[i].get_text() for i in range(pages_to_read)]
            text = "\n\n".join(text_parts)
            meta = self._extract_meta_pymupdf(file_path)
            backend = "pymupdf" + ("(marker降级)" if fallback_err else "")
            return ToolResult(
                success=True,
                content=[{"type": "text", "text": text, "metadata": meta, "backend": backend}],
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"PDF 解析失败（{fallback_err or e}）",
                duration_ms=(time.time() - start) * 1000,
            )

    @staticmethod
    def _extract_meta_pymupdf(file_path: str) -> dict:
        try:
            import fitz

            doc = fitz.open(file_path)
            info = doc.metadata or {}
            return {
                "title": info.get("title", ""),
                "author": info.get("author", ""),
                "pages": len(doc),
                "subject": info.get("subject", ""),
            }
        except Exception:
            return {}
