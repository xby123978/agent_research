"""自定义 MCP Server 框架。

对应架构 3.5 MCP 协议层。每个 Server 封装一组工具，对外暴露 MCP 协议接口。
降级方案：进程内实现（不启动独立 stdio/HTTP 进程），与现有 MCPClient 兼容。
阶段二：学术检索 Server + 文档处理 Server（PDF 解析打桩）。
"""

from __future__ import annotations

import abc
import time
from typing import Any

from ..common.data_models import PaperMeta
from ..common.logger import get_logger
from ..tools.base import ToolResult
from ..tools.crossref_tool import CrossRefTool
from ..tools.filesystem_tool import FilesystemTool
from ..tools.knowledge_base_tool import KnowledgeBaseTool
from ..tools.openalex_tool import OpenAlexTool
from ..tools.pdf_marker_tool import PdfMarkerTool

logger = get_logger(__name__)


class BaseMCPServer(abc.ABC):
    """MCP Server 抽象基类。"""

    name: str = "base_mcp_server"

    @abc.abstractmethod
    def list_tools(self) -> list[dict[str, Any]]:
        """列出本 Server 提供的工具 schema。"""
        ...

    @abc.abstractmethod
    def call_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """调用工具。"""
        ...


class AcademicSearchServer(BaseMCPServer):
    """学术检索 MCP Server。

    聚合 OpenAlex + CrossRef 多源检索能力。
    """

    name = "academic_search"

    def __init__(self) -> None:
        self._openalex = OpenAlexTool()
        self._crossref = CrossRefTool()
        self._tools = {
            "search_openalex": self._openalex,
            "search_crossref": self._crossref,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": tool.schema.description,
                "parameters": tool.schema.parameters,
            }
            for name, tool in self._tools.items()
        ]

    def call_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(success=False, error=f"未知工具: {tool_name}")
        return tool.execute(params)

    def search_all(self, query: str, max_results: int = 10) -> list[PaperMeta]:
        """多源检索：OpenAlex + CrossRef，合并去重。"""
        all_papers: list[PaperMeta] = []
        # OpenAlex
        r1 = self._openalex.execute({"query": query, "max_results": max_results})
        if r1.success:
            for p_dict in r1.content:
                try:
                    all_papers.append(PaperMeta(**p_dict))
                except Exception:
                    pass
        # CrossRef（补充）
        r2 = self._crossref.execute({"query": query, "max_results": max_results})
        if r2.success:
            for p_dict in r2.content:
                try:
                    all_papers.append(PaperMeta(**p_dict))
                except Exception:
                    pass
        # 去重
        seen: set[str] = set()
        deduped = [
            p for p in all_papers
            if p.title and not (p.title in seen or seen.add(p.title))
        ]
        return deduped[:max_results]


class DocumentProcessorServer(BaseMCPServer):
    """文档处理 MCP Server。

    PDF 解析基于 PdfMarkerTool（Marker 优先，PyMuPDF 降级，均为本地 Python 库）。
    元数据提取复用 PyMuPDF（无需 Grobid Docker 依赖）。
    """

    name = "document_processor"

    def __init__(self) -> None:
        self._pdf = PdfMarkerTool()
        self._tools = {"parse_pdf": self._pdf}

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "parse_pdf",
                "description": self._pdf.schema.description,
                "parameters": self._pdf.schema.parameters,
            },
            {
                "name": "extract_metadata",
                "description": "提取文献元数据（基于 PyMuPDF，无需 Grobid）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "PDF 文件路径"},
                    },
                    "required": ["file_path"],
                },
            },
        ]

    def call_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        start = time.time()
        if tool_name == "parse_pdf":
            return self._pdf.call(**params)
        if tool_name == "extract_metadata":
            return self._extract_metadata(params.get("file_path", ""), start)
        return ToolResult(success=False, error=f"未知工具: {tool_name}")

    def _extract_metadata(self, file_path: str, start: float) -> ToolResult:
        """提取 PDF 元数据（基于 PyMuPDF，无需 Grobid Docker）。"""
        try:
            import fitz

            doc = fitz.open(file_path)
            info = doc.metadata or {}
            meta = {
                "title": info.get("title", ""),
                "author": info.get("author", ""),
                "pages": len(doc),
                "subject": info.get("subject", ""),
                "keywords": info.get("keywords", ""),
            }
            return ToolResult(
                success=True,
                content=[meta],
                duration_ms=(time.time() - start) * 1000,
            )
        except ImportError:
            return ToolResult(
                success=False,
                error="PyMuPDF 未安装，请 pip install pymupdf",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), duration_ms=(time.time() - start) * 1000)


class FilesystemServer(BaseMCPServer):
    """文件系统 MCP Server（本地实现，非 Docker）。

    对应架构 3.4.2 文件系统 Server。基于 FilesystemTool 封装，
    限 outputs/ 与 data/ 目录操作，禁止路径穿越。
    """

    name = "filesystem"

    def __init__(self) -> None:
        self._fs = FilesystemTool()
        self._tools = {"filesystem": self._fs}

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "filesystem",
                "description": self._fs.schema.description,
                "parameters": self._fs.schema.parameters,
            }
        ]

    def call_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        if tool_name != "filesystem":
            return ToolResult(success=False, error=f"未知工具: {tool_name}")
        return self._fs.call(**params)


class KnowledgeBaseServer(BaseMCPServer):
    """知识库 MCP Server（本地实现，非 Docker）。

    对应架构 3.4.2 知识库 Server。对接记忆模块 MemoryManager，
    提供 search/add/get/graph 四类操作。
    """

    name = "knowledge_base"

    def __init__(self) -> None:
        self._kb = KnowledgeBaseTool()
        self._tools = {"knowledge_base": self._kb}

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "knowledge_base",
                "description": self._kb.schema.description,
                "parameters": self._kb.schema.parameters,
            }
        ]

    def call_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        if tool_name != "knowledge_base":
            return ToolResult(success=False, error=f"未知工具: {tool_name}")
        return self._kb.call(**params)
