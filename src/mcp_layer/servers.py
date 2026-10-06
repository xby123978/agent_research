"""MCP Server 实现（官方 mcp SDK / FastMCP）。

对应架构 3.5 MCP 协议层。每个 Server 都是一个**真实的 MCP Server**：
- 用官方 SDK（mcp.server.fastmcp.FastMCP）定义 Tool
- 支持两种传输：stdio（本地子进程）与 streamable-http（远程）
- 可被任意标准 MCP 客户端（Claude Desktop / Cursor / MCP Inspector）连接

独立启动：
    python -m src.mcp_layer.servers academic_search stdio
    python -m src.mcp_layer.servers academic_search http
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..common.logger import get_logger
from ..tools.base import ToolResult
from ..tools.crossref_tool import CrossRefTool
from ..tools.filesystem_tool import FilesystemTool
from ..tools.knowledge_base_tool import KnowledgeBaseTool
from ..tools.openalex_tool import OpenAlexTool
from ..tools.pdf_marker_tool import PdfMarkerTool

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _envelope(result: ToolResult) -> str:
    """把 ToolResult 序列化为 JSON 文本，作为 MCP 工具返回内容。

    客户端据此还原 ToolResult（保留 success / error / duration_ms）。
    """
    return json.dumps(
        {
            "tool_name": result.tool_name,
            "success": result.success,
            "content": result.content,
            "error": result.error,
            "duration_ms": result.duration_ms,
        },
        ensure_ascii=False,
    )


def build_academic_search_server() -> FastMCP:
    """学术检索 Server：OpenAlex + CrossRef 多源检索。"""
    server = FastMCP(
        name="academic_search",
        instructions="学术检索 Server，聚合 OpenAlex 与 CrossRef 多源文献检索能力。",
    )
    openalex = OpenAlexTool()
    crossref = CrossRefTool()

    @server.tool(
        name="search_openalex",
        description="检索 OpenAlex 学术文献库（2.5亿+文献，含引用数与摘要）",
    )
    def search_openalex(
        query: str, max_results: int = 10, sort_by: str = "relevance"
    ) -> str:
        """检索 OpenAlex 学术文献库。"""
        return _envelope(
            openalex.call(query=query, max_results=max_results, sort_by=sort_by)
        )

    @server.tool(
        name="search_crossref",
        description="通过 CrossRef 检索文献元数据（按标题/作者/DOI）",
    )
    def search_crossref(query: str, max_results: int = 10, doi: str = "") -> str:
        """检索 CrossRef 文献元数据。"""
        params: dict[str, Any] = {"query": query, "max_results": max_results}
        if doi:
            params["doi"] = doi
        return _envelope(crossref.call(**params))

    @server.resource("config://academic_search")
    def server_config() -> str:
        """Server 元信息（对应 MCP resources/read）。"""
        return json.dumps(
            {
                "name": "academic_search",
                "version": "1.0.0",
                "sources": ["openalex", "crossref"],
            },
            ensure_ascii=False,
        )

    @server.prompt(
        name="literature_review",
        description="生成文献综述写作的提示词模板。",
    )
    def literature_review_prompt(topic: str, language: str = "中文") -> str:
        """文献综述提示词（对应 MCP prompts/get）。"""
        return (
            f"你是一位学术研究助理，请围绕下列主题撰写一篇结构化文献综述。\n\n"
            f"主题：{topic}\n语言：{language}\n\n"
            f"要求：\n1. 梳理研究脉络与关键转折\n2. 归纳主要方法与流派\n"
            f"3. 指出争议与空白\n4. 给出未来方向\n5. 严格标注引用来源"
        )

    return server


def _extract_pdf_metadata(file_path: str) -> ToolResult:
    """提取 PDF 元数据（基于 PyMuPDF，无需 Grobid Docker 依赖）。"""
    start = time.time()
    try:
        import pymupdf as fitz

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
            tool_name="extract_metadata",
            success=True,
            content=[meta],
            duration_ms=(time.time() - start) * 1000,
        )
    except ImportError:
        return ToolResult(
            tool_name="extract_metadata",
            success=False,
            error="PyMuPDF 未安装，请 pip install pymupdf",
            duration_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ToolResult(
            tool_name="extract_metadata",
            success=False,
            error=str(e),
            duration_ms=(time.time() - start) * 1000,
        )


def build_document_processor_server() -> FastMCP:
    """文档处理 Server：PDF 解析（Marker 优先，PyMuPDF 降级）+ 元数据提取。"""
    server = FastMCP(
        name="document_processor",
        instructions="文档处理 Server，提供 PDF 解析与元数据提取能力。",
    )
    pdf = PdfMarkerTool()

    @server.tool(
        name="parse_pdf",
        description="解析 PDF 提取正文与元数据（Marker 优先，PyMuPDF 降级）",
    )
    def parse_pdf(file_path: str, max_pages: int = 20) -> str:
        """解析 PDF 正文。"""
        return _envelope(pdf.call(file_path=file_path, max_pages=max_pages))

    @server.tool(
        name="extract_metadata",
        description="提取文献元数据（基于 PyMuPDF，无需 Grobid）",
    )
    def extract_metadata(file_path: str) -> str:
        """提取 PDF 元数据。"""
        return _envelope(_extract_pdf_metadata(file_path))

    return server


def build_filesystem_server() -> FastMCP:
    """文件系统 Server：限 outputs/ 与 data/ 目录操作，禁止路径穿越。"""
    server = FastMCP(
        name="filesystem",
        instructions="文件系统 Server，限 outputs/ 与 data/ 目录操作。",
    )
    fs = FilesystemTool()

    @server.tool(
        name="filesystem",
        description="本地文件读写（限 outputs/ 与 data/ 目录）",
    )
    def filesystem(action: str, path: str, content: str = "") -> str:
        """本地文件 read / write / list / mkdir。"""
        return _envelope(fs.call(action=action, path=path, content=content))

    return server


def build_knowledge_base_server() -> FastMCP:
    """知识库 Server：对接记忆模块 MemoryManager，提供 search/add/get/graph。"""
    server = FastMCP(
        name="knowledge_base",
        instructions="知识库 Server，对接四层记忆，提供 search/add/get/graph 操作。",
    )
    kb = KnowledgeBaseTool()

    @server.tool(
        name="knowledge_base",
        description="记忆读写与知识图谱查询（对接四层记忆）",
    )
    def knowledge_base(
        action: str,
        query: str,
        layer: str = "auto",
        top_k: int = 5,
        item_id: str = "",
        depth: int = 1,
    ) -> str:
        """记忆读写与知识图谱查询。"""
        return _envelope(
            kb.call(
                action=action,
                query=query,
                layer=layer,
                top_k=top_k,
                item_id=item_id,
                depth=depth,
            )
        )

    return server


# Server 名称 → 构建函数
SERVER_BUILDERS: dict[str, Any] = {
    "academic_search": build_academic_search_server,
    "document_processor": build_document_processor_server,
    "filesystem": build_filesystem_server,
    "knowledge_base": build_knowledge_base_server,
}


def main(argv: list[str] | None = None) -> None:
    """独立启动入口：python -m src.mcp_layer.servers <server> [stdio|http]"""
    args = list(sys.argv[1:] if argv is None else argv)
    name = args[0] if args else "academic_search"
    transport = args[1] if len(args) > 1 else "stdio"

    builder = SERVER_BUILDERS.get(name)
    if builder is None:
        raise SystemExit(
            f"未知 Server: {name}（可选: {', '.join(SERVER_BUILDERS)}）"
        )

    server = builder()
    logger.info(f"启动 MCP Server: {name} transport={transport}")
    if transport in ("http", "streamable-http"):
        server.run(transport="streamable-http")
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
