"""工具集层。

对应架构 3.7 工具集层：所有工具均通过 MCP 协议接入。
MVP 阶段主用 OpenAlex（国内访问稳定、免费、覆盖 2.5 亿+ 文献）；
arXiv 与 Semantic Scholar 作为备选保留，阶段二可扩展。
"""
from .arxiv_tool import ArxivTool
from .openalex_tool import OpenAlexTool
from .semantic_scholar_tool import SemanticScholarTool
from .base import ToolResult, ToolSchema

__all__ = [
    "ArxivTool",
    "OpenAlexTool",
    "SemanticScholarTool",
    "ToolResult",
    "ToolSchema",
]
