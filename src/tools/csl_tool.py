"""CSL 引用格式转换工具（架构 3.7）。

支持将文献元数据转换为多种引用格式：
- APA
- MLA
- Chicago
- IEEE
- GB/T 7714

输入：PaperMeta 的 dict
输出：格式化后的引用字符串
"""

from __future__ import annotations

from typing import Any

from ..common.logger import get_logger
from ..common.data_models import PaperMeta
from .base import BaseTool, ToolResult, ToolSchema

logger = get_logger(__name__)


class CSLTool(BaseTool):
    """CSL 引用格式转换工具。"""

    schema = ToolSchema(
        name="format_citation",
        description="将文献元数据转换为指定引用格式（APA/MLA/Chicago/IEEE/GB-T7714）",
        input_schema={
            "type": "object",
            "properties": {
                "paper": {"type": "object", "description": "文献元数据"},
                "style": {
                    "type": "string",
                    "enum": ["apa", "mla", "chicago", "ieee", "gbt7714"],
                    "default": "apa",
                },
            },
            "required": ["paper"],
        },
    )

    def call(self, **kwargs: Any) -> ToolResult:
        return self.execute(kwargs)

    def execute(self, args: dict[str, Any]) -> ToolResult:
        import time

        start = time.time()
        try:
            paper_data = args.get("paper", {})
            style = args.get("style", "apa").lower()

            # 兼容 PaperMeta 对象或 dict
            if isinstance(paper_data, PaperMeta):
                paper = paper_data
            else:
                paper = PaperMeta(
                    paper_id=paper_data.get("paper_id", "unknown"),
                    title=paper_data.get("title", ""),
                    authors=paper_data.get("authors", []),
                    abstract=paper_data.get("abstract", ""),
                    publication_date=paper_data.get("publication_date", ""),
                    doi=paper_data.get("doi", ""),
                    citations=paper_data.get("citations", 0),
                    source=paper_data.get("source", ""),
                )

            formatter = self._get_formatter(style)
            citation = formatter(paper)

            return ToolResult(
                tool_name=self.name,
                success=True,
                content=[{"citation": citation, "style": style}],
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"CSL 转换失败: {e}", exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _get_formatter(self, style: str):
        """获取指定格式的格式化函数。"""
        formatters = {
            "apa": self._format_apa,
            "mla": self._format_mla,
            "chicago": self._format_chicago,
            "ieee": self._format_ieee,
            "gbt7714": self._format_gbt7714,
        }
        return formatters.get(style, self._format_apa)

    def _format_authors(self, authors: list[str], style: str = "apa") -> str:
        """格式化作者列表。"""
        if not authors:
            return "Unknown"
        if style == "apa":
            # APA: Last, F. M., & Last, F. M.
            if len(authors) == 1:
                return authors[0]
            if len(authors) <= 20:
                return ", ".join(authors[:-1]) + ", & " + authors[-1]
            return ", ".join(authors[:6]) + ", ... " + authors[-1]
        elif style == "mla":
            # MLA: Last, First, et al.
            if len(authors) <= 3:
                return ", ".join(authors)
            return authors[0] + ", et al."
        elif style == "ieee":
            # IEEE: F. M. Last, F. M. Last
            return ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
        elif style == "gbt7714":
            # GB/T 7714: 主要责任者. 其他责任者
            if len(authors) <= 3:
                return ", ".join(authors)
            return authors[0] + ", 等"
        return ", ".join(authors)

    def _format_apa(self, p: PaperMeta) -> str:
        """APA 格式。"""
        authors = self._format_authors(p.authors, "apa")
        year = p.publication_date[:4] if p.publication_date else "n.d."
        title = p.title or "Untitled"
        doi = f" https://doi.org/{p.doi}" if p.doi else ""
        return f"{authors} ({year}). {title}.{doi}"

    def _format_mla(self, p: PaperMeta) -> str:
        """MLA 格式。"""
        authors = self._format_authors(p.authors, "mla")
        title = p.title or "Untitled"
        year = p.publication_date[:4] if p.publication_date else ""
        return f'{authors}. "{title}" ({year}).'

    def _format_chicago(self, p: PaperMeta) -> str:
        """Chicago 格式。"""
        authors = self._format_authors(p.authors, "apa")
        title = p.title or "Untitled"
        year = p.publication_date[:4] if p.publication_date else "n.d."
        return f'{authors}. "{title}" ({year}).'

    def _format_ieee(self, p: PaperMeta) -> str:
        """IEEE 格式。"""
        authors = self._format_authors(p.authors, "ieee")
        title = p.title or "Untitled"
        year = p.publication_date[:4] if p.publication_date else ""
        return f'{authors}, "{title}" ({year}).'

    def _format_gbt7714(self, p: PaperMeta) -> str:
        """GB/T 7714 格式（中国国标）。"""
        authors = self._format_authors(p.authors, "gbt7714")
        title = p.title or "Untitled"
        year = p.publication_date[:4] if p.publication_date else ""
        doi = f" DOI: {p.doi}" if p.doi else ""
        return f"{authors}. {title} ({year}).{doi}"
