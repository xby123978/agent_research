"""引用管理 Skill。

对应架构 3.4.3。管理文献引用元数据，支持多种引用风格格式化。
流程：元数据提取 → 格式校验 → 标准转换 → 参考文献列表生成
"""

from __future__ import annotations

from typing import Any

from ..common.data_models import PaperMeta
from ..common.logger import get_logger
from .base import Skill

logger = get_logger(__name__)


class CitationManagerSkill(Skill):
    name = "citation_manager"
    description = "引用元数据管理与多风格格式化（APA/IEEE/Chicago）"
    quality_threshold = 0.6

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        papers: list[PaperMeta] = [
            PaperMeta(**p) if isinstance(p, dict) else p
            for p in inputs.get("papers", [])
        ]
        style: str = inputs.get("style", "apa")

        citations: list[dict] = []
        bibliography: list[str] = []
        for idx, p in enumerate(papers, 1):
            citation = {
                "index": idx,
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "year": p.publication_date[:4] if p.publication_date else "",
                "venue": p.venue,
                "doi": p.doi,
                "citations": p.citations,
            }
            citations.append(citation)
            bibliography.append(self._format(citation, style))

        # 质量评分：每条引用 +0.2，最多 1.0
        score = min(1.0, len(citations) * 0.2)
        return {
            "citations": citations,
            "bibliography": bibliography,
            "style": style,
            "citation_count": len(citations),
            "quality_score": round(score, 2),
        }

    @staticmethod
    def _format(c: dict, style: str) -> str:
        """格式化单条引用。简化实现，覆盖 APA/IEEE/Chicago 三种风格。"""
        authors = c.get("authors", [])
        year = c.get("year", "")
        title = c.get("title", "")
        venue = c.get("venue", "")
        doi = c.get("doi", "")

        # 作者列表处理
        if authors:
            if style == "apa":
                # APA: Author, A. A., & Author, B. B. (Year). Title. Venue. doi
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += ", et al."
                ref = f"{author_str} ({year}). {title}. {venue}."
                if doi:
                    ref += f" https://doi.org/{doi}"
            elif style == "ieee":
                # IEEE: [n] A. Author, B. Author, "Title," Venue, Year.
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += ", et al."
                ref = f"[{c['index']}] {author_str}, \"{title},\" {venue}, {year}."
                if doi:
                    ref += f" doi: {doi}"
            else:  # chicago / default
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += ", et al."
                ref = f"{author_str}. {year}. \"{title}.\" {venue}."
                if doi:
                    ref += f" doi:{doi}"
        else:
            ref = f"[{c['index']}] {title}. {year}. {venue}."
        return ref
