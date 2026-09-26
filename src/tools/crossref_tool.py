"""CrossRef 文献元数据查询工具。

CrossRef 是 DOI 注册机构，提供权威文献元数据查询。
免费公开 API，国内访问稳定。
文档：https://www.crossref.org/documentation/retrieve_metadata/rest_api/
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..common.data_models import PaperMeta
from ..common.logger import get_logger
from .base import BaseTool, ToolResult, ToolSchema

logger = get_logger(__name__)

CROSSREF_URL = "https://api.crossref.org/works"


class CrossRefTool(BaseTool):
    """CrossRef 文献元数据查询工具。"""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_crossref",
            description="通过 CrossRef 检索文献元数据（按标题/作者/DOI）",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词（标题或作者）"},
                    "max_results": {"type": "integer", "description": "最大返回数", "default": 10},
                    "doi": {"type": "string", "description": "直接按 DOI 查询（可选）"},
                },
                "required": ["query"],
            },
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        start = time.time()
        try:
            if "doi" in params and params["doi"]:
                return self._lookup_doi(params["doi"], start)
            return self._search(params.get("query", ""), int(params.get("max_results", 10)), start)
        except Exception as e:
            # 注意：loguru 把 {xxx} 当占位符，异常含 {} 时会 KeyError，用拼接避免
            logger.error("CrossRef 检索失败: " + str(e), exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def call(self, **kwargs: Any) -> ToolResult:
        """BaseTool 抽象方法实现，转发到 execute。"""
        return self.execute(kwargs)

    def _search(self, query: str, max_results: int, start: float) -> ToolResult:
        headers = {"User-Agent": "research-agent/1.0 (mailto:research-agent@example.com)"}
        params = {"query": query, "rows": max_results, "select": "DOI,title,author,published-print,published-online,is-referenced-by-count,container-title,abstract"}
        with httpx.Client(timeout=20) as client:
            resp = client.get(CROSSREF_URL, params=params, headers=headers)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        papers = [self._to_paper(it) for it in items]
        papers = [p for p in papers if p.title]
        logger.info(f"CrossRef 命中 {len(papers)} 篇 用时 {(time.time()-start):.2f}s")
        return ToolResult(
            tool_name=self.name,
            success=True,
            content=[p.model_dump() for p in papers],
            duration_ms=(time.time() - start) * 1000,
        )

    def _lookup_doi(self, doi: str, start: float) -> ToolResult:
        headers = {"User-Agent": "research-agent/1.0 (mailto:research-agent@example.com)"}
        with httpx.Client(timeout=20) as client:
            resp = client.get(f"{CROSSREF_URL}/{doi}", headers=headers)
        resp.raise_for_status()
        item = resp.json().get("message", {})
        paper = self._to_paper(item)
        return ToolResult(
            tool_name=self.name,
            success=True,
            content=[paper.model_dump()],
            duration_ms=(time.time() - start) * 1000,
        )

    @staticmethod
    def _to_paper(it: dict) -> PaperMeta:
        import uuid

        authors = []
        for a in it.get("author", []):
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(name)
        date_parts = (
            it.get("published-print", {}).get("date-parts", [[]])
            or it.get("published-online", {}).get("date-parts", [[]])
        )
        year = date_parts[0][0] if date_parts and date_parts[0] else ""
        return PaperMeta(
            paper_id=f"crossref_{uuid.uuid4().hex[:8]}",
            title=(it.get("title") or [""])[0] if it.get("title") else "",
            authors=authors,
            abstract=(it.get("abstract") or "")[:1000],
            doi=it.get("DOI", ""),
            publication_date=str(year),
            venue=(it.get("container-title") or [""])[0] if it.get("container-title") else "",
            citations=int(it.get("is-referenced-by-count", 0)),
            source="crossref",
        )
