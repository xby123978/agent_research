"""Semantic Scholar 文献检索工具。

对应架构 3.7 信息检索类：Semantic Scholar 检索，官方 API 封装。
直接调用 REST API，带指数退避重试，支持可选 API Key 提升配额。
"""

from __future__ import annotations

import time
from typing import Any

import requests

from ..common.config import get_env
from ..common.data_models import PaperMeta
from ..common.logger import get_logger
from .base import BaseTool, ToolResult, ToolSchema

logger = get_logger(__name__)

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarTool(BaseTool):
    """Semantic Scholar 论文检索工具。"""

    schema = ToolSchema(
        name="search_semantic_scholar",
        description="检索 Semantic Scholar 论文库，返回含引用数的论文元数据",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词"},
                "max_results": {"type": "integer", "description": "最大返回数", "default": 10},
                "fields": {
                    "type": "string",
                    "description": "返回字段，逗号分隔",
                    "default": "title,authors,abstract,year,venue,citationCount,externalIds",
                },
            },
            "required": ["query"],
        },
    )

    def __init__(self, max_retries: int = 3, backoff: float = 2.0) -> None:
        env = get_env()
        self.api_key = env.get("s2_api_key") or None
        self.max_retries = max_retries
        self.backoff = backoff

    def call(
        self,
        query: str,
        max_results: int = 10,
        fields: str = "title,authors,abstract,year,venue,citationCount,externalIds",
        **_: Any,
    ) -> ToolResult:
        start = time.time()
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": fields,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(S2_SEARCH_URL, params=params, headers=headers, timeout=30)
                # 429 限流时退避
                if resp.status_code == 429:
                    wait = self.backoff ** attempt * 2
                    logger.warning(f"S2 限流，{wait:.1f}s 后重试")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                papers = [self._to_paper(p) for p in data.get("data", [])]
                logger.info(f"S2 检索完成: query={query} 命中 {len(papers)} 篇")
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    content=[p.model_dump() for p in papers],
                    duration_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                last_error = e
                wait = self.backoff ** attempt
                logger.warning(f"S2 检索第 {attempt} 次失败: {e}，{wait:.1f}s 后重试")
                time.sleep(wait)

        return ToolResult(
            tool_name=self.name,
            success=False,
            error=f"Semantic Scholar 检索失败: {last_error}",
            duration_ms=(time.time() - start) * 1000,
        )

    @staticmethod
    def _to_paper(raw: dict) -> PaperMeta:
        ext = raw.get("externalIds") or {}
        return PaperMeta(
            paper_id=raw.get("paperId", ""),
            title=raw.get("title", "") or "",
            authors=[a.get("name", "") for a in raw.get("authors", []) or []],
            abstract=raw.get("abstract", "") or "",
            doi=ext.get("DOI", ""),
            arxiv_id=ext.get("ArXiv", ""),
            publication_date=str(raw.get("year", "") or ""),
            venue=raw.get("venue", "") or "",
            citations=raw.get("citationCount", 0) or 0,
            source="semantic_scholar",
        )
