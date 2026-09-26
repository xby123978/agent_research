"""OpenAlex 学术文献检索工具。

OpenAlex 是开放的学术文献数据库（取代 Microsoft Academic），覆盖 2.5 亿+ 文献，
完全免费、无需 API Key、无配额限制，国内访问稳定（实测 7s 延迟）。

替代 arXiv + Semantic Scholar 作为 MVP 阶段的主检索源。
文档：https://docs.openalex.org/
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..common.data_models import PaperMeta
from ..common.logger import get_logger
from .base import BaseTool, ToolResult, ToolSchema

logger = get_logger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# 礼貌池配置：OpenAlex 建议在邮箱参数，列入「礼貌池」可获得更稳定配额
MAILTO = "research-agent@example.com"


def _abstract_from_inverted_index(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex 摘要采用倒排索引格式（{word: [positions]}），需还原为正文。"""
    if not inverted:
        return ""
    # 构建 (position, word) 列表
    pos_word: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for p in positions:
            pos_word.append((p, word))
    pos_word.sort()
    return " ".join(w for _, w in pos_word)


class OpenAlexTool(BaseTool):
    """OpenAlex 学术文献检索工具。

    特点：
    - 完全免费、无需 API Key
    - 国内访问稳定（实测 5-10s）
    - 覆盖 2.5 亿+文献，含引用数、作者、摘要、DOI、年份
    - 支持 relevance（默认）/ citations 排序
    """

    schema = ToolSchema(
        name="search_openalex",
        description="检索 OpenAlex 学术文献库（2.5亿+文献，含引用数与摘要）",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或主题"},
                "max_results": {"type": "integer", "description": "最大返回数", "default": 10},
                "sort_by": {
                    "type": "string",
                    "description": "排序：relevance（相关性）| citations（引用数）| year（年份）",
                    "default": "relevance",
                },
            },
            "required": ["query"],
        },
    )

    def __init__(
        self,
        max_retries: int = 3,
        backoff: float = 2.0,
        per_request_timeout: float = 10.0,
    ) -> None:
        self.max_retries = max_retries
        self.backoff = backoff
        self.per_request_timeout = per_request_timeout

    def call(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        **_: Any,
    ) -> ToolResult:
        start = time.time()

        # 排序参数映射到 OpenAlex API
        sort_map = {
            "relevance": "relevance_score:desc",
            "citations": "cited_by_count:desc",
            "year": "publication_year:desc",
        }
        sort_param = sort_map.get(sort_by, "relevance_score:desc")

        params = {
            "search": query,
            "per-page": min(max_results, 25),
            "sort": sort_param,
            "mailto": MAILTO,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.per_request_timeout) as client:
                    resp = client.get(OPENALEX_WORKS_URL, params=params)
                if resp.status_code == 429:
                    wait = self.backoff ** attempt * 2
                    logger.warning(f"OpenAlex 限流，{wait:.1f}s 后重试")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                works = data.get("results", [])
                papers = [self._to_paper(w) for w in works]
                logger.info(
                    f"OpenAlex 检索完成: query={query[:30]} 命中 {len(papers)} 篇 "
                    f"(总 {data.get('meta', {}).get('count', 0)})"
                )
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    content=[p.model_dump() for p in papers],
                    duration_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                last_error = e
                wait = self.backoff ** attempt
                logger.warning(
                    f"OpenAlex 检索第 {attempt}/{self.max_retries} 次失败: {e}，{wait:.1f}s 后重试"
                )
                time.sleep(wait)

        return ToolResult(
            tool_name=self.name,
            success=False,
            error=f"OpenAlex 检索失败: {last_error}",
            duration_ms=(time.time() - start) * 1000,
        )

    @staticmethod
    def _to_paper(w: dict) -> PaperMeta:
        """将 OpenAlex work 转为 PaperMeta。"""
        # 提取 paper_id（去掉前缀 https://openalex.org/W123 → W123）
        raw_id = w.get("id", "")
        paper_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""

        # 提取 DOI（去掉 https://doi.org/ 前缀）
        doi_url = w.get("doi") or ""
        doi = doi_url.replace("https://doi.org/", "") if doi_url else ""

        # 作者列表
        authorships = w.get("authorships", []) or []
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in authorships
            if a.get("author")
        ]

        # 摘要从倒排索引还原
        abstract = _abstract_from_inverted_index(w.get("abstract_inverted_index"))

        # 主题
        topic = w.get("primary_topic") or {}
        tags = [topic.get("display_name")] if topic.get("display_name") else []

        # PDF/开放访问 URL
        primary_location = w.get("primary_location") or {}
        source = primary_location.get("source") or {}
        pdf_url = ""
        oa = w.get("open_access") or {}
        if oa.get("oa_url"):
            pdf_url = oa.get("oa_url", "")
        elif primary_location.get("pdf_url"):
            pdf_url = primary_location.get("pdf_url", "")

        venue = source.get("display_name", "") if source else ""

        return PaperMeta(
            paper_id=paper_id,
            title=w.get("title", "") or w.get("display_name", "") or "",
            authors=authors,
            abstract=abstract,
            doi=doi,
            publication_date=str(w.get("publication_date", "") or "")[:10],
            venue=venue,
            citations=int(w.get("cited_by_count", 0) or 0),
            pdf_url=pdf_url,
            tags=tags,
            relevance_score=float(w.get("relevance_score", 0.0) or 0.0),
            source="openalex",
        )
