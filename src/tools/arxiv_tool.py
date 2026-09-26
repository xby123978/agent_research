"""arXiv 文献检索工具。

对应架构 3.7 信息检索类：arXiv 检索，官方 API 封装。
使用 arxiv 库实现，带指数退避重试。
"""

from __future__ import annotations

import time
from typing import Any

from ..common.data_models import PaperMeta
from ..common.logger import get_logger
from .base import BaseTool, ToolResult, ToolSchema

logger = get_logger(__name__)


class ArxivTool(BaseTool):
    """arXiv 论文检索工具。"""

    schema = ToolSchema(
        name="search_arxiv",
        description="检索 arXiv 论文库，返回相关论文元数据列表",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词"},
                "max_results": {"type": "integer", "description": "最大返回数", "default": 10},
                "sort_by": {
                    "type": "string",
                    "description": "排序方式：relevance/submittedDate",
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
        per_request_timeout: float = 30.0,
    ) -> None:
        self.max_retries = max_retries
        self.backoff = backoff
        # arxiv.Client 每页请求超时（秒），避免 urllib 默认行为挂死
        self.per_request_timeout = per_request_timeout

    def call(self, query: str, max_results: int = 10, sort_by: str = "relevance", **_: Any) -> ToolResult:
        import arxiv

        start = time.time()
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # arxiv.Client(num_retries=0) 关闭其内部重试，由我们统一控制
                # 超时通过 page_size 与延迟控制；arxiv 库未直接暴露 timeout，
                # 用 socket 默认超时兜底（最坏 30s）
                import socket

                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(self.per_request_timeout)
                try:
                    client = arxiv.Client(num_retries=0)
                    results = list(
                        client.results(
                            arxiv.Search(
                                query=query,
                                max_results=max_results,
                                sort_by=sort_criterion,
                            )
                        )
                    )
                finally:
                    socket.setdefaulttimeout(old_timeout)
                papers = [self._to_paper(r) for r in results]
                logger.info(f"arXiv 检索完成: query={query} 命中 {len(papers)} 篇")
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    content=[p.model_dump() for p in papers],
                    duration_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                last_error = e
                wait = self.backoff ** attempt
                logger.warning(f"arXiv 检索第 {attempt} 次失败: {e}，{wait:.1f}s 后重试")
                time.sleep(wait)

        return ToolResult(
            tool_name=self.name,
            success=False,
            error=f"arXiv 检索失败: {last_error}",
            duration_ms=(time.time() - start) * 1000,
        )

    @staticmethod
    def _to_paper(r: Any) -> PaperMeta:
        """将 arxiv.Result 转为 PaperMeta。"""
        # 提取 arxiv id（去版本号）
        arxiv_id = str(getattr(r, "entry_id", "")).split("/abs/")[-1]
        if "v" in arxiv_id and arxiv_id.rsplit("v", 1)[0]:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]
        return PaperMeta(
            paper_id=arxiv_id,
            title=getattr(r, "title", "").replace("\n", " ").strip(),
            authors=[str(a) for a in getattr(r, "authors", [])],
            abstract=getattr(r, "summary", "").strip(),
            arxiv_id=arxiv_id,
            publication_date=str(getattr(r, "published", "") or "")[:10],
            pdf_url=getattr(r, "pdf_url", "") or "",
            tags=[
                (t.term if hasattr(t, "term") else str(t))
                for t in (getattr(r, "categories", []) or [])
            ],
            source="arxiv",
        )
