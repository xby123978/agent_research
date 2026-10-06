"""文献综述 Skill。

对应架构 3.4.1。整合多源检索、去重、相关性筛选、质量评分流程。
输入：主题 + 深度 + 最大文献数
输出：去重排序后的文献列表 + 综述摘要
"""

from __future__ import annotations

from typing import Any

from ..common.data_models import PaperMeta
from ..common.logger import get_logger
from ..mcp_layer.client import MCPClient, get_mcp_client
from .base import Skill

logger = get_logger(__name__)


class LiteratureReviewSkill(Skill):
    name = "literature_review"
    description = "多源文献检索、去重、相关性筛选与综述生成"

    def __init__(self, mcp: MCPClient | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mcp = mcp or get_mcp_client()

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        topic: str = inputs["topic"]
        max_papers: int = int(inputs.get("max_papers", 20))
        queries: list[str] = inputs.get("queries", [topic])

        # 1. 多源检索
        all_papers: list[PaperMeta] = []
        for q in queries[:2]:  # 限制 2 个 query 避免过多请求
            result = self.mcp.call_tool("search_openalex", {"query": q, "max_results": 10})
            if result.success:
                for p_dict in result.content:
                    try:
                        all_papers.append(PaperMeta(**p_dict))
                    except Exception as e:
                        logger.debug(f"文献解析跳过: {e}")

        # 2. 去重
        seen_titles: set[str] = set()
        deduped = [
            p for p in all_papers
            if p.title and not (p.title in seen_titles or seen_titles.add(p.title))
        ]

        # 3. 相关性筛选：按引用数 + 标题相关性排序
        topic_lower = topic.lower()
        for p in deduped:
            title_match = sum(1 for w in topic_lower.split() if w in p.title.lower())
            p.relevance_score = min(1.0, p.citations / 1000 + title_match * 0.1)
        deduped.sort(key=lambda p: (p.relevance_score, p.citations), reverse=True)

        selected = deduped[:max_papers]

        # 4. 生成综述摘要
        summary = self._generate_summary(topic, selected)

        # 质量评分：基于命中率与文献数
        score = min(1.0, len(selected) / 10 * 0.5 + (1.0 if selected else 0.0) * 0.5)

        return {
            "papers": [p.model_dump() for p in selected],
            "paper_count": len(selected),
            "summary": summary,
            "quality_score": round(score, 2),
        }

    def _generate_summary(self, topic: str, papers: list[PaperMeta]) -> str:
        if not papers:
            return f"主题「{topic}」未检索到文献。"
        papers_text = "\n".join(
            f"- {p.title} (引用={p.citations}, 年份={p.publication_date})"
            for p in papers[:10]
        )
        try:
            return self.llm.chat(
                [
                    {"role": "system", "content": "你是文献综述助手，输出 200 字内的简要综述。"},
                    {"role": "user", "content": f"主题: {topic}\n文献:\n{papers_text}\n请输出简短综述。"},
                ],
                temperature=0.3,
            )
        except Exception as e:
            logger.warning(f"综述生成失败: {e}")
            return f"主题「{topic}」共检索到 {len(papers)} 篇文献。"
