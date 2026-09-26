"""知识卡片 Skill。

对应架构 3.4.2。从文献中抽取概念，生成结构化知识卡片。
流程：概念抽取 → 原理组织 → 优劣分析 → 关系匹配 → 卡片生成
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from ..common.data_models import KnowledgePoint, PaperMeta
from ..common.logger import get_logger
from .base import Skill

logger = get_logger(__name__)

CARD_PROMPT = """你是学术知识工程师。从以下文献中抽取核心概念并生成知识卡片。

主题: {topic}

文献:
{papers}

请输出 JSON 数组，每个元素：
{{
  "name": "概念名",
  "definition": "定义（1-2句）",
  "category": "method|concept|dataset|benchmark|other",
  "importance": 0.0-1.0,
  "pros": "优点（如有）",
  "cons": "缺点/局限（如有）",
  "source_paper": "来源论文标题"
}}

最多 6 张卡片，只输出 JSON。
"""


class KnowledgeCardSkill(Skill):
    name = "knowledge_card"
    description = "从文献抽取概念并生成结构化知识卡片"
    quality_threshold = 0.6

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        topic: str = inputs["topic"]
        papers: list[PaperMeta] = [
            PaperMeta(**p) if isinstance(p, dict) else p
            for p in inputs.get("papers", [])
        ]
        if not papers:
            return {"cards": [], "card_count": 0, "quality_score": 0.0}

        papers_text = "\n".join(
            f"- {p.title} (引用={p.citations}) 摘要: {p.abstract[:200]}"
            for p in papers[:10]
        )

        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": "你是学术知识工程师，输出 JSON。"},
                    {"role": "user", "content": CARD_PROMPT.format(topic=topic, papers=papers_text)},
                ],
                temperature=0.3,
            )
            cards = self._parse_cards(raw, papers)
        except Exception as e:
            logger.warning(f"知识卡片抽取失败: {e}")
            cards = []

        # 转换为 KnowledgePoint
        title_to_id = {p.title: p.paper_id for p in papers}
        kps: list[KnowledgePoint] = []
        for c in cards:
            src_title = c.get("source_paper", "")
            src_ids = [title_to_id[src_title]] if src_title in title_to_id else []
            kps.append(
                KnowledgePoint(
                    kp_id=f"kp_{uuid.uuid4().hex[:8]}",
                    name=c.get("name", ""),
                    definition=c.get("definition", ""),
                    category=c.get("category", "other"),
                    importance=float(c.get("importance", 0.5)),
                    source_papers=src_ids,
                )
            )

        score = min(1.0, len(kps) / 5)
        return {
            "cards": cards,
            "knowledge_points": [kp.model_dump() for kp in kps],
            "card_count": len(kps),
            "quality_score": round(score, 2),
        }

    @staticmethod
    def _parse_cards(raw: str, papers: list[PaperMeta]) -> list[dict]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group(0))
        except Exception:
            return []
        return items[:6]
