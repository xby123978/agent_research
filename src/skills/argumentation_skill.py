"""论证构建 Skill（架构 3.5.2 第 4 个 Skill）。

职责：基于文献与知识点，构建逻辑论证链，
包含论点-论据-论证三要素，支持正反对比论证。
"""

from __future__ import annotations

from typing import Any

from ..common.data_models import KnowledgePoint, PaperMeta
from ..common.logger import get_logger
from ..models.jev import get_jev
from .base import Skill

logger = get_logger(__name__)

ARGUMENTATION_PROMPT = """你是学术论证专家。基于以下文献与知识点，围绕主题「{topic}」构建论证链。

文献（共 {n_papers} 篇）：
{papers}

知识点（共 {n_kps} 个）：
{kps}

请输出 JSON（不要 markdown 包裹）：
{{
  "main_argument": "核心论点（一句话）",
  "sub_arguments": [
    {{
      "point": "子论点",
      "evidence": "支撑论据（引用文献）",
      "reasoning": "论证过程",
      "source_papers": ["paper_id1", "paper_id2"]
    }}
  ],
  "counter_arguments": [
    {{
      "counter_point": "反方论点",
      "rebuttal": "驳斥论证"
    }}
  ],
  "conclusion": "结论（含局限与展望）"
}}

约束：
- 子论点 3-5 个
- 每个论据必须引用至少 1 篇文献
- 包含至少 1 个反方论点与驳斥
"""


class ArgumentationSkill(Skill):
    """论证构建 Skill：构建论点-论据-论证链。"""

    name = "argumentation"
    version = "1.0"
    description = "基于文献与知识点构建逻辑论证链，含正反对比论证"
    quality_threshold = 0.7

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        topic = inputs.get("topic", "")
        papers: list[PaperMeta] = inputs.get("papers", [])
        kps: list[KnowledgePoint] = inputs.get("knowledge_points", [])

        papers_text = "\n".join(
            f"- {p.title} ({p.publication_date}): {p.abstract[:100]}"
            for p in papers[:10]
        )
        kps_text = "\n".join(
            f"- {kp.name}: {kp.definition}" for kp in kps[:5]
        ) or "（无）"

        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是学术论证专家，输出 JSON 格式论证链。",
                },
                {
                    "role": "user",
                    "content": ARGUMENTATION_PROMPT.format(
                        topic=topic,
                        n_papers=len(papers),
                        papers=papers_text or "（无）",
                        n_kps=len(kps),
                        kps=kps_text,
                    ),
                },
            ],
            temperature=0.4,
        )

        import json
        import re

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[start : end + 1]) if start >= 0 else {}

        # JEV 评分论证质量
        try:
            content = data.get("main_argument", "") + " " + str(
                data.get("sub_arguments", [])
            )
            jev_score = get_jev().score(
                content[:1500], "论证链质量：论点清晰度、论据充分性、逻辑严谨性"
            )
        except Exception as e:
            logger.warning(f"JEV 论证评分失败: {e}")
            jev_score = 0.7

        data["quality_score"] = jev_score
        data["skill"] = self.name
        return data
