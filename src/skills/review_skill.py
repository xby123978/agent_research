"""研究复盘 Skill（架构 3.5.2 第 5 个 Skill）。

职责：对整个研究过程做复盘分析，
识别方法缺陷、质量瓶颈、改进点，沉淀经验到长期记忆。
"""

from __future__ import annotations

from typing import Any

from ..common.data_models import ResearchTask, TaskStatus
from ..common.logger import get_logger
from ..models.jev import get_jev
from .base import Skill

logger = get_logger(__name__)

REVIEW_PROMPT = """你是研究复盘专家。对以下研究任务做复盘分析。

任务主题：{topic}
任务深度：{depth}
最终状态：{status}
质量评分：{quality_score}
文献数：{n_papers}
知识点数：{n_kps}
阶段日志：
{stage_logs}

请输出 JSON（不要 markdown 包裹）：
{{
  "success_summary": "成功经验总结",
  "method_issues": ["方法缺陷1", "方法缺陷2"],
  "quality_bottlenecks": ["质量瓶颈1", "质量瓶颈2"],
  "improvements": ["改进建议1", "改进建议2", "改进建议3"],
  "reusable_insights": ["可沉淀经验1", "可沉淀经验2"],
  "overall_rating": float  // 0-1 综合评分
}}

约束：
- 方法缺陷识别 2-3 个
- 改进建议 3-5 个
- 可沉淀经验 1-3 个
"""


class ReviewSkill(Skill):
    """研究复盘 Skill：过程反思与经验沉淀。"""

    name = "review"
    version = "1.0"
    description = "研究过程复盘：识别方法缺陷、质量瓶颈、改进点，沉淀经验"
    quality_threshold = 0.7

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        task: ResearchTask | None = inputs.get("task")
        stage_logs: list = inputs.get("stage_logs", [])

        if not task:
            return {
                "skill": self.name,
                "error": "缺少 task 输入",
                "quality_score": 0.0,
            }

        logs_text = "\n".join(
            f"- [{log.get('stage', '?')}] {log.get('message', '')}"
            for log in stage_logs[-20:]
        )

        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是研究复盘专家，输出 JSON 格式复盘报告。",
                },
                {
                    "role": "user",
                    "content": REVIEW_PROMPT.format(
                        topic=task.topic,
                        depth=task.depth.value,
                        status=task.status.value,
                        quality_score=task.quality_score,
                        n_papers=len(task.papers),
                        n_kps=len(task.knowledge_points),
                        stage_logs=logs_text or "（无）",
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

        # JEV 评分复盘质量
        try:
            content = str(data.get("success_summary", "")) + " " + str(
                data.get("improvements", [])
            )
            jev_score = get_jev().score(
                content[:1500],
                "复盘质量：问题识别深度、改进建议可执行性、经验沉淀价值",
            )
        except Exception as e:
            logger.warning(f"JEV 复盘评分失败: {e}")
            jev_score = 0.7

        data["quality_score"] = jev_score
        data["skill"] = self.name

        # 经验沉淀到长期记忆
        try:
            for insight in data.get("reusable_insights", [])[:3]:
                self.memory.add(
                    f"复盘经验: {insight}",
                    metadata={
                        "task_id": task.task_id,
                        "type": "review_insight",
                        "topic": task.topic,
                    },
                    layer="long_term",
                )
        except Exception as e:
            logger.warning(f"经验沉淀失败（不影响主流程）: {e}")

        return data
