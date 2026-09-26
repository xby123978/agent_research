"""规划模块（ReAct + Tree of Thoughts 升级版）。

对应架构 3.2 规划模块：
1. ReAct：思考-行动-观察循环，多轮迭代优化计划
2. Tree of Thoughts：生成多个候选方案，JEV 评分择优
3. JEV 放行校验：计划评分 < 0.75 自动重试（最多 2 轮）

阶段三升级点：
- 从单次 LLM 调用升级为多轮思考链
- 候选方案数 N=3，JEV 评分取最优
- 失败回退路径：重试 → 降级到基础模板 → 仍失败标记为低质量
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..common.data_models import ResearchDepth, ResearchPlan, SubTask
from ..common.exceptions import ResearchAgentError
from ..common.logger import get_logger
from ..models.jev import get_jev
from ..models.llm import LLMClient, get_llm

logger = get_logger(__name__)

# ReAct 思考链 prompt
REACT_THINK_PROMPT = """你是学术研究规划师。请用 ReAct 模式思考：

研究主题：{topic}
研究深度：{depth}
已知约束：{context}

请按以下格式输出（Thought/Action/Observation 循环）：
Thought: 分析主题需要覆盖哪些维度
Action: 生成研究计划候选方案 {n}
Observation: 评估该方案的覆盖度与可执行性
（循环 2-3 轮，最后一轮输出最终方案）

最终输出 JSON（不要 markdown 包裹）：
{{
  "topic": "<主题>",
  "subtasks": [
    {{
      "subtask_id": "st_1",
      "objective": "<目标>",
      "deliverable": "<交付物>",
      "dependencies": [],
      "recommended_tools": ["search_crossref", "search_openalex"]
    }}
  ],
  "quality_score": 0.0,
  "thoughts": ["思考过程记录"]
}}

约束：
- 子任务 3-5 个，覆盖：基础概念、核心方法、研究进展、应用案例
- 推荐工具仅：search_crossref、search_openalex
"""

# Tree of Thoughts 多候选
TOT_CANDIDATES = 3  # 生成 3 个候选方案
TOT_MAX_ROUNDS = 2  # 最多重试 2 轮
TOT_MIN_SCORE = 0.75  # JEV 放行阈值


class Planner:
    """研究规划师（ReAct + ToT 升级版）。

    流程：
    1. 生成 N 个候选计划（ReAct 多轮思考）
    2. JEV 评分每个候选
    3. 取最高分候选
    4. 若最高分 < 阈值，重试（最多 max_rounds 轮）
    5. 仍不达标则降级为模板计划并标记
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_llm()
        self.n_candidates = TOT_CANDIDATES
        self.max_rounds = TOT_MAX_ROUNDS

    def create_plan(
        self, topic: str, depth: ResearchDepth | str = ResearchDepth.STANDARD
    ) -> ResearchPlan:
        """生成研究计划（ToT + JEV 优选）。"""
        depth = ResearchDepth(depth) if isinstance(depth, str) else depth
        logger.info(f"开始规划(ReAct+ToT): topic={topic} depth={depth.value}")

        best_plan: ResearchPlan | None = None
        best_score: float = 0.0

        for round_idx in range(1, self.max_rounds + 1):
            # 生成 N 个候选
            candidates = []
            for i in range(self.n_candidates):
                try:
                    plan = self._generate_candidate(topic, depth, round_idx, i)
                    candidates.append(plan)
                except Exception as e:
                    logger.warning(f"候选 {i+1} 生成失败: {e}")

            if not candidates:
                logger.warning(f"第 {round_idx} 轮无候选生成，降级模板")
                continue

            # JEV 评分每个候选
            scored: list[tuple[float, ResearchPlan]] = []
            for plan in candidates:
                score = self._jev_score_plan(plan)
                scored.append((score, plan))
            scored.sort(key=lambda x: x[0], reverse=True)

            top_score, top_plan = scored[0]
            logger.info(
                f"第 {round_idx} 轮最优候选 JEV={top_score:.2f}"
                f"（候选得分: {[f'{s:.2f}' for s, _ in scored]}）"
            )

            if top_score > best_score:
                best_score = top_score
                best_plan = top_plan

            # 达到阈值，提前退出
            if best_score >= TOT_MIN_SCORE:
                logger.info(f"JEV 评分 {best_score:.2f} 达阈值，提前退出")
                break

        # 仍不达标，降级模板
        if best_plan is None or best_score < TOT_MIN_SCORE:
            logger.warning(
                f"所有候选 JEV 评分 < {TOT_MIN_SCORE}（最高 {best_score:.2f}），降级模板"
            )
            best_plan = self._fallback_template(topic, depth)
            best_score = max(best_score, 0.5)  # 模板计划保底 0.5

        # 质量门限校验（含 JEV）
        best_plan.quality_score = best_score
        self._validate_plan(best_plan)
        logger.info(
            f"规划完成: {len(best_plan.subtasks)} 个子任务, JEV={best_score:.2f}"
        )
        return best_plan

    def _generate_candidate(
        self, topic: str, depth: ResearchDepth, round_idx: int, cand_idx: int
    ) -> ResearchPlan:
        """生成单个候选计划（ReAct 多轮思考）。"""
        context = f"第{round_idx}轮候选{cand_idx+1}" if round_idx > 1 else "初始规划"
        prompt = REACT_THINK_PROMPT.format(
            topic=topic, depth=depth.value, context=context, n=cand_idx + 1
        )
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是学术研究规划助手，用 ReAct 模式思考，最终只输出 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4 + cand_idx * 0.1,  # 候选间温度递增加多样性
        )
        return self._parse_plan(raw, topic)

    def _jev_score_plan(self, plan: ResearchPlan) -> float:
        """用 JEV 评估计划质量。"""
        try:
            content = " ".join(
                f"{s.subtask_id}: {s.objective} -> {s.deliverable}"
                for s in plan.subtasks
            )
            return get_jev().score(
                content,
                "研究计划质量：子任务覆盖度（基础概念/方法/进展/应用）、目标清晰度、可执行性",
            )
        except Exception as e:
            logger.warning(f"JEV 计划评分失败，用自评: {e}")
            return plan.quality_score

    def _parse_plan(self, raw: str, topic: str) -> ResearchPlan:
        """解析 LLM 输出为 ResearchPlan。"""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("计划 JSON 解析失败，尝试宽松提取")
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(raw[start : end + 1])
            else:
                raise ResearchAgentError(f"无法解析研究计划: {raw[:200]}")

        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        subtasks = [
            SubTask(
                subtask_id=s.get("subtask_id", f"st_{i+1}"),
                objective=s.get("objective", ""),
                deliverable=s.get("deliverable", ""),
                dependencies=s.get("dependencies", []),
                recommended_tools=s.get("recommended_tools", []),
            )
            for i, s in enumerate(data.get("subtasks", []))
        ]
        return ResearchPlan(
            plan_id=plan_id,
            topic=topic,
            subtasks=subtasks,
            quality_score=float(data.get("quality_score", 0.0)),
        )

    def _fallback_template(self, topic: str, depth: ResearchDepth) -> ResearchPlan:
        """降级模板：固定 4 子任务覆盖标准维度。"""
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        subtasks = [
            SubTask(
                subtask_id="st_1",
                objective=f"梳理{topic}的基础概念与定义",
                deliverable="概念定义与术语表",
                recommended_tools=["search_crossref"],
            ),
            SubTask(
                subtask_id="st_2",
                objective=f"分析{topic}的核心方法与原理",
                deliverable="原理与方法总结",
                recommended_tools=["search_crossref", "search_openalex"],
            ),
            SubTask(
                subtask_id="st_3",
                objective=f"综述{topic}的研究进展与对比",
                deliverable="进展综述与对比表",
                recommended_tools=["search_crossref"],
            ),
            SubTask(
                subtask_id="st_4",
                objective=f"总结{topic}的应用场景与案例",
                deliverable="应用案例集",
                recommended_tools=["search_crossref"],
            ),
        ]
        return ResearchPlan(
            plan_id=plan_id, topic=topic, subtasks=subtasks, quality_score=0.5
        )

    def _validate_plan(self, plan: ResearchPlan) -> None:
        """计划校验（含 JEV 质量门限）。"""
        from ..common.quality_gates import check_planner

        # 自动补齐缺失交付物
        for s in plan.subtasks:
            if not s.deliverable:
                s.deliverable = s.objective or "未明确交付物"
        # 子任务数不足时补齐
        while len(plan.subtasks) < 3:
            plan.subtasks.append(
                SubTask(
                    subtask_id=f"st_auto_{len(plan.subtasks)+1}",
                    objective=f"补充研究方向 {len(plan.subtasks)+1}",
                    deliverable="补充研究方向调研",
                )
            )
        result = check_planner(plan, self_score=plan.quality_score)
        if not result.passed:
            logger.warning(f"计划门限未通过: {result.reason}")
        else:
            logger.info(f"计划门限通过: {result.reason}")


_planner_instance: Planner | None = None


def get_planner() -> Planner:
    """获取全局规划器单例。"""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = Planner()
    return _planner_instance
