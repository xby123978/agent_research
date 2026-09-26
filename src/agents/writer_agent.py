"""Writer Agent - 研究报告撰写员。

职责：基于文献与知识点，撰写结构化 Markdown 研究报告。
对应架构 3.2.1 状态机 pending_output → completed 阶段。
阶段二新增：结合知识点生成更结构化的报告。
"""

from __future__ import annotations

from ..common.data_models import KnowledgePoint, PaperMeta, TaskStatus
from ..common.logger import get_logger
from ..common.quality_gates import check_writer
from ..orchestrator.shared_board import SharedBoard
from .base import BaseAgent

logger = get_logger(__name__)

REPORT_PROMPT = """你是一位学术研究助手。根据以下文献与知识点，围绕主题「{topic}」撰写一份结构化研究报告。

文献列表（共 {n_papers} 篇）：
{papers}

知识点（共 {n_kps} 个）：
{kps}

请输出 Markdown 格式，包含以下章节：
1. 研究主题概述（背景与意义）
2. 核心知识点（按重要性排序，逐点说明定义与作用）
3. 关键文献综述（标注标题、作者、引用数）
4. 研究脉络与趋势
5. 研究局限与建议

字数控制在 1200 字以内，引用必须使用文献列表中的真实标题。
"""


class WriterAgent(BaseAgent):
    name = "WriterAgent"

    def execute(self, board: SharedBoard) -> SharedBoard:
        task = board.task
        # 状态保护：仅 pending_output 可进入此节点
        current = board.get_task_state()
        if current != TaskStatus.PENDING_OUTPUT:
            logger.info(
                f"[{self.name}] 当前状态 {current.value} 非本节点职责，跳过执行"
            )
            if current == TaskStatus.PENDING_PLANNING:
                self.transition_to(board, TaskStatus.PENDING_SEARCH)
            return board

        papers: list[PaperMeta] = board.read("papers", [])
        kps: list[KnowledgePoint] = board.read("knowledge_points", [])

        # 构造文献与知识点文本
        papers_text = "\n".join(
            f"- [{p.source}] {p.title} ({p.publication_date}) [引用={p.citations}] 摘要: {p.abstract[:150]}"
            for p in papers[:15]
        )
        kps_text = "\n".join(
            f"- [{kp.category}] {kp.name} (重要度={kp.importance:.2f}): {kp.definition}"
            for kp in kps
        ) or "（无知识点）"

        # LLM 生成报告
        report = self.llm.chat(
            [
                {"role": "system", "content": "你是学术研究助手，输出 Markdown 研究报告。"},
                {"role": "user", "content": REPORT_PROMPT.format(
                    topic=task.topic,
                    n_papers=len(papers),
                    papers=papers_text or "（无文献）",
                    n_kps=len(kps),
                    kps=kps_text,
                )},
            ],
            temperature=0.4,
        )

        board.write("final_report", report)

        # 事实校验与幻觉防控（架构 6.4）
        try:
            from ..common.fact_check import verify_report

            fact_result = verify_report(report, papers)
            board.log_stage(self.name, fact_result.summary())
            # 记录 JEV 评分到黑板
            jev_scores = board.read("jev_scores", {}) or {}
            jev_scores["writer"] = fact_result.consistency_score
            board.write("jev_scores", jev_scores)
            # 需要重写时附加修改建议到报告末尾（不阻断交付）
            if fact_result.needs_rewrite and fact_result.suggestions:
                report += "\n\n---\n**事实校验建议**（需人工复核）:\n"
                report += "\n".join(f"- {s}" for s in fact_result.suggestions)
                board.write("final_report", report)
                board.log_stage(
                    self.name,
                    f"事实校验未通过(一致性={fact_result.consistency_score:.2f})，已附加修改建议",
                    level="WARN",
                )
        except Exception as e:
            logger.warning(f"事实校验失败（不影响主流程）: {e}")

        # 沉淀到长期记忆（写失败不影响主流程）
        try:
            self.memory.add(
                f"研究报告: {task.topic}\n{report[:500]}",
                metadata={"task_id": task.task_id, "type": "report"},
                layer="long_term",
            )
        except Exception as e:
            logger.warning(f"报告沉淀失败（不影响主流程）: {e}")

        # 质量门限校验（架构 6.1）：未通过标记但仍完成（避免阻断交付）
        gate = check_writer(report, papers, kps)
        task.quality_score = round(gate.score, 2)
        # 记录 Writer JEV 评分到黑板
        jev_scores = board.read("jev_scores", {}) or {}
        jev_scores["writer_gate"] = gate.jev_score
        board.write("jev_scores", jev_scores)
        if not gate.passed:
            board.log_stage(self.name, f"质量门限未通过: {gate.reason}", level="WARN")
        else:
            board.log_stage(self.name, f"质量门限通过: {gate.reason}")

        board.log_stage(self.name, f"报告生成完成，质量评分={task.quality_score}")
        # 转移到完成状态
        self.transition_to(board, TaskStatus.COMPLETED)
        return board
