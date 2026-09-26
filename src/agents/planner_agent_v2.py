"""Planner Agent - 研究规划师。

职责：根据主题与深度，调用规划模块生成研究计划（3-5 个子任务）。
对应架构 3.2.1 状态机 pending_planning 阶段。
"""

from __future__ import annotations

from ..common.data_models import TaskStatus
from ..common.logger import get_logger
from ..orchestrator.shared_board import SharedBoard
from ..planning.planner import Planner, get_planner
from .base import BaseAgent

logger = get_logger(__name__)


class PlannerAgent(BaseAgent):
    name = "PlannerAgent"

    def __init__(self, planner: Planner | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.planner = planner or get_planner()

    def execute(self, board: SharedBoard) -> SharedBoard:
        task = board.task
        board.log_stage(self.name, f"开始规划主题: {task.topic}")
        # 调用规划模块生成计划
        plan = self.planner.create_plan(task.topic, task.depth)
        board.write("plan", plan)
        # 写入短期记忆
        self.memory.add(
            f"研究计划已生成: {task.topic}，共 {len(plan.subtasks)} 个子任务",
            metadata={"task_id": task.task_id, "plan_id": plan.plan_id},
            layer="short_term",
        )
        board.log_stage(self.name, f"规划完成: {len(plan.subtasks)} 个子任务")
        # 转移到检索阶段
        self.transition_to(board, TaskStatus.PENDING_SEARCH)
        return board
