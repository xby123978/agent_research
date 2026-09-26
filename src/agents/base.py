"""Agent 基类。

对应架构 3.2 多 Agent 协作层。每个 Agent 有单一职责，
通过共享黑板读写中间产物，由状态机驱动执行顺序。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..common.data_models import ResearchTask, TaskStatus
from ..common.logger import get_logger
from ..common.utils import stage_timeout
from ..memory.manager import MemoryManager, get_memory
from ..models.llm import LLMClient, get_llm
from ..orchestrator.shared_board import SharedBoard
from ..orchestrator.state_machine import StateMachine

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Agent 抽象基类。子类实现 execute() 完成单一职责。"""

    name: str = "BaseAgent"

    def __init__(
        self,
        llm: LLMClient | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.memory = memory or get_memory()

    @abstractmethod
    def execute(self, board: SharedBoard) -> SharedBoard:
        """执行 Agent 职责，读写共享黑板。子类必须实现。"""
        ...

    def transition_to(self, board: SharedBoard, target: TaskStatus) -> None:
        """校验并执行状态转移。"""
        current = board.get_task_state()
        new_state = StateMachine.next_state(current, target)
        board.update_task_state(new_state)

    def safe_execute(self, board: SharedBoard, timeout: float = 90, retries: int = 1) -> SharedBoard:
        """带超时与异常隔离的执行入口。

        架构 6.3 任务级容错：失败先重试，仍失败则升级给 Planner 决策。
        - 首次失败：原地重试一次（retries=1）
        - 重试仍失败：回退到上一阶段由 Planner 重新规划，而非直接 FAILED
        """
        task = board.task
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                board = self._run_with_timeout(timeout, board)
                logger.info(f"[{self.name}] task={task.task_id} 执行完成 state={board.get_task_state().value}")
                return board
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[{self.name}] task={task.task_id} 第 {attempt+1}/{retries+1} 次执行失败: {e}",
                    exc_info=True,
                )
                if attempt < retries:
                    continue
                # 重试耗尽，尝试回退到上一阶段由 Planner 处理（架构 3.1.2 异常升级）
                if self._try_escalate(board):
                    logger.info(f"[{self.name}] task={task.task_id} 已回退到规划阶段，由 Planner 重新决策")
                    return board
                # 无法回退则标记 FAILED
                logger.error(f"[{self.name}] task={task.task_id} 升级失败，终止任务")
                board.log_stage(self.name, f"失败: {last_err}", level="ERROR")
                board.update_task_state(TaskStatus.FAILED)
                return board
        return board  # 不可达

    def _try_escalate(self, board: SharedBoard) -> bool:
        """异常升级：回退到上一阶段（架构 3.1.2）。

        注意：LangGraph 固定顺序执行不支持节点回头，回退会导致
        后续节点状态错乱。MVP 阶段关闭回退，失败直接走 FAILED，
        由编排器层重试整任务。阶段三切到支持循环的图后再启用。
        """
        return False

    def _run_with_timeout(self, timeout: float, board: SharedBoard) -> SharedBoard:
        from ..common.utils import run_with_timeout
        return run_with_timeout(self.execute, timeout, board)
