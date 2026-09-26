"""四 Agent 协作状态机。

对应架构 3.2.1 状态转移图：
pending_planning → pending_search → pending_filter → pending_reading
→ pending_organizing → pending_output → completed / failed

状态机驱动 Agent 执行顺序，并对非法转移做校验。
降级方案：纯 Python 实现，不依赖 LangGraph（避免重型依赖）。
"""

from __future__ import annotations

from ..common.data_models import TaskStatus
from ..common.logger import get_logger

logger = get_logger(__name__)


# 合法状态转移表（from → allowed next states）
# 正向流转 + 回退路径（架构 3.1.2 异常升级 + 3.2.2 回退机制）
# 回退：search/filter/reading/organizing 均可回退到 planning，由 Planner 重新规划
_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING_PLANNING: {TaskStatus.PENDING_SEARCH, TaskStatus.FAILED},
    TaskStatus.PENDING_SEARCH: {
        TaskStatus.PENDING_FILTER,
        TaskStatus.PENDING_READING,
        TaskStatus.PENDING_ORGANIZING,
        TaskStatus.PENDING_PLANNING,  # 回退：重新规划
        TaskStatus.FAILED,
    },
    TaskStatus.PENDING_FILTER: {
        TaskStatus.PENDING_READING,
        TaskStatus.PENDING_ORGANIZING,
        TaskStatus.PENDING_SEARCH,     # 回退：重检索
        TaskStatus.PENDING_PLANNING,  # 回退：重规划
        TaskStatus.FAILED,
    },
    TaskStatus.PENDING_READING: {
        TaskStatus.PENDING_ORGANIZING,
        TaskStatus.PENDING_OUTPUT,
        TaskStatus.PENDING_SEARCH,     # 回退：补充检索
        TaskStatus.PENDING_PLANNING,
        TaskStatus.FAILED,
    },
    TaskStatus.PENDING_ORGANIZING: {
        TaskStatus.PENDING_OUTPUT,
        TaskStatus.PENDING_READING,    # 回退：补充精读
        TaskStatus.PENDING_PLANNING,
        TaskStatus.FAILED,
    },
    TaskStatus.PENDING_OUTPUT: {TaskStatus.COMPLETED, TaskStatus.PENDING_ORGANIZING, TaskStatus.FAILED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
}

# 每个 Agent 负责的进入态（状态 → 处理 Agent 名）
_STATE_AGENT: dict[TaskStatus, str] = {
    TaskStatus.PENDING_PLANNING: "PlannerAgent",
    TaskStatus.PENDING_SEARCH: "CollectorAgent",
    TaskStatus.PENDING_FILTER: "CollectorAgent",
    TaskStatus.PENDING_READING: "EngineerAgent",
    TaskStatus.PENDING_ORGANIZING: "EngineerAgent",
    TaskStatus.PENDING_OUTPUT: "WriterAgent",
}


class StateMachine:
    """研究任务状态机。"""

    @staticmethod
    def next_state(current: TaskStatus, target: TaskStatus) -> TaskStatus:
        """校验状态转移是否合法，合法则返回 target，否则抛 ValueError。"""
        allowed = _TRANSITIONS.get(current, set())
        if target not in allowed:
            raise ValueError(
                f"非法状态转移: {current.value} → {target.value}（允许: {[s.value for s in allowed]}）"
            )
        return target

    @staticmethod
    def can_transition(current: TaskStatus, target: TaskStatus) -> bool:
        return target in _TRANSITIONS.get(current, set())

    @staticmethod
    def agent_for_state(state: TaskStatus) -> str:
        return _STATE_AGENT.get(state, "Unknown")

    @staticmethod
    def is_terminal(state: TaskStatus) -> bool:
        return state in (TaskStatus.COMPLETED, TaskStatus.FAILED)
