"""四 Agent 编排器（LangGraph 实现）。

对应架构 3.2.1 状态机驱动流程。
使用 LangGraph StateGraph 替代手写顺序循环：
- 节点 = 四个 Agent 的 execute 包装
- 边 = 状态机驱动的顺序流转
- 条件边 = 失败短路到 END

降级方案：LangGraph 不可用时回退到顺序循环实现。
"""

from __future__ import annotations

from typing import Any

from ..agents.base import BaseAgent
from ..agents.collector_agent import CollectorAgent
from ..agents.engineer_agent import EngineerAgent
from ..agents.planner_agent_v2 import PlannerAgent
from ..agents.writer_agent import WriterAgent
from ..common.data_models import ResearchDepth, ResearchTask, TaskStatus
from ..common.logger import get_logger, set_trace_id
from ..memory.manager import MemoryManager, get_memory
from .shared_board import SharedBoard

logger = get_logger(__name__)


# LangGraph State 定义：仅承载共享黑板引用，节点间数据通过黑板传递
# 必须模块级定义，否则 _wrap/_route 函数引用会 NameError
try:
    from typing_extensions import TypedDict

    class _GraphState(TypedDict, total=False):
        board: SharedBoard
except ImportError:
    _GraphState = dict  # type: ignore[assignment,misc]


class Orchestrator:
    """四 Agent 协作编排器（LangGraph 驱动）。

    用 LangGraph StateGraph 替代手写循环：
    - entry → planner → collector → engineer → writer → END
    - 任一节点失败 → 条件边直接到 END（状态已转 FAILED）
    - 每个节点带阶段超时与异常隔离（由 BaseAgent.safe_execute 保证）
    """

    def __init__(
        self,
        planner: PlannerAgent | None = None,
        collector: CollectorAgent | None = None,
        engineer: EngineerAgent | None = None,
        writer: WriterAgent | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self.memory = memory or get_memory()
        self.planner = planner or PlannerAgent(memory=self.memory)
        self.collector = collector or CollectorAgent(memory=self.memory)
        self.engineer = engineer or EngineerAgent(memory=self.memory)
        self.writer = writer or WriterAgent(memory=self.memory)
        # 各 Agent 阶段超时（秒）
        self._timeouts: dict[str, float] = {
            "planner": 60,
            "collector": 150,
            "engineer": 90,
            "writer": 90,
        }
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """构建 LangGraph StateGraph。失败时回退顺序循环。"""
        try:
            from langgraph.graph import END, START, StateGraph

            def _wrap(agent: BaseAgent, timeout_key: str):
                """将 Agent.execute 包装为 LangGraph 节点函数。"""
                def _node(state: _GraphState) -> _GraphState:
                    board: SharedBoard = state["board"]
                    timeout = self._timeouts.get(timeout_key, 90)
                    new_board = agent.safe_execute(board, timeout=timeout)
                    return {"board": new_board}
                return _node

            def _route(state: _GraphState) -> str:
                """条件路由：失败或完成直接到 END，否则继续。"""
                board: SharedBoard = state["board"]
                status = board.get_task_state()
                if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    return END
                return "continue"

            builder = StateGraph(_GraphState)
            builder.add_node("planner", _wrap(self.planner, "planner"))
            builder.add_node("collector", _wrap(self.collector, "collector"))
            builder.add_node("engineer", _wrap(self.engineer, "engineer"))
            builder.add_node("writer", _wrap(self.writer, "writer"))

            # 顺序边 + 失败短路条件边
            builder.add_edge(START, "planner")
            builder.add_conditional_edges(
                "planner", _route, {"continue": "collector", END: END}
            )
            builder.add_conditional_edges(
                "collector", _route, {"continue": "engineer", END: END}
            )
            builder.add_conditional_edges(
                "engineer", _route, {"continue": "writer", END: END}
            )
            builder.add_edge("writer", END)

            graph = builder.compile()
            logger.info("LangGraph StateGraph 构建成功")
            return graph
        except ImportError:
            logger.warning("LangGraph 未安装，回退到顺序循环实现")
            return None
        except Exception as e:
            logger.warning(f"LangGraph 构建失败（{e}），回退到顺序循环实现")
            return None

    def create_task(
        self, topic: str, depth: str | ResearchDepth = ResearchDepth.STANDARD
    ) -> ResearchTask:
        """创建任务并写入短期记忆。"""
        import uuid

        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = ResearchTask(
            task_id=task_id,
            topic=topic,
            depth=ResearchDepth(depth) if isinstance(depth, str) else depth,
            status=TaskStatus.PENDING_PLANNING,
        )
        self.memory.add(
            f"创建研究任务: {topic}",
            metadata={"task_id": task_id, "depth": task.depth.value},
            layer="short_term",
        )
        return task

    def run(
        self,
        topic: str,
        depth: str | ResearchDepth = ResearchDepth.STANDARD,
    ) -> ResearchTask:
        """端到端执行：四 Agent 顺序协作。"""
        from datetime import datetime

        from ..common.observability import get_observability

        trace_id = set_trace_id()
        logger.info(f"[{trace_id}] 启动研究任务: {topic}")
        started_at = datetime.now()

        task = self.create_task(topic, depth)
        board = SharedBoard(task)
        error_msg = ""

        if self._graph is not None:
            # LangGraph 路径
            try:
                result = self._graph.invoke({"board": board})
                board = result.get("board", board)
            except Exception as e:
                logger.error(f"[{trace_id}] LangGraph 执行失败，回退到顺序循环: {e}", exc_info=True)
                error_msg = str(e)
                board = self._run_sequential(board, trace_id)
        else:
            # 降级路径
            try:
                board = self._run_sequential(board, trace_id)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[{trace_id}] 顺序执行失败: {e}", exc_info=True)

        # 同步黑板产物到任务对象
        task = board.sync_to_task()
        finished_at = datetime.now()
        logger.info(
            f"[{trace_id}] 任务完成: {task.task_id} status={task.status.value} "
            f"papers={len(task.papers)} kps={len(task.knowledge_points)} score={task.quality_score}"
        )
        # 任务档案归档（架构 6.5）
        try:
            get_observability().archive_task(board, started_at, finished_at, error=error_msg)
        except Exception as e:
            logger.warning(f"任务档案归档失败（不影响主流程）: {e}")
        return task

    def _run_sequential(self, board: SharedBoard, trace_id: str) -> SharedBoard:
        """降级路径：顺序循环（LangGraph 不可用时使用）。

        架构 3.1.2 异常升级：工人失败回退到 PENDING_PLANNING 后，
        重新调用 Planner 重新规划，最多重试 2 轮避免死循环。
        """
        max_replans = 2
        replan_count = 0
        # 用状态→Agent 映射，支持回退后跳回 Planner
        while replan_count <= max_replans:
            state = board.get_task_state()
            if state in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                break
            # 选 Agent
            if state == TaskStatus.PENDING_PLANNING:
                agent, timeout = self.planner, self._timeouts["planner"]
                if replan_count > 0:
                    board.log_stage("Orchestrator", f"第 {replan_count} 次重新规划", level="WARN")
            elif state in (TaskStatus.PENDING_SEARCH, TaskStatus.PENDING_FILTER):
                agent, timeout = self.collector, self._timeouts["collector"]
            elif state in (TaskStatus.PENDING_READING, TaskStatus.PENDING_ORGANIZING):
                agent, timeout = self.engineer, self._timeouts["engineer"]
            elif state == TaskStatus.PENDING_OUTPUT:
                agent, timeout = self.writer, self._timeouts["writer"]
            else:
                break
            logger.info(f"[{trace_id}] 执行 {agent.name} (state={state.value})")
            prev_state = state
            board = agent.safe_execute(board, timeout=timeout)
            new_state = board.get_task_state()
            # 检测是否回退到了 PENDING_PLANNING（异常升级）
            if new_state == TaskStatus.PENDING_PLANNING and prev_state != TaskStatus.PENDING_PLANNING:
                replan_count += 1
                logger.warning(f"[{trace_id}] 检测到回退，重新规划计数 {replan_count}/{max_replans}")
                if replan_count > max_replans:
                    logger.error(f"[{trace_id}] 重规划次数超限，终止任务")
                    board.update_task_state(TaskStatus.FAILED)
                    break
            elif new_state == TaskStatus.FAILED:
                logger.error(f"[{trace_id}] {agent.name} 失败，任务终止")
                break
        return board

    # ===== 流式接口：供 web/app.py 逐步 yield 进度 =====
    def run_stream(self, board: SharedBoard) -> Any:
        """流式执行：按阶段逐步返回 board，供 Gradio 流式 UI 使用。

        Yields:
            SharedBoard: 每个 Agent 执行后的黑板状态
        """
        trace_id = set_trace_id()
        logger.info(f"[{trace_id}] 启动流式任务: {board.task.topic}")

        stages: list[tuple[BaseAgent, float]] = [
            (self.planner, self._timeouts["planner"]),
            (self.collector, self._timeouts["collector"]),
            (self.engineer, self._timeouts["engineer"]),
            (self.writer, self._timeouts["writer"]),
        ]
        # 流式执行不能用 LangGraph（它会一次性走完），用顺序循环逐阶段 yield
        for agent, timeout in stages:
            if board.get_task_state() in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                break
            logger.info(f"[{trace_id}] 执行 {agent.name} (state={board.get_task_state().value})")
            board = agent.safe_execute(board, timeout=timeout)
            yield board
            if board.get_task_state() == TaskStatus.FAILED:
                logger.error(f"[{trace_id}] {agent.name} 失败，任务终止")
                break


# 全局单例
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
