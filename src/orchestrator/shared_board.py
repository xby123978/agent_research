"""共享黑板。

对应架构 3.2.2 四 Agent 协作机制：Agent 间通过共享黑板读写中间产物，
黑板内部维护任务状态、计划、文献、知识点、报告等结构化数据。
降级方案：内存字典实现，无外部依赖（Neo4j Lite 不可用时直接用内存版）。
"""

from __future__ import annotations

import threading
from typing import Any

from ..common.data_models import PaperMeta, ResearchPlan, ResearchTask, TaskStatus
from ..common.logger import get_logger

logger = get_logger(__name__)


class SharedBoard:
    """Agent 协作共享黑板。

    线程安全，单任务周期内有效。任务完成后由 Writer Agent 沉淀到长期记忆。
    """

    def __init__(self, task: ResearchTask) -> None:
        self._task = task
        self._lock = threading.RLock()
        # 中间产物区：各 Agent 写入、其他 Agent 读取
        self._artifacts: dict[str, Any] = {
            "plan": None,            # ResearchPlan
            "papers": [],            # list[PaperMeta]
            "knowledge_points": [],  # list[KnowledgePoint]
            "relations": [],        # list[Relation]
            "draft_report": "",     # 草稿报告
            "final_report": "",     # 最终报告
            "stage_logs": [],        # 各阶段日志
            "tool_calls": [],        # 工具调用记录
        }

    # ===== 任务状态管理 =====
    def get_task_state(self) -> TaskStatus:
        with self._lock:
            return self._task.status

    def update_task_state(self, new_state: TaskStatus) -> None:
        with self._lock:
            old = self._task.status
            self._task.status = new_state
            logger.info(
                f"状态转移: {old.value} → {new_state.value} (task={self._task.task_id})"
            )

    # ===== 读写中间产物 =====
    def read(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._artifacts.get(key, default)

    def write(self, key: str, value: Any) -> None:
        with self._lock:
            self._artifacts[key] = value
            logger.debug(f"黑板写入: {key} type={type(value).__name__}")

    def append(self, key: str, item: Any) -> None:
        """向列表型产物追加元素（如 papers / knowledge_points）。"""
        with self._lock:
            if key not in self._artifacts:
                self._artifacts[key] = []
            if not isinstance(self._artifacts[key], list):
                raise ValueError(f"黑板键 {key} 非列表类型，无法 append")
            self._artifacts[key].append(item)

    def extend(self, key: str, items: list[Any]) -> None:
        with self._lock:
            if key not in self._artifacts:
                self._artifacts[key] = []
            self._artifacts[key].extend(items)

    # ===== 任务对象访问 =====
    @property
    def task(self) -> ResearchTask:
        return self._task

    def sync_to_task(self) -> ResearchTask:
        """将黑板产物同步回任务对象（用于持久化与对外输出）。"""
        with self._lock:
            self._task.plan = self._artifacts.get("plan")
            papers = self._artifacts.get("papers") or []
            self._task.papers = papers if isinstance(papers, list) else []
            kps = self._artifacts.get("knowledge_points") or []
            self._task.knowledge_points = kps if isinstance(kps, list) else []
            report = self._artifacts.get("final_report") or self._artifacts.get("draft_report") or ""
            self._task.report = report
        return self._task

    def log_stage(self, stage: str, message: str, level: str = "INFO") -> None:
        with self._lock:
            self._artifacts["stage_logs"].append({"stage": stage, "message": message, "level": level})
