"""可观测性与任务档案。

对应架构 6.5 可观测性设计：
- 核心指标聚合：任务成功率、平均耗时、Token 消耗、工具调用成功率
- 任务档案：每个任务完整存档到 outputs/archive/{task_id}.json
  含输入、中间过程、输出、质量评分，支持复盘与排查

存储：SQLite（本地，非 Docker）+ JSON 文件归档。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..common.config import get_env
from ..common.logger import get_logger
from ..orchestrator.shared_board import SharedBoard

logger = get_logger(__name__)


class Observability:
    """可观测性管理器：指标聚合 + 任务档案。

    单例，线程安全。
    """

    def __init__(self) -> None:
        env = get_env()
        self.output_dir = Path(env["output_dir"])
        self.archive_dir = self.output_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(Path(env["data_dir"]) / "sqlite" / "metrics.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS task_metrics (
                    task_id TEXT PRIMARY KEY,
                    topic TEXT,
                    status TEXT,
                    depth TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_sec REAL,
                    paper_count INTEGER,
                    kp_count INTEGER,
                    quality_score REAL,
                    stage_logs TEXT,
                    error TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    tool_name TEXT,
                    success INTEGER,
                    duration_ms REAL,
                    called_at TEXT
                )"""
            )

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def archive_task(
        self,
        board: SharedBoard,
        started_at: datetime,
        finished_at: datetime,
        error: str = "",
    ) -> str:
        """任务完整存档：JSON 文件 + metrics 表。

        对应架构 6.5 任务档案：输入/中间过程/输出/质量评分。
        """
        task = board.task
        duration = (finished_at - started_at).total_seconds()
        stage_logs = board.read("stage_logs", [])
        archive = {
            "task_id": task.task_id,
            "topic": task.topic,
            "depth": task.depth.value,
            "status": task.status.value,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_sec": round(duration, 2),
            "quality_score": task.quality_score,
            "papers": [p.model_dump() for p in task.papers],
            "knowledge_points": [kp.model_dump() for kp in task.knowledge_points],
            "report": task.report,
            "stage_logs": stage_logs,
            "plan": task.plan.model_dump() if task.plan else None,
            "error": error,
            # 阶段三：JEV 各阶段评分记录
            "jev_scores": board.read("jev_scores", {}),
        }
        # 写 JSON 文件
        archive_path = self.archive_dir / f"{task.task_id}.json"
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(archive, f, ensure_ascii=False, indent=2, default=str)
        # 写 metrics 表
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO task_metrics
                   (task_id, topic, status, depth, started_at, finished_at,
                    duration_sec, paper_count, kp_count, quality_score, stage_logs, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.task_id,
                    task.topic,
                    task.status.value,
                    task.depth.value,
                    started_at.isoformat(),
                    finished_at.isoformat(),
                    duration,
                    len(task.papers),
                    len(task.knowledge_points),
                    task.quality_score,
                    json.dumps(stage_logs, ensure_ascii=False, default=str),
                    error,
                ),
            )
        logger.info(f"任务档案已归档: {archive_path}")
        return str(archive_path)

    def record_tool_call(
        self,
        task_id: str,
        tool_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """记录单次工具调用（架构 6.5 工具调用成功率）。"""
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """INSERT INTO tool_calls (task_id, tool_name, success, duration_ms, called_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    task_id,
                    tool_name,
                    1 if success else 0,
                    duration_ms,
                    datetime.now().isoformat(),
                ),
            )

    def get_metrics_summary(self) -> dict[str, Any]:
        """聚合核心指标（架构 6.5）。"""
        with self._get_conn() as conn:
            # 任务统计
            row = conn.execute(
                """SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS success,
                    AVG(duration_sec) AS avg_duration,
                    AVG(quality_score) AS avg_quality
                   FROM task_metrics"""
            ).fetchone()
            total, success, avg_dur, avg_q = row
            # 工具调用统计
            tool_row = conn.execute(
                """SELECT
                    COUNT(*) AS total_calls,
                    SUM(success) AS success_calls
                   FROM tool_calls"""
            ).fetchone()
            tool_total, tool_success = tool_row
        return {
            "tasks": {
                "total": total or 0,
                "success": success or 0,
                "success_rate": round((success or 0) / max(total or 1, 1), 3),
                "avg_duration_sec": round(avg_dur or 0, 2),
                "avg_quality_score": round(avg_q or 0, 3),
            },
            "tools": {
                "total_calls": tool_total or 0,
                "success_calls": tool_success or 0,
                "success_rate": round(
                    (tool_success or 0) / max(tool_total or 1, 1), 3
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }


_observability_instance: Observability | None = None


def get_observability() -> Observability:
    """全局单例。"""
    global _observability_instance
    if _observability_instance is None:
        _observability_instance = Observability()
    return _observability_instance
