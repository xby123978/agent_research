"""短期记忆模块。

对应架构 3.3.1 短期记忆层：
- 存储内容：会话历史、推理过程、工具调用日志
- 存储介质：内存 + SQLite
- 召回策略：全量窗口召回
- 生命周期：会话结束后保留 7 天
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from ..common.config import get_env, get_settings
from ..common.data_models import MemoryItem
from ..common.exceptions import MemoryError
from ..common.logger import get_logger
from .base import BaseMemory

logger = get_logger(__name__)


class ShortTermMemory(BaseMemory):
    """短期记忆：内存 + SQLite 持久化，全量窗口召回。"""

    layer = "short_term"

    def __init__(self, db_path: str | None = None) -> None:
        env = get_env()
        settings = get_settings().get("memory", {}).get("short_term", {})
        self.retention_days = int(settings.get("retention_days", 7))
        self.max_window = int(settings.get("max_window", 50))

        data_dir = Path(env["data_dir"]) / "sqlite"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(data_dir / "short_term.db")
        self._lock = threading.Lock()
        self._buffer: list[MemoryItem] = []  # 内存窗口
        self._init_db()

    def _init_db(self) -> None:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS memory_items (
                        item_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        source TEXT,
                        created_at TEXT
                    )"""
                )
        except Exception as e:
            raise MemoryError(f"短期记忆数据库初始化失败: {e}") from e

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add(self, item: MemoryItem) -> str:
        if not item.item_id:
            item.item_id = f"stm_{uuid.uuid4().hex[:12]}"
        item.source = self.layer
        item.created_at = datetime.now()
        with self._lock:
            self._buffer.append(item)
            if len(self._buffer) > self.max_window:
                self._buffer = self._buffer[-self.max_window :]
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_items VALUES (?,?,?,?,?)",
                    (
                        item.item_id,
                        item.content,
                        str(item.metadata),
                        item.source,
                        item.created_at.isoformat(),
                    ),
                )
        except Exception as e:
            logger.warning(f"短期记忆持久化失败: {e}")
        logger.debug(f"短期记忆写入: {item.item_id}")
        return item.item_id

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """全量窗口召回：按时间倒序返回最近 top_k 条。

        短期记忆不做语义检索，仅按时间窗口返回。
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        # 内存窗口优先
        items = [x for x in self._buffer if x.created_at >= cutoff]
        if not items:
            try:
                with self._get_conn() as conn:
                    rows = conn.execute(
                        "SELECT item_id, content, metadata, source, created_at "
                        "FROM memory_items WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                        (cutoff.isoformat(), self.max_window),
                    ).fetchall()
                items = [
                    MemoryItem(
                        item_id=r[0],
                        content=r[1],
                        metadata=eval(r[2]) if r[2] else {},  # noqa: S307
                        source=r[3],
                        created_at=datetime.fromisoformat(r[4]),
                    )
                    for r in rows
                ]
            except Exception as e:
                logger.warning(f"短期记忆查询失败: {e}")
                items = []
        return items[:top_k]

    def get(self, item_id: str) -> MemoryItem | None:
        for x in self._buffer:
            if x.item_id == item_id:
                return x
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT item_id, content, metadata, source, created_at "
                    "FROM memory_items WHERE item_id = ?",
                    (item_id,),
                ).fetchone()
            if row:
                return MemoryItem(
                    item_id=row[0],
                    content=row[1],
                    metadata=eval(row[2]) if row[2] else {},  # noqa: S307
                    source=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                )
        except Exception as e:
            logger.warning(f"短期记忆 get 失败: {e}")
        return None

    def cleanup_expired(self) -> int:
        """清理过期记忆，返回删除条数。"""
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
        try:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "DELETE FROM memory_items WHERE created_at < ?", (cutoff,)
                )
                return cur.rowcount
        except Exception as e:
            logger.warning(f"短期记忆清理失败: {e}")
            return 0
