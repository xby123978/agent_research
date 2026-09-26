"""工作记忆 - 任务周期内的中间产物。

对应架构 3.3.3 工作记忆层。
生命周期：任务执行期间有效，完成后由 Writer 沉淀到长期记忆。
存储：内存字典 + SQLite 持久化（Neo4j 不可用时降级）。
功能：维护任务产物（papers/kps/relations），支持图谱式关联查询。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from ..common.data_models import MemoryItem
from ..common.logger import get_logger
from .base import BaseMemory

logger = get_logger(__name__)


class WorkingMemory(BaseMemory):
    """任务周期内工作记忆。

    降级方案：内存 + SQLite 实现，Neo4j Lite 不可用时直接使用本实现。
    """

    def __init__(self, db_path: str = "data/working.db") -> None:
        self.db_path = db_path
        self._items: dict[str, MemoryItem] = {}
        # 图结构：实体 id → 关联 id 列表（简化版图谱）
        self._edges: dict[str, list[str]] = {}
        self._init_db()

    def _init_db(self) -> None:
        import os

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS working_memory (
                item_id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                source TEXT,
                created_at TEXT
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS working_edges (
                from_id TEXT,
                to_id TEXT,
                relation_type TEXT,
                PRIMARY KEY (from_id, to_id, relation_type)
            )"""
        )
        self._conn.commit()

    def add(self, item: MemoryItem) -> str:
        item.source = "working"
        if not item.item_id:
            import uuid

            item.item_id = f"work_{uuid.uuid4().hex[:10]}"
        self._items[item.item_id] = item
        self._conn.execute(
            "INSERT OR REPLACE INTO working_memory VALUES (?, ?, ?, ?, ?)",
            (item.item_id, item.content, json.dumps(item.metadata), item.source, datetime.now().isoformat()),
        )
        self._conn.commit()
        return item.item_id

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        # 工作记忆主要按 keyword 匹配（图谱式查询另走 graph_query）
        q = query.lower()
        results = [
            item for item in self._items.values()
            if q in item.content.lower() or any(q in str(v).lower() for v in item.metadata.values())
        ]
        return results[:top_k]

    def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

    # ===== 图谱式查询（关联召回） =====
    def add_edge(self, from_id: str, to_id: str, relation_type: str = "related") -> None:
        self._edges.setdefault(from_id, []).append(to_id)
        self._edges.setdefault(to_id, []).append(from_id)
        self._conn.execute(
            "INSERT OR IGNORE INTO working_edges VALUES (?, ?, ?)",
            (from_id, to_id, relation_type),
        )
        self._conn.commit()

    def graph_query(self, entity_id: str, depth: int = 1) -> list[str]:
        """从某实体出发，按深度检索关联实体 id（BFS）。"""
        visited: set[str] = {entity_id}
        frontier = [entity_id]
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                for neighbor in self._edges.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return list(visited - {entity_id})

    def clear(self) -> None:
        """任务完成后清空工作记忆。"""
        self._items.clear()
        self._edges.clear()
        self._conn.execute("DELETE FROM working_memory")
        self._conn.execute("DELETE FROM working_edges")
        self._conn.commit()
