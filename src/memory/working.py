"""工作记忆 - 任务周期内的中间产物。

对应架构 3.3.3 工作记忆层。
生命周期：任务执行期间有效，完成后由 Writer 沉淀到长期记忆。
存储：内存字典 + SQLite 持久化（Neo4j 不可用时降级）。
功能：维护任务产物（papers/kps/relations），支持图谱式关联查询。
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import deque
from datetime import datetime
from typing import Any, Callable

from ..common.data_models import MemoryItem
from ..common.logger import get_logger
from .base import BaseMemory

logger = get_logger(__name__)

# ===== 关系类型 → 权重映射 =====
# cites: 直接引用，最强；supports: 证据支撑；contradicts: 矛盾（仍相关但需区分）
# similar: 语义相似；related: 默认弱关联
_RELATION_WEIGHTS: dict[str, float] = {
    "cites": 1.0,
    "supports": 0.9,
    "contradicts": 0.7,
    "similar": 0.6,
    "related": 0.5,
}
_DEFAULT_RELATION_WEIGHT = 0.5

# 路径衰减系数：1 跳 = 1.0, 2 跳 = 0.5, 3 跳 = 0.25
_PATH_DECAY = 0.5


class WorkingMemory(BaseMemory):
    """任务周期内工作记忆。

    降级方案：内存 + SQLite 实现，Neo4j Lite 不可用时直接使用本实现。
    """

    def __init__(self, db_path: str = "data/working.db") -> None:
        self.db_path = db_path
        self._items: dict[str, MemoryItem] = {}
        # 图结构：实体 id → [(neighbor_id, relation_type, edge_weight), ...]
        # edge_weight 为具体实例权重（0~1），区别于关系类型权重
        self._edges: dict[str, list[tuple[str, str, float]]] = {}
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
                weight REAL DEFAULT 1.0,
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
        # 工作记忆主要按 keyword 匹配（图谱式查询另走 graph_search）
        q = query.lower()
        results = [
            item for item in self._items.values()
            if q in item.content.lower() or any(q in str(v).lower() for v in item.metadata.values())
        ]
        return results[:top_k]

    def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

    # ===== 图谱式查询（关联召回） =====
    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation_type: str = "related",
        weight: float | None = None,
    ) -> None:
        """建边。weight 为 None 时按关系类型自动赋权。

        edge_weight 表示该条边的实例强度（0~1），
        与关系类型权重 _RELATION_WEIGHTS 独立，最终有效权重 = edge_weight × relation_weight。
        """
        # 实例权重：显式 > 关系类型默认 > 1.0
        ew = weight if weight is not None else 1.0
        self._edges.setdefault(from_id, []).append((to_id, relation_type, ew))
        self._edges.setdefault(to_id, []).append((from_id, relation_type, ew))
        self._conn.execute(
            "INSERT OR IGNORE INTO working_edges VALUES (?, ?, ?, ?)",
            (from_id, to_id, relation_type, ew),
        )
        self._conn.commit()

    def graph_query(self, entity_id: str, depth: int = 1) -> list[str]:
        """从某实体出发，按深度检索关联实体 id（BFS）。

        保留向后兼容：API server / knowledge_base_tool 仍调用此方法。
        """
        visited: set[str] = {entity_id}
        frontier = [entity_id]
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                for neighbor, _rel, _w in self._edges.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return list(visited - {entity_id})

    def graph_search(
        self,
        query: str,
        top_k: int = 10,
        max_depth: int = 2,
        query_relevance_fn: Callable[[str, str], float] | None = None,
    ) -> list[tuple[float, MemoryItem]]:
        """图谱多因子关联召回。

        评分 = 路径分数 × 节点重要性 × 查询相关性

        五个因子：
        1. 边权重(edge_weight)：实例级强度，建边时指定
        2. 路径长度衰减：decay^hop，hop=0 → 1.0, hop=1 → 0.5, hop=2 → 0.25
        3. 节点重要性：度中心性 (degree / max_degree)
        4. 关系类型权重：cites/supports/contradicts/similar/related 各不同
        5. 查询相关性：节点内容与 query 的语义匹配度（由 query_relevance_fn 提供，
           无回调时用关键词重叠率降级）

        流程：关键词匹配找种子 → BFS 扩散 → 多路径取最大路径分 → 融合排序
        """
        if not self._items:
            return []

        # --- 1. 定位种子节点（关键词命中） ---
        q_lower = query.lower()
        query_terms = set(q_lower.split())
        seeds: set[str] = set()
        for nid, item in self._items.items():
            content_lower = item.content.lower()
            if q_lower in content_lower:
                seeds.add(nid)
            elif query_terms and query_terms & set(content_lower.split()):
                seeds.add(nid)
        if not seeds:
            # 无种子：退化为全部节点（小图可接受），大图则返回空
            if len(self._items) <= 50:
                seeds = set(self._items.keys())
            else:
                return []

        # --- 2. 节点重要性（度中心性） ---
        node_importance = self._compute_node_importance()

        # --- 3. 查询相关性函数 ---
        def _default_relevance(nid: str, content: str) -> float:
            """关键词重叠率作为降级相关性。"""
            if not query_terms:
                return 0.0
            content_terms = set(content.lower().split())
            if not content_terms:
                return 0.0
            return len(query_terms & content_terms) / len(query_terms)

        relevance_fn = query_relevance_fn or _default_relevance

        # --- 4. BFS 多跳扩散，累积路径分数 ---
        # node_best_path[nid] = 最大路径分数（多路径取最大，避免被弱路径拉低）
        node_best_path: dict[str, float] = {s: 1.0 for s in seeds}  # 种子 hop=0 → 1.0

        frontier: deque[tuple[str, float]] = deque((s, 1.0) for s in seeds)
        visited_hop: dict[str, int] = {s: 0 for s in seeds}

        while frontier:
            node, path_score = frontier.popleft()
            cur_hop = visited_hop[node]
            if cur_hop >= max_depth:
                continue
            for neighbor, rel_type, edge_w in self._edges.get(node, []):
                rel_w = _RELATION_WEIGHTS.get(rel_type, _DEFAULT_RELATION_WEIGHT)
                # 有效边权重 = 实例权重 × 关系类型权重
                eff_edge_w = edge_w * rel_w
                # 路径分数 = 父路径分 × 衰减 × 有效边权
                new_score = path_score * _PATH_DECAY * eff_edge_w
                # 未访问或找到更高分路径（衰减更少 = 更高分）
                if neighbor not in node_best_path or new_score > node_best_path[neighbor]:
                    node_best_path[neighbor] = new_score
                    visited_hop[neighbor] = cur_hop + 1
                    frontier.append((neighbor, new_score))

        # --- 5. 融合三因子评分 ---
        scored: list[tuple[float, MemoryItem]] = []
        for nid, path_score in node_best_path.items():
            item = self._items.get(nid)
            if item is None or not item.content:
                continue
            importance = node_importance.get(nid, 0.0)
            relevance = max(0.0, min(1.0, relevance_fn(nid, item.content)))
            final_score = path_score * importance * relevance
            if final_score > 0:
                scored.append((final_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def _compute_node_importance(self) -> dict[str, float]:
        """节点重要性：度中心性 = degree(node) / max_degree。

        度 = 入边 + 出边总数（无向图）。孤立节点重要性为 0。
        """
        if not self._edges:
            # 无边图：所有节点同等重要（避免全 0 导致分数归零）
            uniform = 1.0 / max(len(self._items), 1) if self._items else 0.0
            return {nid: uniform for nid in self._items}
        degrees: dict[str, int] = {}
        for nid, neighbors in self._edges.items():
            degrees[nid] = degrees.get(nid, 0) + len(neighbors)
        # 也统计无邻居但作为邻居出现的节点
        for neighbors in self._edges.values():
            for neighbor, _, _ in neighbors:
                degrees[neighbor] = degrees.get(neighbor, 0) + 1
        max_deg = max(degrees.values()) if degrees else 1
        if max_deg == 0:
            max_deg = 1
        # 归一化到 [0, 1]，孤立节点（度=0）给一个小底值避免完全抹零
        return {
            nid: (degrees.get(nid, 0) / max_deg) if max_deg > 0 else 0.0
            for nid in self._items
        }

    def clear(self) -> None:
        """任务完成后清空工作记忆。"""
        self._items.clear()
        self._edges.clear()
        self._conn.execute("DELETE FROM working_memory")
        self._conn.execute("DELETE FROM working_edges")
        self._conn.commit()
