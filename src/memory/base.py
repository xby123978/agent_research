"""记忆层抽象基类。

对应架构 3.3 记忆模块，定义统一记忆接口，
不同层级实现各自存储与召回策略，替换实现不影响上层。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..common.data_models import MemoryItem


class BaseMemory(ABC):
    """记忆层抽象接口。"""

    layer: str = "base"

    @abstractmethod
    def add(self, item: MemoryItem) -> str:
        """写入一条记忆，返回 item_id。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """根据查询文本召回记忆。"""

    @abstractmethod
    def get(self, item_id: str) -> MemoryItem | None:
        """按 id 获取单条记忆。"""
