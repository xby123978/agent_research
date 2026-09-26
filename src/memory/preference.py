"""偏好记忆 - 用户研究偏好。

对应架构 3.3.4 偏好记忆层。
存储：配置文件 + 向量库（任务启动时全量加载）。
内容：研究风格、关注领域、质量阈值、输出偏好。
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..common.data_models import MemoryItem
from ..common.logger import get_logger
from .base import BaseMemory

logger = get_logger(__name__)

DEFAULT_PREFS: dict[str, Any] = {
    "research_style": "balanced",          # concise | balanced | detailed
    "focus_areas": [],                      # 关注领域列表
    "min_quality_threshold": 0.7,           # 最低质量阈值
    "max_papers_per_topic": 20,
    "preferred_sources": ["openalex", "crossref"],
    "language": "zh",
    "output_format": "markdown",
    "citation_style": "apa",                # apa | ieee | chicago
    "report_length": "medium",              # short | medium | long
}


class PreferenceMemory(BaseMemory):
    """用户偏好记忆。永久存储，可手动修改。

    降级方案：JSON 配置文件实现（向量库不可用时，按字段名匹配）。
    """

    def __init__(self, prefs_path: str = "data/preferences.json") -> None:
        self.prefs_path = prefs_path
        self._prefs: dict[str, Any] = {}
        self._items: dict[str, MemoryItem] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.prefs_path):
            try:
                with open(self.prefs_path, encoding="utf-8") as f:
                    self._prefs = json.load(f)
                logger.info(f"偏好记忆加载: {len(self._prefs)} 项")
            except Exception as e:
                logger.warning(f"偏好记忆加载失败，使用默认值: {e}")
                self._prefs = dict(DEFAULT_PREFS)
        else:
            self._prefs = dict(DEFAULT_PREFS)
            self._save()
            logger.info("偏好记忆初始化为默认值")

        # 同步到 items 字典
        for k, v in self._prefs.items():
            item_id = f"pref_{k}"
            self._items[item_id] = MemoryItem(
                item_id=item_id, content=str(v), metadata={"key": k, "value": v}, source="preference"
            )

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.prefs_path) or ".", exist_ok=True)
        with open(self.prefs_path, "w", encoding="utf-8") as f:
            json.dump(self._prefs, f, ensure_ascii=False, indent=2)

    def get_pref(self, key: str, default: Any = None) -> Any:
        """读取单个偏好。"""
        return self._prefs.get(key, default if default is not None else DEFAULT_PREFS.get(key))

    def set_pref(self, key: str, value: Any) -> None:
        """更新偏好（立即持久化）。"""
        self._prefs[key] = value
        item_id = f"pref_{key}"
        self._items[item_id] = MemoryItem(
            item_id=item_id, content=str(value), metadata={"key": k for k in [key]}, source="preference"
        )
        self._save()
        logger.info(f"偏好更新: {key} = {value}")

    def all_prefs(self) -> dict[str, Any]:
        """返回全部偏好。"""
        return dict(self._prefs)

    # ===== BaseMemory 接口实现 =====
    def add(self, item: MemoryItem) -> str:
        # 偏好记忆通过 set_pref 更新，add 仅作兼容
        key = item.metadata.get("key", "custom")
        self.set_pref(key, item.metadata.get("value", item.content))
        return f"pref_{key}"

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        q = query.lower()
        results = [
            item for item in self._items.values()
            if q in item.content.lower() or q in item.metadata.get("key", "").lower()
        ]
        return results[:top_k]

    def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)
