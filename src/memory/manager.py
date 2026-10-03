"""统一记忆管理接口。

对应架构 4 架构遵循原则第 5 条：调用走统一记忆接口。
Agent 通过 MemoryManager 透明地读写四层记忆，无需感知底层实现。
阶段二：四层记忆完整接入（短期/工作/长期/偏好）。
"""

from __future__ import annotations

from ..common.data_models import MemoryItem
from ..common.logger import get_logger
from .base import BaseMemory
from .long_term import LongTermMemory
from .preference import PreferenceMemory
from .short_term import ShortTermMemory
from .working import WorkingMemory

logger = get_logger(__name__)


class MemoryManager:
    """统一记忆接口，聚合四层记忆。

    层次：
    - short_term: 会话短期记忆（SQLite）
    - working: 任务周期工作记忆（图谱 + SQLite）
    - long_term: 永久长期记忆（向量库）
    - preference: 用户偏好（配置文件）
    """

    def __init__(self) -> None:
        self.short_term: BaseMemory = ShortTermMemory()
        self.working: WorkingMemory | None = None
        self.long_term: BaseMemory | None = None  # 延迟初始化，避免嵌入未就绪
        self.preference: PreferenceMemory | None = None
        self._init_all()

    def _init_all(self) -> None:
        # 工作记忆（每个任务独立实例，但全局单例也行，clear 后可复用）
        try:
            self.working = WorkingMemory()
        except Exception as e:
            logger.warning(f"工作记忆初始化失败: {e}")
        # 长期记忆
        try:
            self.long_term = LongTermMemory()
        except Exception as e:
            logger.warning(f"长期记忆初始化失败，仅启用短期记忆: {e}")
            self.long_term = None
        # 偏好记忆
        try:
            self.preference = PreferenceMemory()
        except Exception as e:
            logger.warning(f"偏好记忆初始化失败: {e}")

    def add(self, content: str, metadata: dict | None = None, layer: str = "auto") -> str:
        """写入记忆。

        Args:
            content: 记忆内容
            metadata: 元数据
            layer: short_term | working | long_term | preference | auto
        """
        # 写入前质量校验（架构 3.3.2）：空内容或过短内容不进入长期记忆
        if not content or not content.strip():
            logger.warning("记忆写入被拒：内容为空")
            return ""
        if layer in ("long_term", "auto") and len(content.strip()) < 10:
            logger.warning(f"记忆写入被拒：内容过短（{len(content)} 字），不进入长期记忆")
            if layer == "long_term":
                return ""
        # JEV 质量评分（架构 3.3.2 阶段三）：低于 0.3 不进入长期记忆
        jev_score = 0.5
        if layer in ("long_term", "auto"):
            try:
                from ..models.jev import get_jev
                jev_score = get_jev().score(content, "记忆内容质量：信息密度、学术价值、可复用性")
            except Exception as e:
                logger.debug(f"JEV 记忆评分失败，用默认 0.5: {e}")
            if jev_score < 0.3:
                logger.warning(f"记忆写入被拒：JEV 评分 {jev_score:.2f} 低于 0.3，不进入长期记忆")
                if layer == "long_term":
                    return ""
        # JEV 语义去重（架构 3.3.2）：与已有记忆相似度 > 0.85 则跳过
        if layer in ("long_term", "auto") and self.long_term is not None:
            if self._is_duplicate(content):
                logger.info("记忆写入被拒：与已有记忆语义重复")
                if layer == "long_term":
                    return ""
        item = MemoryItem(content=content, metadata=metadata or {})
        item.metadata["jev_score"] = jev_score  # 记录评分便于召回排序
        if layer in ("short_term", "auto"):
            sid = self.short_term.add(item.model_copy())
        if layer == "working" and self.working is not None:
            return self.working.add(item.model_copy())
        if layer in ("long_term", "auto"):
            if self.long_term is not None:
                lid = self.long_term.add(item.model_copy())
                # 自动建立图谱关联（架构 3.3.2）：新记忆与已有最相似记忆建立边
                self._auto_link(item.item_id, content)
                if layer == "long_term":
                    return lid
        if layer == "preference" and self.preference is not None:
            return self.preference.add(item.model_copy())
        if layer == "short_term":
            return sid  # type: ignore[possibly-undefined]
        if layer == "auto":
            return sid  # type: ignore[possibly-undefined]
        return item.item_id

    def _is_duplicate(self, content: str) -> bool:
        """JEV 语义去重：判断新内容是否与已有记忆重复。

        用 JEV judge 任务判断内容是否重复。
        """
        if self.long_term is None:
            return False
        try:
            # 取最相似的 1 条已有记忆
            related = self.long_term.search(content, top_k=1)
            if not related:
                return False
            existing = related[0].content
            # 用 JEV 判断是否语义重复
            from ..models.jev import get_jev
            prompt_content = f"已有记忆：{existing[:200]}\n\n新记忆：{content[:200]}"
            return get_jev().judge(
                prompt_content,
                "判断新记忆与已有记忆是否语义重复（相似度>0.85）",
            )
        except Exception as e:
            logger.debug(f"JEV 去重判断失败（不阻断）: {e}")
            return False

    def _auto_link(self, item_id: str, content: str) -> None:
        """自动建立图谱关联：找最相似记忆建边（架构 3.3.2）。

        降级：工作记忆可用时写入边，不可用时仅记录日志。
        """
        if self.working is None:
            return
        try:
            # 用工作记忆图结构记录关联（简化版：按关键词重叠建边）
            related = self.working.search(content, top_k=1)
            if related and related[0].item_id and related[0].item_id != item_id:
                # 边权重 = 关键词重叠率（0~1），无 score 时用 0.5 兜底
                ew = related[0].score if related[0].score and 0 < related[0].score <= 1 else 0.5
                self.working.add_edge(item_id, related[0].item_id, "similar", weight=ew)
                logger.debug(f"自动建边: {item_id} ↔ {related[0].item_id} (w={ew:.2f})")
        except Exception as e:
            logger.debug(f"自动建边失败（不影响主流程）: {e}")

    def search(self, query: str, top_k: int = 5, layers: list[str] | None = None) -> list[MemoryItem]:
        """跨层召回记忆，合并去重后按相关性排序。"""
        layers = layers or ["short_term", "long_term"]
        results: list[MemoryItem] = []
        if "short_term" in layers:
            results.extend(self.short_term.search(query, top_k))
        if "working" in layers and self.working is not None:
            results.extend(self.working.search(query, top_k))
        if "long_term" in layers and self.long_term is not None:
            results.extend(self.long_term.search(query, top_k))
        if "preference" in layers and self.preference is not None:
            results.extend(self.preference.search(query, top_k))
        # 去重
        seen: set[str] = set()
        deduped = [x for x in results if not (x.item_id in seen or seen.add(x.item_id))]  # type: ignore[func-returns-value]
        # 按 score 排序（短期记忆 score=0 排后）
        deduped.sort(key=lambda x: x.score, reverse=True)
        return deduped[:top_k]

    def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        max_tokens: int | None = None,
        layers: list[str] | None = None,
    ) -> tuple[list[MemoryItem], str]:
        """跨层召回 + 短期记忆摘要（方案 B 接口）。

        返回 (memory_items, short_term_summary)：
        - memory_items: 跨层召回的记忆条目（短期原文 + 长期语义等）
        - short_term_summary: 旧轮次累积摘要，供 Agent 拼 prompt 时作为前置上下文

        Args:
            query: 查询文本
            top_k: 每层召回上限
            max_tokens: token 预算，短期记忆原文 + 摘要按此裁剪。
                        None 不限制。长期/工作/偏好记忆不受此预算约束。
            layers: 参与召回的层级，默认 ["short_term", "long_term"]
        """
        layers = layers or ["short_term", "long_term"]
        summary = ""
        results: list[MemoryItem] = []

        # 短期记忆：用 search_with_summary 拿原文 + 摘要
        if "short_term" in layers and isinstance(self.short_term, ShortTermMemory):
            recent, summary = self.short_term.search_with_summary(
                query, top_k=top_k, max_tokens=max_tokens
            )
            results.extend(recent)
        elif "short_term" in layers:
            results.extend(self.short_term.search(query, top_k))

        if "working" in layers and self.working is not None:
            results.extend(self.working.search(query, top_k))
        if "long_term" in layers and self.long_term is not None:
            results.extend(self.long_term.search(query, top_k))
        if "preference" in layers and self.preference is not None:
            results.extend(self.preference.search(query, top_k))

        # 去重 + 排序
        seen: set[str] = set()
        deduped = [x for x in results if not (x.item_id in seen or seen.add(x.item_id))]  # type: ignore[func-returns-value]
        deduped.sort(key=lambda x: x.score, reverse=True)
        return deduped[:top_k], summary

    def get(self, item_id: str, layer: str = "auto") -> MemoryItem | None:
        if layer in ("short_term", "auto"):
            item = self.short_term.get(item_id)
            if item:
                return item
        if layer in ("working", "auto") and self.working is not None:
            item = self.working.get(item_id)
            if item:
                return item
        if layer in ("long_term", "auto") and self.long_term is not None:
            return self.long_term.get(item_id)
        if layer in ("preference", "auto") and self.preference is not None:
            return self.preference.get(item_id)
        return None

    def get_pref(self, key: str, default=None):
        """便捷读取偏好。"""
        if self.preference is not None:
            return self.preference.get_pref(key, default)
        return default

    def clear_working(self) -> None:
        """任务结束后清空工作记忆。"""
        if self.working is not None:
            self.working.clear()


_memory_instance: MemoryManager | None = None


def get_memory() -> MemoryManager:
    """获取全局记忆管理单例。"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = MemoryManager()
    return _memory_instance
