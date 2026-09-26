"""记忆模块包。

对应架构 3.3 四层记忆架构。MVP 阶段实现短期记忆与长期向量记忆两层，
阶段二补齐工作记忆与偏好记忆。所有记忆调用走统一 MemoryManager 接口。
"""
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .manager import MemoryManager, get_memory

__all__ = [
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryManager",
    "get_memory",
]
