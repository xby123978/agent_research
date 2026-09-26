"""长期记忆模块测试（使用内置降级向量库 + Hashing 嵌入）。"""

from __future__ import annotations

from src.common.data_models import MemoryItem
from src.memory.long_term import LongTermMemory


def test_long_term_add_and_search():
    mem = LongTermMemory()
    mem.add(MemoryItem(content="Transformer uses self-attention mechanism"))
    mem.add(MemoryItem(content="CNN is for image convolution"))
    results = mem.search("attention transformer", top_k=2)
    assert len(results) >= 1
    # 相关项应排在前面
    assert "Transformer" in results[0].content or "attention" in results[0].content.lower()


def test_long_term_get():
    mem = LongTermMemory()
    item = MemoryItem(content="RAG retrieval augmented generation")
    item_id = mem.add(item)
    got = mem.get(item_id)
    assert got is not None
    assert "RAG" in got.content
