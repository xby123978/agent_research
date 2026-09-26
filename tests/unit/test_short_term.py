"""短期记忆模块测试。"""

from __future__ import annotations

from src.common.data_models import MemoryItem
from src.memory.short_term import ShortTermMemory


def test_short_term_add_and_get(tmp_data_dir):
    mem = ShortTermMemory(db_path=str(tmp_data_dir / "st.db"))
    item = MemoryItem(content="hello world")
    item_id = mem.add(item)
    assert item_id.startswith("stm_")
    got = mem.get(item_id)
    assert got is not None
    assert got.content == "hello world"


def test_short_term_search_returns_recent(tmp_data_dir):
    mem = ShortTermMemory(db_path=str(tmp_data_dir / "st2.db"))
    for i in range(5):
        mem.add(MemoryItem(content=f"item {i}"))
    results = mem.search("query", top_k=3)
    assert len(results) == 3
