"""知识库工具。

对应架构 3.4.2 知识库 Server：记忆读写、知识图谱查询。
本地实现，对接记忆模块 MemoryManager，非 Docker 依赖。
"""

from __future__ import annotations

import time
from typing import Any

from ..common.logger import get_logger
from ..memory.manager import get_memory
from .base import BaseTool, ToolResult, ToolSchema

logger = get_logger(__name__)


class KnowledgeBaseTool(BaseTool):
    """知识库读写工具（对接记忆模块）。"""

    schema = ToolSchema(
        name="knowledge_base",
        description="记忆读写与知识图谱查询（对接四层记忆）",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "search | add | get | graph",
                    "default": "search",
                },
                "query": {"type": "string", "description": "检索/写入内容"},
                "layer": {
                    "type": "string",
                    "description": "short_term | long_term | working | preference | auto",
                    "default": "auto",
                },
                "top_k": {"type": "integer", "default": 5},
                "item_id": {"type": "string", "description": "get/graph 用"},
                "depth": {"type": "integer", "default": 1, "description": "graph 查询深度"},
                "metadata": {"type": "object", "description": "add 时的元数据"},
            },
            "required": ["action", "query"],
        },
    )

    def call(self, **kwargs: Any) -> ToolResult:
        start = time.time()
        action = kwargs.get("action", "search")
        memory = get_memory()
        try:
            if action == "search":
                query = kwargs.get("query", "")
                top_k = int(kwargs.get("top_k", 5))
                layers = kwargs.get("layer", "auto")
                # search 仅接受多层列表或默认
                if layers in ("auto", "all"):
                    layer_list = None
                else:
                    layer_list = [layers]
                items = memory.search(query, top_k=top_k, layers=layer_list)
                return ToolResult(
                    success=True,
                    content=[i.model_dump() for i in items],
                    duration_ms=(time.time() - start) * 1000,
                )
            if action == "add":
                content = kwargs.get("query", "")
                layer = kwargs.get("layer", "auto")
                metadata = kwargs.get("metadata", {})
                item_id = memory.add(content, metadata=metadata, layer=layer)
                return ToolResult(
                    success=bool(item_id),
                    content=[{"item_id": item_id}],
                    duration_ms=(time.time() - start) * 1000,
                )
            if action == "get":
                item_id = kwargs.get("item_id", "")
                item = memory.get(item_id)
                if not item:
                    return ToolResult(success=False, error="条目不存在", duration_ms=(time.time() - start) * 1000)
                return ToolResult(
                    success=True,
                    content=[item.model_dump()],
                    duration_ms=(time.time() - start) * 1000,
                )
            if action == "graph":
                if memory.working is None:
                    return ToolResult(success=False, error="工作记忆未初始化", duration_ms=(time.time() - start) * 1000)
                entity_id = kwargs.get("item_id", "")
                depth = int(kwargs.get("depth", 1))
                related = memory.working.graph_query(entity_id, depth)
                nodes = []
                for rid in related:
                    it = memory.working.get(rid)
                    if it:
                        nodes.append(it.model_dump())
                return ToolResult(
                    success=True,
                    content=nodes,
                    duration_ms=(time.time() - start) * 1000,
                )
            return ToolResult(success=False, error=f"未知 action: {action}", duration_ms=(time.time() - start) * 1000)
        except Exception as e:
            logger.error(f"知识库操作失败: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e), duration_ms=(time.time() - start) * 1000)
