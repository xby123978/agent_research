"""端到端链路测试（使用 Mock 工具与 Mock LLM，不依赖外部网络）。

对应 Claude.md 阶段一验收标准：端到端任务成功率。
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[2] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from src.agents.planner_agent import PlannerAgent
from src.common.data_models import PaperMeta
from src.mcp.client import MCPClient
from src.models.llm import _MockLLM
from src.tools.base import BaseTool, ToolResult, ToolSchema


class _FakeSearchTool(BaseTool):
    """返回伪造文献的检索工具，用于端到端测试。"""

    def __init__(self, name: str) -> None:
        self.schema = ToolSchema(name=name, description="fake", input_schema={})

    def call(self, query: str = "", max_results: int = 5, **_):  # type: ignore[override]
        papers = [
            PaperMeta(
                paper_id=f"fake_{self.schema.name}_{i}",
                title=f"{query} paper {i} from {self.schema.name}",
                authors=[f"Author {i}"],
                abstract=f"This paper discusses {query} in depth. Paper number {i}.",
                source=self.schema.name,
                citations=i * 10,
            )
            for i in range(min(max_results, 3))
        ]
        return ToolResult(
            tool_name=self.schema.name,
            success=True,
            content=[p.model_dump() for p in papers],
        )


def test_end_to_end_chain():
    """验证 规划→检索→整理→生成 全链路（Mock 工具 + Mock LLM）。"""
    agent = PlannerAgent()
    # 注入 Mock LLM 与伪造工具
    agent.llm = _MockLLM()
    mcp = MCPClient()
    # 清空真实工具，注册伪造工具
    mcp._tools.clear()
    mcp.register_tool(_FakeSearchTool("search_openalex"))
    agent.mcp = mcp

    task = agent.run("attention mechanism", depth="basic", max_papers=6)

    assert task.status.value == "completed"
    assert len(task.papers) > 0, "应检索到文献"
    assert len(task.report) > 0, "应生成报告"
    # 验证记忆写入
    mem_results = agent.memory.search("attention", top_k=5)
    assert len(mem_results) > 0, "记忆应可召回"
    print(f"\n[OK] 端到端成功: papers={len(task.papers)}, report_len={len(task.report)}")
