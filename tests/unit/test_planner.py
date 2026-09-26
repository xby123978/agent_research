"""规划模块测试（使用 Mock LLM，无 API Key）。"""

from __future__ import annotations

from src.planning.planner import Planner
from src.models.llm import _MockLLM


def test_planner_creates_plan(monkeypatch):
    planner = Planner()
    # 强制使用 Mock LLM
    planner.llm = _MockLLM()
    plan = planner.create_plan("Transformer 原理", "standard")
    assert len(plan.subtasks) >= 3
    assert all(s.deliverable for s in plan.subtasks)


def test_plan_subtasks_have_objectives():
    planner = Planner()
    planner.llm = _MockLLM()
    plan = planner.create_plan("RAG 优化", "basic")
    for s in plan.subtasks:
        assert s.objective
        assert "st_" in s.subtask_id
