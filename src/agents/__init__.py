"""多 Agent 编排层。

对应架构 3.1 多 Agent 编排层。MVP 阶段为单 Agent（研究规划师兼任全流程），
阶段二升级为四 Agent 协作架构。
"""
from .planner_agent import PlannerAgent, get_agent

__all__ = ["PlannerAgent", "get_agent"]
