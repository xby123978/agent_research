"""研究任务编排入口与状态定义。

对应架构 2.2 核心数据流：
规划 → 检索 → 整理 → 生成 → 沉淀
单 Agent 在 MVP 阶段串联完整链路。
"""

from __future__ import annotations

import uuid
from typing import Any

from ..common.config import get_env
from ..common.data_models import (
    MemoryItem,
    PaperMeta,
    ResearchDepth,
    ResearchTask,
    TaskStatus,
)
from ..common.logger import get_logger
from ..common.utils import stage_timeout
from ..memory.manager import MemoryManager, get_memory
from ..mcp_layer.client import MCPClient, get_mcp_client
from ..models.llm import LLMClient, get_llm
from ..planning.planner import Planner, get_planner

logger = get_logger(__name__)

SUMMARY_PROMPT = """你是一位学术研究助手。请根据以下文献信息，围绕主题「{topic}」撰写一份结构化研究摘要。

文献列表（共 {n} 篇）：
{papers}

请输出 Markdown 格式，包含：
1. 研究主题概述
2. 核心发现与知识点（分点列出）
3. 关键文献引用（标注标题与来源）
4. 研究局限与建议

控制在 800 字以内。
"""


class PlannerAgent:
    """研究规划师 Agent。

    MVP 阶段由该 Agent 统筹规划→检索→整理→生成全流程。
    阶段二将拆分为四角色协作。
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        planner: Planner | None = None,
        mcp: MCPClient | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.planner = planner or get_planner()
        self.mcp = mcp or get_mcp_client()
        self.memory = memory or get_memory()
        env = get_env()
        self.default_max_papers = int(env.get("default_max_papers", 20))

    def create_task(self, topic: str, depth: str | ResearchDepth = ResearchDepth.STANDARD) -> ResearchTask:
        """创建研究任务并初始化。"""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = ResearchTask(
            task_id=task_id,
            topic=topic,
            depth=ResearchDepth(depth) if isinstance(depth, str) else depth,
            status=TaskStatus.PENDING_PLANNING,
        )
        # 写入短期记忆
        self.memory.add(
            f"创建研究任务: {topic}",
            metadata={"task_id": task_id, "depth": task.depth.value},
            layer="short_term",
        )
        return task

    @stage_timeout(180)
    def plan(self, task: ResearchTask) -> ResearchTask:
        """规划阶段：生成研究计划（硬超时 180s）。

        时间预算：3 候选 × LLM(~6s) + OpenJev 首次加载(~7s) + JEV 评分(~6s) ≈ 30s
        重试 1 次 + 余量 = 180s
        """
        task.status = TaskStatus.PENDING_PLANNING
        task.plan = self.planner.create_plan(task.topic, task.depth)
        task.status = TaskStatus.PENDING_SEARCH
        task.updated_at = __import__("datetime").datetime.now()
        self.memory.add(
            f"研究计划已生成: {task.topic}，共 {len(task.plan.subtasks)} 个子任务",
            metadata={"task_id": task.task_id, "plan_id": task.plan.plan_id},
            layer="short_term",
        )
        logger.info(f"任务 {task.task_id} 规划完成")
        return task

    @stage_timeout(150)
    def search(self, task: ResearchTask, max_papers: int | None = None) -> ResearchTask:
        """检索阶段：调用 MCP 工具多源检索文献（硬超时 150s 兜底）。

        时间预算控制（避免国内网络慢导致 Gradio 长期无响应）：
        - 单工具单次最坏 20s 超时 × 3 次重试 ≈ 60s
        - 内部预算 120s，装饰器硬上限 150s 兜底
        """
        task.status = TaskStatus.PENDING_SEARCH
        max_papers = max_papers or self.default_max_papers
        # 按计划子任务的关键词检索，MVP 直接用主题 + 子任务目标
        queries = [task.topic] + [s.objective for s in (task.plan.subtasks if task.plan else [])]
        # 去重
        seen_queries = set()
        unique_queries = [q for q in queries if not (q in seen_queries or seen_queries.add(q))]  # type: ignore[func-returns-value]

        import time as _time

        search_start = _time.time()
        budget_seconds = 120  # OpenAlex 7s/次，3次约 21s，预算 2 分钟足够
        all_papers: list[PaperMeta] = []
        # OpenAlex 单次即可覆盖主题，最多用 2 个 query 做补充
        for q in unique_queries[:2]:
            elapsed = _time.time() - search_start
            if elapsed > budget_seconds:
                logger.warning(f"检索总预算 {budget_seconds}s 已耗尽（已用 {elapsed:.0f}s），停止后续检索")
                break
            logger.info(f"检索中: tool=search_openalex query={q[:30]} 已用 {elapsed:.0f}s")
            result = self.mcp.call_tool("search_openalex", {"query": q, "max_results": 10})
            if result.success:
                for p_dict in result.content:
                    try:
                        all_papers.append(PaperMeta(**p_dict))
                    except Exception as e:
                        logger.debug(f"文献解析跳过: {e}")
            else:
                logger.warning(f"工具 search_openalex 检索失败: {result.error}")

        # 去重（按标题）
        seen_titles: set[str] = set()
        task.papers = [
            p for p in all_papers if p.title and not (p.title in seen_titles or seen_titles.add(p.title))  # type: ignore[func-returns-value]
        ][:max_papers]
        task.status = TaskStatus.PENDING_FILTER
        task.updated_at = __import__("datetime").datetime.now()
        elapsed_total = _time.time() - search_start
        self.memory.add(
            f"检索完成: 命中 {len(task.papers)} 篇文献（用时 {elapsed_total:.0f}s）",
            metadata={"task_id": task.task_id, "papers": len(task.papers)},
            layer="short_term",
        )
        logger.info(
            f"任务 {task.task_id} 检索完成: {len(task.papers)} 篇 用时 {elapsed_total:.0f}s"
        )
        return task

    @stage_timeout(90)
    def organize(self, task: ResearchTask) -> ResearchTask:
        """整理阶段：将文献元数据沉淀到长期记忆（硬超时 90s，避免嵌入挂死）。"""
        task.status = TaskStatus.PENDING_ORGANIZING
        if task.papers:
            # 文献索引沉淀到长期记忆
            for p in task.papers[:10]:
                content = f"文献: {p.title}\n作者: {', '.join(p.authors[:3])}\n摘要: {p.abstract[:300]}"
                self.memory.add(content, metadata={"task_id": task.task_id, "paper_id": p.paper_id}, layer="long_term")
        task.status = TaskStatus.PENDING_OUTPUT
        task.updated_at = __import__("datetime").datetime.now()
        logger.info(f"任务 {task.task_id} 整理完成")
        return task

    @stage_timeout(90)
    def generate(self, task: ResearchTask) -> ResearchTask:
        """生成阶段：撰写研究摘要报告（硬超时 90s，避免 LLM 挂死）。"""
        task.status = TaskStatus.PENDING_OUTPUT
        papers_text = "\n".join(
            f"- [{p.source}] {p.title} ({p.publication_date}) [citations={p.citations}]"
            for p in task.papers[:15]
        )
        report = self.llm.chat(
            [
                {"role": "system", "content": "你是学术研究助手，输出 Markdown 报告。"},
                {"role": "user", "content": SUMMARY_PROMPT.format(
                    topic=task.topic, n=len(task.papers), papers=papers_text
                )},
            ],
            temperature=0.4,
        )
        task.report = report
        # 沉淀到长期记忆（写失败不影响主流程）
        try:
            self.memory.add(
                f"研究报告: {task.topic}\n{report[:500]}",
                metadata={"task_id": task.task_id, "type": "report"},
                layer="long_term",
            )
        except Exception as e:
            logger.warning(f"报告沉淀到长期记忆失败（不影响主流程）: {e}")
        task.status = TaskStatus.COMPLETED
        task.updated_at = __import__("datetime").datetime.now()
        task.quality_score = 0.8  # MVP 简化评分
        logger.info(f"任务 {task.task_id} 生成完成")
        return task

    def run(self, topic: str, depth: str | ResearchDepth = ResearchDepth.STANDARD, max_papers: int | None = None) -> ResearchTask:
        """端到端执行研究任务，串联完整链路。"""
        set_trace_id_if_needed = None
        from ..common.logger import set_trace_id

        trace_id = set_trace_id()
        logger.info(f"[{trace_id}] 启动研究任务: {topic}")
        task = self.create_task(topic, depth)
        task = self.plan(task)
        task = self.search(task, max_papers)
        task = self.organize(task)
        task = self.generate(task)
        logger.info(f"任务完成: {task.task_id} status={task.status.value}")
        return task


_agent_instance: PlannerAgent | None = None


def get_agent() -> PlannerAgent:
    """获取全局 Agent 单例。"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PlannerAgent()
    return _agent_instance
