"""Collector Agent - 文献收集与筛选员。

职责：根据计划子任务多源检索文献，去重，相关性筛选。
对应架构 3.2.1 状态机 pending_search → pending_filter 阶段。
"""

from __future__ import annotations

import time as _time

from ..common.config import get_env
from ..common.data_models import PaperMeta, TaskStatus
from ..common.logger import get_logger
from ..common.quality_gates import check_collector
from ..mcp.client import MCPClient, get_mcp_client
from ..orchestrator.shared_board import SharedBoard
from .base import BaseAgent

logger = get_logger(__name__)


class CollectorAgent(BaseAgent):
    name = "CollectorAgent"

    def __init__(self, mcp: MCPClient | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mcp = mcp or get_mcp_client()
        env = get_env()
        self.default_max_papers = int(env.get("default_max_papers", 20))

    def execute(self, board: SharedBoard) -> SharedBoard:
        task = board.task
        # 状态保护：仅 pending_search/pending_filter 可进入此节点
        current = board.get_task_state()
        if current not in (TaskStatus.PENDING_SEARCH, TaskStatus.PENDING_FILTER):
            logger.info(
                f"[{self.name}] 当前状态 {current.value} 非本节点职责，跳过执行"
            )
            if current == TaskStatus.PENDING_PLANNING:
                self.transition_to(board, TaskStatus.PENDING_SEARCH)
            return board

        plan = board.read("plan")
        if plan is None:
            board.log_stage(self.name, "无计划可用，跳过检索", level="WARN")
            self.transition_to(board, TaskStatus.PENDING_FILTER)
            return board

        # 1. 构造检索 query：主题 + 各子任务目标
        queries = [task.topic] + [s.objective for s in plan.subtasks]
        seen_queries: set[str] = set()
        unique_queries = [q for q in queries if not (q in seen_queries or seen_queries.add(q))]
        # 最多 2 个 query（OpenAlex 单次覆盖好，避免过多请求）
        unique_queries = unique_queries[:2]

        board.log_stage(self.name, f"开始检索，共 {len(unique_queries)} 个 query")

        # 2. 多源检索（CrossRef 为主源无日配额限制，OpenAlex 补充）
        search_start = _time.time()
        budget_seconds = 120
        all_papers: list[PaperMeta] = []
        for q in unique_queries:
            elapsed = _time.time() - search_start
            if elapsed > budget_seconds:
                logger.warning(f"检索预算 {budget_seconds}s 已耗尽（已用 {elapsed:.0f}s）")
                break
            logger.info(f"检索中: query={q[:30]} 已用 {elapsed:.0f}s")
            # 主源：CrossRef（无日配额限制，国内访问稳定）
            result = self.mcp.call_tool("search_crossref", {"query": q, "max_results": 10})
            if result.success and result.content:
                for p_dict in result.content:
                    try:
                        all_papers.append(PaperMeta(**p_dict))
                    except Exception as e:
                        logger.debug(f"CrossRef 文献解析跳过: {e}")
            else:
                logger.warning(f"CrossRef 检索失败: {result.error}，切到 OpenAlex 兜底")
                # 兜底源：OpenAlex（有日配额限制，可能 429）
                fb = self.mcp.call_tool(
                    "search_openalex", {"query": q, "max_results": 10}
                )
                if fb.success and fb.content:
                    for p_dict in fb.content:
                        try:
                            all_papers.append(PaperMeta(**p_dict))
                        except Exception as e:
                            logger.debug(f"OpenAlex 文献解析跳过: {e}")
                else:
                    logger.warning(f"OpenAlex 兜底也失败: {fb.error}")

        # 3. 去重（按标题）
        seen_titles: set[str] = set()
        deduped = [
            p for p in all_papers
            if p.title and not (p.title in seen_titles or seen_titles.add(p.title))
        ]
        # 4. 相关性初筛：按引用数排序，保留 top N
        deduped.sort(key=lambda p: p.citations, reverse=True)
        filtered = deduped[: self.default_max_papers]

        board.write("papers", filtered)
        elapsed_total = _time.time() - search_start
        self.memory.add(
            f"检索完成: 命中 {len(filtered)} 篇文献（用时 {elapsed_total:.0f}s）",
            metadata={"task_id": task.task_id, "papers": len(filtered)},
            layer="short_term",
        )
        board.log_stage(
            self.name,
            f"检索完成: {len(filtered)} 篇 用时 {elapsed_total:.0f}s",
        )
        # 质量门限校验（架构 6.1）：未通过记录但放行（避免单源检索无果阻断全链路）
        gate = check_collector(filtered)
        if not gate.passed:
            board.log_stage(self.name, f"质量门限未通过: {gate.reason}", level="WARN")
        else:
            board.log_stage(self.name, f"质量门限通过: {gate.reason}")
        # 转移到精读阶段（阶段二：filter → reading）
        self.transition_to(board, TaskStatus.PENDING_READING)
        return board
