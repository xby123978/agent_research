"""Engineer Agent - 知识工程师。

职责：从文献中抽取知识点与关系，沉淀到长期记忆。
对应架构 3.2.1 状态机 pending_reading → pending_organizing 阶段。
阶段二新增：用 LLM 抽取知识点，构建实体关系图。
"""

from __future__ import annotations

import uuid

from ..common.data_models import KnowledgePoint, PaperMeta, Relation, RelationType, TaskStatus
from ..common.logger import get_logger
from ..common.quality_gates import check_engineer
from ..orchestrator.shared_board import SharedBoard
from .base import BaseAgent

logger = get_logger(__name__)

# 知识点抽取 prompt
EXTRACT_PROMPT = """你是一位学术知识工程师。从以下文献信息中抽取核心知识点。

主题: {topic}

文献列表:
{papers}

请输出 JSON 数组，每个元素格式：
{{
  "name": "知识点名称（简短）",
  "definition": "定义说明（1-2句）",
  "category": "分类（method/concept/dataset/benchmark/other）",
  "importance": 0.0-1.0 的重要度,
  "source_paper_title": "来源论文标题"
}}

最多输出 8 个核心知识点，只输出 JSON，无其他文字。
"""


class EngineerAgent(BaseAgent):
    name = "EngineerAgent"

    def execute(self, board: SharedBoard) -> SharedBoard:
        task = board.task
        # 状态保护：仅 pending_reading/pending_organizing 可进入此节点
        # 回退到 pending_planning 时由编排器重新调度，不应执行知识点抽取
        current = board.get_task_state()
        if current not in (TaskStatus.PENDING_READING, TaskStatus.PENDING_ORGANIZING):
            logger.info(
                f"[{self.name}] 当前状态 {current.value} 非本节点职责，跳过执行"
            )
            # 若停在 planning，直接推进到 search 让编排器重新调度 Collector
            if current == TaskStatus.PENDING_PLANNING:
                self.transition_to(board, TaskStatus.PENDING_SEARCH)
            return board

        papers: list[PaperMeta] = board.read("papers", [])
        if not papers:
            board.log_stage(self.name, "无文献可整理", level="WARN")
            self.transition_to(board, TaskStatus.PENDING_OUTPUT)
            return board

        # 1. 文献沉淀到长期记忆（嵌入向量写入）
        for p in papers[:10]:
            content = (
                f"文献: {p.title}\n作者: {', '.join(p.authors[:3])}\n"
                f"摘要: {p.abstract[:300]}"
            )
            try:
                self.memory.add(
                    content,
                    metadata={"task_id": task.task_id, "paper_id": p.paper_id},
                    layer="long_term",
                )
            except Exception as e:
                logger.warning(f"文献 {p.paper_id} 沉淀失败: {e}")

        # 2. LLM 抽取知识点
        papers_text = "\n".join(
            f"- {p.title} (引用数={p.citations}) 摘要: {p.abstract[:200]}"
            for p in papers[:10]
        )
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": "你是学术知识工程师，输出 JSON。"},
                    {"role": "user", "content": EXTRACT_PROMPT.format(
                        topic=task.topic, papers=papers_text
                    )},
                ],
                temperature=0.3,
            )
            kps = self._parse_kps(raw, papers)
            board.write("knowledge_points", kps)
            board.log_stage(self.name, f"抽取知识点 {len(kps)} 个")
        except Exception as e:
            logger.warning(f"知识点抽取失败: {e}")
            board.write("knowledge_points", [])
            board.log_stage(self.name, f"知识点抽取失败: {e}", level="WARN")

        # 质量门限校验（架构 6.1）：知识点数不足时补齐占位，避免阻断下游
        gate = check_engineer(kps, papers)
        if not gate.passed:
            board.log_stage(self.name, f"质量门限未通过: {gate.reason}", level="WARN")
            # 降级：知识点数不足 min_kps 时，基于主题与文献标题补齐到 3 个
            from ..common.config import get_settings

            min_kps = int(
                get_settings().get("research", {})
                .get("quality_gates", {})
                .get("min_knowledge_points", 3)
            )
            # 用文献标题作为知识点来源（保证非空且相关）
            fallback_pool = [p.title for p in papers if p.title] or [task.topic]
            idx = 0
            while len(kps) < min_kps:
                src = fallback_pool[idx % len(fallback_pool)]
                kps.append(
                    KnowledgePoint(
                        kp_id=f"kp_fallback_{uuid.uuid4().hex[:6]}",
                        name=f"{task.topic} 相关方向 {len(kps)+1}",
                        definition=f"（基于「{src[:40]}」补充）{task.topic} 相关概念，待精化",
                        category="concept",
                        importance=0.5,
                    )
                )
                idx += 1
            board.write("knowledge_points", kps)
            board.log_stage(self.name, f"已补齐知识点至 {len(kps)} 个")
        else:
            board.log_stage(self.name, f"质量门限通过: {gate.reason}")

        # 3. 转移到生成阶段
        self.transition_to(board, TaskStatus.PENDING_OUTPUT)
        return board

    @staticmethod
    def _parse_kps(raw: str, papers: list[PaperMeta]) -> list[KnowledgePoint]:
        """解析 LLM 输出为 KnowledgePoint 列表，并匹配来源论文 id。"""
        import json
        import re

        # 尝试提取 JSON 数组
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group(0))
        except Exception:
            return []

        title_to_id = {p.title: p.paper_id for p in papers}
        kps: list[KnowledgePoint] = []
        for it in items[:8]:
            src_title = it.get("source_paper_title", "")
            source_ids = [title_to_id[src_title]] if src_title in title_to_id else []
            kps.append(
                KnowledgePoint(
                    kp_id=f"kp_{uuid.uuid4().hex[:8]}",
                    name=it.get("name", ""),
                    definition=it.get("definition", ""),
                    category=it.get("category", "other"),
                    importance=float(it.get("importance", 0.5)),
                    source_papers=source_ids,
                )
            )
        return kps
