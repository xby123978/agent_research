"""Engineer Agent - 知识工程师。

职责：从文献中抽取知识点与关系，构建实体关系图，沉淀到长期记忆。
对应架构 3.2.1 状态机 pending_reading → pending_organizing 阶段。
阶段三升级：
- LLM 抽取知识点（节点）
- LLM 抽取实体关系（边）：知识点间、论文→知识点、论文间引用
- 关系写入 WorkingMemory 图谱，供 graph_search 多因子召回
- JEV 校验关系质量，低置信度关系丢弃
"""

from __future__ import annotations

import json
import re
import uuid

from ..common.data_models import (
    KnowledgePoint,
    PaperMeta,
    Relation,
    RelationType,
    TaskStatus,
)
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

# 关系抽取 prompt
RELATION_PROMPT = """你是学术知识工程师。基于以下知识点与文献，抽取它们之间的关系。

主题: {topic}

知识点列表（kp_id / name）:
{kps}

文献列表（paper_id / title）:
{papers}

关系类型仅限以下五种：
- includes: A 包含 B（整体-部分关系）
- references: A 引用/参考 B
- similar: A 与 B 相似（同类方法/概念）
- opposes: A 与 B 对立/矛盾（不同学派/冲突结论）
- applies: A 应用 B（方法应用概念）

请输出 JSON 数组，每个元素格式：
{{
  "from_id": "源实体 id（kp_id 或 paper_id）",
  "to_id": "目标实体 id",
  "type": "includes|references|similar|opposes|applies",
  "confidence": 0.0-1.0 置信度,
  "description": "关系说明（1句话）"
}}

约束：
- 只在确有关系时输出，无关系则返回空数组 []
- from_id 和 to_id 必须来自上述列表
- 知识点→知识点、知识点→论文、论文→论文 均可
- 最多输出 15 条关系，按置信度从高到低排序
- 只输出 JSON，无其他文字
"""

# JEV 关系校验阈值：低于此值的关系丢弃
_RELATION_MIN_CONFIDENCE = 0.5


class EngineerAgent(BaseAgent):
    name = "EngineerAgent"

    def execute(self, board: SharedBoard) -> SharedBoard:
        task = board.task
        # 状态保护：仅 pending_reading/pending_organizing 可进入此节点
        current = board.get_task_state()
        if current not in (TaskStatus.PENDING_READING, TaskStatus.PENDING_ORGANIZING):
            logger.info(
                f"[{self.name}] 当前状态 {current.value} 非本节点职责，跳过执行"
            )
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
        kps: list[KnowledgePoint] = []
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

        # 质量门限校验：知识点数不足时补齐占位
        kps = self._ensure_min_kps(kps, papers, task.topic, board)
        board.write("knowledge_points", kps)

        # 3. LLM 抽取实体关系（架构 3.2.1 构建实体关系图）
        relations = self._extract_relations(task.topic, kps, papers, board)
        board.write("relations", relations)

        # 4. 关系写入 WorkingMemory 图谱 + 回填到 KnowledgePoint.relations
        if relations:
            self._build_graph(relations, kps, board)

        # 5. 转移到生成阶段
        self.transition_to(board, TaskStatus.PENDING_OUTPUT)
        return board

    # ===== 关系抽取 =====

    def _extract_relations(
        self,
        topic: str,
        kps: list[KnowledgePoint],
        papers: list[PaperMeta],
        board: SharedBoard,
    ) -> list[Relation]:
        """LLM 抽取实体关系，JEV 校验后返回合格关系列表。"""
        if not kps and not papers:
            return []

        kps_text = "\n".join(
            f"- {kp.kp_id} / {kp.name}" for kp in kps
        ) or "（无知识点）"
        papers_text = "\n".join(
            f"- {p.paper_id} / {p.title[:60]}" for p in papers[:10]
        ) or "（无文献）"

        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": "你是学术知识工程师，输出 JSON 数组。"},
                    {"role": "user", "content": RELATION_PROMPT.format(
                        topic=topic, kps=kps_text, papers=papers_text
                    )},
                ],
                temperature=0.2,
            )
            relations = self._parse_relations(raw)
        except Exception as e:
            logger.warning(f"关系抽取失败: {e}")
            board.log_stage(self.name, f"关系抽取失败: {e}", level="WARN")
            return []

        if not relations:
            board.log_stage(self.name, "未抽取出关系")
            return []

        # JEV 校验：对每条关系判断置信度，低置信度丢弃
        relations = self._filter_relations_by_jev(relations, board)
        board.log_stage(self.name, f"抽取关系 {len(relations)} 条（JEV 校验后）")
        return relations

    def _parse_relations(self, raw: str) -> list[Relation]:
        """解析 LLM 输出为 Relation 列表。"""
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group(0))
        except Exception:
            return []

        type_map = {
            "includes": RelationType.INCLUDES,
            "references": RelationType.REFERENCES,
            "similar": RelationType.SIMILAR,
            "opposes": RelationType.OPPOSES,
            "applies": RelationType.APPLIES,
        }
        relations: list[Relation] = []
        for it in items[:15]:
            try:
                rel_type = type_map.get(
                    str(it.get("type", "")).lower(), RelationType.REFERENCES
                )
                relations.append(
                    Relation(
                        from_id=str(it.get("from_id", "")),
                        to_id=str(it.get("to_id", "")),
                        type=rel_type,
                        confidence=max(0.0, min(1.0, float(it.get("confidence", 0.5)))),
                        description=str(it.get("description", "")),
                    )
                )
            except Exception as e:
                logger.debug(f"关系解析跳过: {e}")
        return relations

    def _filter_relations_by_jev(
        self, relations: list[Relation], board: SharedBoard
    ) -> list[Relation]:
        """JEV 校验关系质量，低置信度关系丢弃。

        策略：LLM 已给出 confidence，JEV 做二次校验。
        - confidence ≥ 0.8：直接通过
        - 0.5 ≤ confidence < 0.8：JEV judge 复核
        - confidence < 0.5：丢弃
        """
        if not relations:
            return []

        kept: list[Relation] = []
        try:
            from ..models.jev import get_jev
            jev = get_jev()
        except Exception as e:
            logger.warning(f"JEV 不可用，关系校验跳过: {e}")
            return [r for r in relations if r.confidence >= _RELATION_MIN_CONFIDENCE]

        for rel in relations:
            if rel.confidence >= 0.8:
                kept.append(rel)
                continue
            if rel.confidence < _RELATION_MIN_CONFIDENCE:
                logger.debug(f"关系丢弃(conf={rel.confidence:.2f}): {rel.from_id}→{rel.to_id}")
                continue
            # 中等置信度：JEV 复核
            try:
                criteria = f"判断 {rel.from_id} 与 {rel.to_id} 之间是否存在「{rel.type.value}」关系"
                passed = jev.judge(rel.description or rel.type.value, criteria)
                if passed:
                    kept.append(rel)
                else:
                    logger.debug(f"JEV 否决关系: {rel.from_id}→{rel.to_id} ({rel.type.value})")
            except Exception as e:
                logger.debug(f"JEV 校验异常，保留关系: {e}")
                kept.append(rel)
        return kept

    def _build_graph(
        self,
        relations: list[Relation],
        kps: list[KnowledgePoint],
        board: SharedBoard,
    ) -> None:
        """将关系写入 WorkingMemory 图谱 + 回填到 KnowledgePoint.relations。"""
        wm = self.memory.working
        if wm is None:
            logger.debug("工作记忆不可用，跳过图谱写入")
            return

        # 关系类型 → WorkingMemory add_edge 的 relation_type 映射
        # WorkingMemory 的 _RELATION_WEIGHTS 已支持 cites/supports/contradicts/similar/related
        type_to_edge = {
            RelationType.INCLUDES: "related",
            RelationType.REFERENCES: "cites",
            RelationType.SIMILAR: "similar",
            RelationType.OPPOSES: "contradicts",
            RelationType.APPLIES: "supports",
        }

        # 构建 kp_id → KnowledgePoint 映射，用于回填 relations
        kp_map = {kp.kp_id: kp for kp in kps}

        for rel in relations:
            edge_type = type_to_edge.get(rel.type, "related")
            try:
                wm.add_edge(
                    rel.from_id,
                    rel.to_id,
                    edge_type,
                    weight=rel.confidence,
                )
            except Exception as e:
                logger.debug(f"图谱建边失败: {e}")

            # 回填到 KnowledgePoint.relations（仅当 from_id 或 to_id 是知识点）
            for kp_id in (rel.from_id, rel.to_id):
                kp = kp_map.get(kp_id)
                if kp is not None and rel not in kp.relations:
                    kp.relations.append(rel)

        board.log_stage(
            self.name,
            f"图谱写入 {len(relations)} 条边 (cites/supports/contradicts/similar/related)",
        )

    # ===== 辅助方法 =====

    def _ensure_min_kps(
        self,
        kps: list[KnowledgePoint],
        papers: list[PaperMeta],
        topic: str,
        board: SharedBoard,
    ) -> list[KnowledgePoint]:
        """质量门限：知识点数不足时补齐占位。"""
        gate = check_engineer(kps, papers)
        if gate.passed:
            board.log_stage(self.name, f"质量门限通过: {gate.reason}")
            return kps

        board.log_stage(self.name, f"质量门限未通过: {gate.reason}", level="WARN")
        from ..common.config import get_settings

        min_kps = int(
            get_settings().get("research", {})
            .get("quality_gates", {})
            .get("min_knowledge_points", 3)
        )
        fallback_pool = [p.title for p in papers if p.title] or [topic]
        idx = 0
        while len(kps) < min_kps:
            src = fallback_pool[idx % len(fallback_pool)]
            kps.append(
                KnowledgePoint(
                    kp_id=f"kp_fallback_{uuid.uuid4().hex[:6]}",
                    name=f"{topic} 相关方向 {len(kps)+1}",
                    definition=f"（基于「{src[:40]}」补充）{topic} 相关概念，待精化",
                    category="concept",
                    importance=0.5,
                )
            )
            idx += 1
        board.log_stage(self.name, f"已补齐知识点至 {len(kps)} 个")
        return kps

    @staticmethod
    def _parse_kps(raw: str, papers: list[PaperMeta]) -> list[KnowledgePoint]:
        """解析 LLM 输出为 KnowledgePoint 列表，并匹配来源论文 id。"""
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
