"""数据模型单元测试。"""

from __future__ import annotations

from src.common.data_models import (
    KnowledgePoint,
    PaperMeta,
    Relation,
    ResearchDepth,
    ResearchPlan,
    ResearchTask,
    SubTask,
    TaskStatus,
)


def test_task_status_enum():
    assert TaskStatus.PENDING_PLANNING.value == "pending_planning"
    assert TaskStatus.COMPLETED.value == "completed"


def test_paper_meta_creation():
    p = PaperMeta(
        paper_id="p1",
        title="Test Paper",
        authors=["A", "B"],
        abstract="An abstract.",
        arxiv_id="1234.5678",
        source="arxiv",
    )
    assert p.citations == 0
    assert p.source == "arxiv"


def test_research_task_workflow():
    task = ResearchTask(task_id="t1", topic="RAG", depth=ResearchDepth.STANDARD)
    assert task.status == TaskStatus.PENDING_PLANNING
    plan = ResearchPlan(
        plan_id="pl1",
        topic="RAG",
        subtasks=[
            SubTask(subtask_id="st_1", objective="检索", deliverable="文献清单"),
        ],
        quality_score=0.8,
    )
    task.plan = plan
    task.status = TaskStatus.PENDING_SEARCH
    assert task.plan.subtasks[0].objective == "检索"


def test_knowledge_point_with_relations():
    kp = KnowledgePoint(kp_id="k1", name="Transformer", importance=0.9)
    kp.relations.append(
        Relation(from_id="k1", to_id="k2", type="references", confidence=0.95)
    )
    assert len(kp.relations) == 1
