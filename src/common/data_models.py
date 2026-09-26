"""核心数据模型定义。

对应架构第 4 节数据模型定义。使用 pydantic v2 实现，
包含 ResearchTask、ResearchPlan、PaperMeta、KnowledgePoint、Relation 等。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ===== 枚举 =====

class TaskStatus(str, Enum):
    """研究任务状态枚举，对应架构 4.2。"""

    PENDING_PLANNING = "pending_planning"
    PENDING_SEARCH = "pending_search"
    PENDING_FILTER = "pending_filter"
    PENDING_READING = "pending_reading"
    PENDING_ORGANIZING = "pending_organizing"
    PENDING_OUTPUT = "pending_output"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchDepth(str, Enum):
    """研究深度。"""

    BASIC = "basic"
    STANDARD = "standard"
    DEEP = "deep"


class RelationType(str, Enum):
    """实体关系类型，对应架构 4.1.4。"""

    INCLUDES = "includes"        # 包含
    REFERENCES = "references"     # 引用
    SIMILAR = "similar"           # 相似
    OPPOSES = "opposes"           # 对立
    APPLIES = "applies"           # 应用


# ===== 数据模型 =====

class SubTask(BaseModel):
    """研究子任务。"""

    subtask_id: str
    objective: str = Field(..., description="子任务目标")
    deliverable: str = Field(..., description="交付物")
    dependencies: list[str] = Field(default_factory=list, description="依赖的其他子任务 id")
    depth: ResearchDepth = ResearchDepth.STANDARD
    recommended_tools: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING_SEARCH


class ResearchPlan(BaseModel):
    """研究计划，对应架构 3.2 规划模块产出。"""

    plan_id: str
    topic: str
    subtasks: list[SubTask] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, description="JEV 计划评分")
    created_at: datetime = Field(default_factory=datetime.now)


class ResearchTask(BaseModel):
    """研究任务，对应架构 4.1.1。"""

    task_id: str
    topic: str
    depth: ResearchDepth = ResearchDepth.STANDARD
    status: TaskStatus = TaskStatus.PENDING_PLANNING
    plan: ResearchPlan | None = None
    current_step: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    quality_score: float = 0.0
    # 中间产物
    papers: list[PaperMeta] = Field(default_factory=list)
    knowledge_points: list[KnowledgePoint] = Field(default_factory=list)
    report: str = ""


class PaperMeta(BaseModel):
    """文献元数据，对应架构 4.1.2。"""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    doi: str = ""
    arxiv_id: str = ""
    publication_date: str = ""
    venue: str = ""
    citations: int = 0
    pdf_url: str = ""
    tags: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    quality_score: float = 0.0
    source: str = ""  # arxiv | semantic_scholar | crossref


class Relation(BaseModel):
    """实体关系，对应架构 4.1.4。"""

    from_id: str
    to_id: str
    type: RelationType = RelationType.REFERENCES
    confidence: float = 1.0
    description: str = ""


class KnowledgePoint(BaseModel):
    """知识点，对应架构 4.1.3。"""

    kp_id: str
    name: str
    definition: str = ""
    category: str = ""
    importance: float = 0.0
    source_papers: list[str] = Field(default_factory=list, description="来源论文 id")
    relations: list[Relation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class ToolCallRecord(BaseModel):
    """工具调用记录，用于短期记忆与可观测性。"""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    success: bool = True
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class MemoryItem(BaseModel):
    """记忆条目，统一短期/长期记忆的数据结构。"""

    item_id: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None
    score: float = 0.0
    source: str = ""  # short_term | long_term | working | preference
    created_at: datetime = Field(default_factory=datetime.now)


# 前向引用解析
ResearchTask.model_rebuild()
PaperMeta.model_rebuild()
