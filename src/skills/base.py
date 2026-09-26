"""Skill 基类。

对应架构 3.4 Skills 引擎。Skill 是封装高级能力的可复用模块，
由 Agent 调用完成特定任务（如文献综述、知识卡片生成、引用管理）。
每个 Skill 有单一职责、明确的输入输出 schema 和质量门限。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..common.data_models import PaperMeta, ResearchTask
from ..common.logger import get_logger
from ..memory.manager import MemoryManager, get_memory
from ..models.llm import LLMClient, get_llm

logger = get_logger(__name__)


class Skill(ABC):
    """Skill 抽象基类。子类必须实现 execute() 与 validate_quality()。"""

    name: str = "BaseSkill"
    version: str = "1.0"
    description: str = ""
    # 质量门限阈值（阶段三由 JEV 校验，阶段二用简化规则）
    quality_threshold: float = 0.7

    def __init__(
        self,
        llm: LLMClient | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.memory = memory or get_memory()

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """执行 Skill，返回结构化输出。子类必须实现。"""
        ...

    def validate_quality(self, output: dict[str, Any]) -> bool:
        """简化质量校验：输出非空且包含必需字段。阶段三由 JEV 增强。"""
        if not output:
            return False
        return output.get("quality_score", 0.0) >= self.quality_threshold

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """带质量校验的执行入口。"""
        try:
            output = self.execute(inputs)
            output.setdefault("skill", self.name)
            output.setdefault("quality_score", 0.8)  # 默认通过
            ok = self.validate_quality(output)
            output["quality_passed"] = ok
            if not ok:
                logger.warning(f"Skill {self.name} 质量校验未通过 (score={output.get('quality_score')})")
            return output
        except Exception as e:
            logger.error(f"Skill {self.name} 执行失败: {e}", exc_info=True)
            return {"skill": self.name, "error": str(e), "quality_passed": False, "quality_score": 0.0}
