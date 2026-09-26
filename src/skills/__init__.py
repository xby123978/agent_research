"""Skills 引擎 - 高级能力封装。"""

from .argumentation_skill import ArgumentationSkill
from .base import Skill
from .citation_manager import CitationManagerSkill
from .knowledge_card import KnowledgeCardSkill
from .literature_review import LiteratureReviewSkill
from .review_skill import ReviewSkill

__all__ = [
    "Skill",
    "LiteratureReviewSkill",
    "KnowledgeCardSkill",
    "CitationManagerSkill",
    "ArgumentationSkill",
    "ReviewSkill",
]
