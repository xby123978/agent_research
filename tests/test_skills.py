"""阶段二：Skills 引擎测试。"""

import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.common.data_models import PaperMeta
from src.skills.citation_manager import CitationManagerSkill
from src.skills.knowledge_card import KnowledgeCardSkill
from src.skills.literature_review import LiteratureReviewSkill


class TestCitationManagerSkill:
    """引用管理 Skill 测试。"""

    def _make_papers(self, n=5):
        """构造 n 篇测试文献。"""
        return [
            PaperMeta(
                paper_id=f"p{i}",
                title=f"测试论文{i}",
                authors=["张三", "李四"],
                publication_date="2024",
                venue="Nature",
                doi=f"10.1234/test{i}",
                citations=10 + i,
                source="openalex",
            )
            for i in range(n)
        ]

    def test_apa_format(self):
        skill = CitationManagerSkill()
        papers = self._make_papers(5)
        result = skill.run({"papers": [p.model_dump() for p in papers], "style": "apa"})
        assert result["quality_passed"]
        assert result["citation_count"] == 5
        bib = result["bibliography"][0]
        assert "张三" in bib
        assert "李四" in bib
        assert "2024" in bib
        assert "10.1234/test0" in bib

    def test_ieee_format(self):
        skill = CitationManagerSkill()
        papers = self._make_papers(5)
        result = skill.run({"papers": [p.model_dump() for p in papers], "style": "ieee"})
        assert result["quality_passed"]
        bib = result["bibliography"][0]
        assert "[1]" in bib
        assert "张三" in bib

    def test_empty_papers(self):
        skill = CitationManagerSkill()
        result = skill.run({"papers": [], "style": "apa"})
        assert result["citation_count"] == 0
        assert result["quality_score"] == 0.0


class TestKnowledgeCardSkill:
    """知识卡片 Skill 测试。"""

    def test_empty_papers(self):
        skill = KnowledgeCardSkill()
        result = skill.run({"topic": "测试", "papers": []})
        assert result["card_count"] == 0
        assert result["quality_score"] == 0.0


class TestLiteratureReviewSkill:
    """文献综述 Skill 测试（不实际请求网络，仅测试结构）。"""

    def test_missing_topic(self):
        skill = LiteratureReviewSkill()
        # 应该有 topic 字段
        import pytest
        with pytest.raises(KeyError):
            skill.execute({"max_papers": 10})
