"""基准评测集。

对应架构 6.2.3 与 12.1：10 个标准研究任务，覆盖不同类型与难度。
每次代码迭代后跑通全量测试，确保核心指标不退化。

验收指标（架构 9.2）：
- 端到端任务成功率 ≥ 90%
- 文献检索召回数 ≥ 下限
- 报告完整度 ≥ 0.85
- 3 个 Skill 成功率 ≥ 95%

运行：python -m pytest tests/benchmark/ -v --tb=short
注意：需配置 LLM_API_KEY，否则用 Mock LLM 跑通结构。
"""

from __future__ import annotations

import pytest

# 标准评测任务（架构 12.1 示例 + 扩展到 10 个）
BENCHMARK_TASKS = [
    {"topic": "Transformer 架构核心原理", "type": "基础概念", "min_kps": 3, "min_papers": 5},
    {"topic": "RAG 优化技术最新进展", "type": "技术综述", "min_kps": 3, "min_papers": 5},
    {"topic": "Agent 记忆系统优化方案", "type": "前沿研究", "min_kps": 3, "min_papers": 5},
    {"topic": "大模型在安全工程中的应用", "type": "跨领域", "min_kps": 3, "min_papers": 5},
    {"topic": "多模态大模型训练方法", "type": "技术综述", "min_kps": 3, "min_papers": 5},
    {"topic": "强化学习在机器人控制中的应用", "type": "跨领域", "min_kps": 3, "min_papers": 5},
    {"topic": "图神经网络核心原理", "type": "基础概念", "min_kps": 3, "min_papers": 5},
    {"topic": "扩散模型图像生成技术", "type": "前沿研究", "min_kps": 3, "min_papers": 5},
    {"topic": "知识图谱构建方法综述", "type": "技术综述", "min_kps": 3, "min_papers": 5},
    {"topic": "联邦学习隐私保护方案", "type": "前沿研究", "min_kps": 3, "min_papers": 5},
]


@pytest.mark.parametrize("task_spec", BENCHMARK_TASKS, ids=lambda t: t["topic"][:20])
def test_benchmark_task(task_spec):
    """单个基准任务：端到端跑通并校验产出。

    Mock LLM 模式（未配置 LLM_API_KEY）下放宽质量评分断言，
    仅校验流程结构与产出非空；真实 LLM 模式下校验质量门槛。
    """
    import os

    from src.orchestrator import get_orchestrator

    use_mock = not os.environ.get("LLM_API_KEY")
    orch = get_orchestrator()
    task = orch.run(task_spec["topic"], depth="basic")
    assert task.status.value == "completed", f"任务未完成: {task.status.value}"
    # 文献召回数下限
    assert len(task.papers) >= task_spec["min_papers"], (
        f"文献数 {len(task.papers)} < 下限 {task_spec['min_papers']}"
    )
    # 知识点数下限
    assert len(task.knowledge_points) >= task_spec["min_kps"], (
        f"知识点数 {len(task.knowledge_points)} < 下限 {task_spec['min_kps']}"
    )
    # 报告非空（Mock 模式放宽到 50 字，真实模式 200 字）
    min_report_len = 50 if use_mock else 200
    assert task.report and len(task.report) > min_report_len, (
        f"报告为空或过短（{len(task.report) if task.report else 0} 字）"
    )
    # 质量评分：Mock 模式只要求 >0，真实模式要求 >=0.5
    min_score = 0.01 if use_mock else 0.5
    assert task.quality_score >= min_score, (
        f"质量评分 {task.quality_score} 过低（门限 {min_score}）"
    )


def test_benchmark_success_rate():
    """整体成功率指标（架构 9.1: ≥90%）。

    跑完全部任务后统计成功率，单独运行避免重复执行。
    """
    from src.common.observability import get_observability

    metrics = get_observability().get_metrics_summary()
    # 仅当有历史任务时校验，首次运行允许跳过
    if metrics["tasks"]["total"] == 0:
        pytest.skip("无历史任务，跳过成功率校验")
    success_rate = metrics["tasks"]["success_rate"]
    assert success_rate >= 0.9, f"任务成功率 {success_rate} 低于 0.9"


def test_metrics_summary():
    """指标聚合可用性测试（架构 6.5）。"""
    from src.common.observability import get_observability

    summary = get_observability().get_metrics_summary()
    assert "tasks" in summary
    assert "tools" in summary
    assert "total" in summary["tasks"]
    assert "success_rate" in summary["tasks"]


def test_jev_engine_available():
    """JEV 决策引擎可用性测试（阶段三核心）。

    校验三类任务都能正常调用并返回正确类型。
    """
    from src.models.jev import get_jev

    jev = get_jev()
    # 评分任务
    score = jev.score("这是一段关于 Transformer 的技术介绍", "内容完整度")
    assert 0.0 <= score <= 1.0, f"JEV score 越界: {score}"
    # 分类任务
    cat = jev.classify("关于强化学习的论文", ["NLP", "CV", "RL", "其他"])
    assert cat in ["NLP", "CV", "RL", "其他"], f"JEV classify 异常: {cat}"
    # 判断任务
    result = jev.judge("这是一段足够长的内容用于判断是否满足要求", "内容非空且大于50字")
    assert isinstance(result, bool), f"JEV judge 应返回 bool，实际 {type(result)}"


def test_jev_quality_gate_integration():
    """JEV 质量门限集成测试（阶段三核心）。

    校验质量门限的 GateResult 包含 jev_score 字段。
    """
    from src.common.data_models import PaperMeta
    from src.common.quality_gates import check_collector, check_writer

    papers = [
        PaperMeta(
            paper_id=f"p{i}",
            title=f"Transformer Paper {i}",
            authors=["Author"],
            abstract="Abstract",
        )
        for i in range(8)
    ]
    result = check_collector(papers)
    assert hasattr(result, "jev_score"), "GateResult 缺 jev_score 字段"
    assert result.jev_score >= 0.0, "jev_score 应 >= 0"

    # Writer 门限（Mock 短报告）
    short_report = "这是一段简短的报告，不足以通过完整门限校验。" * 20
    w_result = check_writer(short_report, papers, [])
    assert hasattr(w_result, "jev_score"), "Writer GateResult 缺 jev_score"


def test_skill_literature_review():
    """Skill 成功率测试（架构 9.2: ≥95%）。"""
    from src.skills.literature_review import LiteratureReviewSkill

    skill = LiteratureReviewSkill()
    result = skill.run({"topic": "Transformer", "max_papers": 5})
    assert result.get("quality_passed"), f"文献综述 Skill 质量未通过: {result}"


def test_skill_knowledge_card():
    """知识卡片 Skill 测试。"""
    from src.skills.knowledge_card import KnowledgeCardSkill

    skill = KnowledgeCardSkill()
    # 无文献时降级返回空，仍应通过质量校验（score=0）
    result = skill.run({"topic": "RAG", "papers": []})
    assert "quality_passed" in result


def test_skill_citation_manager():
    """引用管理 Skill 测试。"""
    from src.common.data_models import PaperMeta
    from src.skills.citation_manager import CitationManagerSkill

    papers = [
        PaperMeta(
            paper_id=f"p{i}",
            title=f"Test Paper {i}",
            authors=[f"Author {i}"],
            publication_date="2023",
            venue="Test Venue",
            doi=f"10.1000/{i}",
            citations=10,
        )
        for i in range(5)
    ]
    skill = CitationManagerSkill()
    result = skill.run({"papers": [p.model_dump() for p in papers], "style": "apa"})
    assert result.get("quality_passed"), f"引用管理 Skill 质量未通过: {result}"
    assert len(result.get("bibliography", [])) == 5
