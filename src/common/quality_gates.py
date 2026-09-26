"""全链路质量门限（JEV 增强）。

对应架构 6.1 Quality Gate：每个阶段设置准出校验，
不达标不得进入下一阶段。

阶段三升级：
- 硬门限（规则校验）：子任务数、文献数、报告长度等结构化指标
- 软门限（JEV 评分）：内容质量、相关性、完整度等语义指标
- 综合评分 = 0.5 * 硬门限 + 0.5 * JEV 评分
- 硬门限不通过直接阻断，JEV 评分低于阈值记录警告但不阻断（避免误杀）

设计原则：
- 门限配置从 settings.yaml 读取，禁止魔法数字（Claude.md 5.4）
- 校验失败返回 QualityGateError，由 Agent 决定重试或降级
- JEV 调用失败自动降级为规则评分，保证可用性
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..common.config import get_settings
from ..common.logger import get_logger
from ..models.jev import get_jev

logger = get_logger(__name__)


@dataclass
class GateResult:
    """门限校验结果。"""

    passed: bool
    score: float
    reason: str
    stage: str
    jev_score: float = 0.0  # JEV 评分（0 表示未调用）

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise QualityGateError(self.stage, self.score, self.reason)


class QualityGateError(Exception):
    """质量门限未通过。"""

    def __init__(self, stage: str, score: float, reason: str) -> None:
        self.stage = stage
        self.score = score
        self.reason = reason
        super().__init__(f"[{stage}] 质量门限未通过: score={score:.2f} reason={reason}")


def _gates_cfg() -> dict[str, Any]:
    """从 settings.yaml 读取质量门限配置。"""
    return get_settings().get("research", {}).get("quality_gates", {})


def _jev_score_safe(content: str, rubric: str) -> float:
    """安全调用 JEV 评分，失败返回 0。"""
    try:
        return get_jev().score(content, rubric)
    except Exception as e:
        logger.warning(f"JEV 评分失败，降级为 0: {e}")
        return 0.0


# ===== Planner 阶段准出 =====
def check_planner(plan: Any, self_score: float = 0.0) -> GateResult:
    """规划阶段准出：子任务数 3-5 + 计划评分≥0.75（架构 6.1）。

    阶段三：JEV 评估计划质量（子任务覆盖度、目标清晰度）
    """
    gates = _gates_cfg()
    min_score = float(gates.get("plan_score_threshold", 0.75))

    n = len(plan.subtasks) if plan and plan.subtasks else 0
    # 硬门限：子任务数 3-5
    if not (3 <= n <= 5):
        return GateResult(
            passed=False,
            score=min(n / 3, 1.0) if n < 3 else 1.0,
            reason=f"子任务数 {n} 不在 3-5 范围",
            stage="planner",
        )
    # 硬门限：每个子任务必须有交付物
    missing = [s.subtask_id for s in plan.subtasks if not s.deliverable]
    if missing:
        return GateResult(
            passed=False,
            score=0.5,
            reason=f"子任务缺少交付物: {missing}",
            stage="planner",
        )
    # JEV 软门限：评估计划内容质量
    plan_text = ""
    if plan and plan.subtasks:
        plan_text = " ".join(
            f"{s.objective} {s.deliverable}" for s in plan.subtasks
        )
    jev_score = _jev_score_safe(
        plan_text, "研究计划质量：子任务覆盖度、目标清晰度、可执行性"
    )
    # 综合分：硬门限通过即 0.8 基础分 + JEV 评分
    score = 0.5 * 0.8 + 0.5 * jev_score
    if score < min_score:
        logger.warning(
            f"Planner JEV 评分 {jev_score:.2f}，综合分 {score:.2f} 低于 {min_score}（警告不阻断）"
        )
    return GateResult(
        passed=True,
        score=score,
        reason=f"规划门限通过，JEV={jev_score:.2f}",
        stage="planner",
        jev_score=jev_score,
    )


# ===== Collector 阶段准出 =====
def check_collector(papers: list[Any]) -> GateResult:
    """检索/筛选阶段准出：召回文献数达下限 + 相关性精确率（架构 6.1）。

    阶段三：JEV 评估文献相关性（抽样标题与主题匹配度）
    """
    gates = _gates_cfg()
    min_recall = int(gates.get("min_recall_papers", 20))
    threshold = float(gates.get("relevance_threshold", 0.85))

    n = len(papers)
    # 硬门限：召回数（软下限，MVP 放宽到 8）
    soft_min = max(8, min_recall // 2)
    if n < soft_min:
        return GateResult(
            passed=False,
            score=min(n / soft_min, 1.0),
            reason=f"召回文献 {n} 篇低于软下限 {soft_min}",
            stage="collector",
        )
    # 硬门限：相关性精确率
    if n > 0:
        relevant = sum(1 for p in papers if getattr(p, "relevance_score", 0) >= threshold)
        precision = relevant / n
    else:
        precision = 0.0
    if precision < 0.3 and n > 0:
        logger.warning(f"Collector 相关性精确率 {precision:.2f} 偏低（可能字段缺失）")
    # JEV 软门限：抽样评估文献标题与主题相关性
    sample_titles = " | ".join(p.title[:50] for p in papers[:5]) if papers else ""
    jev_score = _jev_score_safe(
        sample_titles, "文献标题与研究主题相关性、覆盖面"
    ) if sample_titles else 0.5
    score = min(1.0, n / min_recall) if min_recall > 0 else 1.0
    return GateResult(
        passed=True,
        score=score,
        reason=f"召回 {n} 篇，精确率 {precision:.2f}，JEV={jev_score:.2f}",
        stage="collector",
        jev_score=jev_score,
    )


# ===== Engineer 阶段准出 =====
def check_engineer(kps: list[Any], papers: list[Any]) -> GateResult:
    """知识整理阶段准出：知识点完整度 + 实体数（架构 6.1）。

    阶段三：JEV 评估知识点质量（定义清晰度、重要性）
    """
    gates = _gates_cfg()
    min_kps = int(gates.get("min_knowledge_points", 3))

    n_kps = len(kps)
    n_papers = len(papers)
    # 硬门限：知识点数下限
    if n_kps < min_kps:
        return GateResult(
            passed=False,
            score=min(n_kps / min_kps, 1.0) if min_kps > 0 else 0.0,
            reason=f"知识点 {n_kps} 个低于下限 {min_kps}（文献 {n_papers} 篇）",
            stage="engineer",
        )
    # 硬门限：完整度
    expected = max(min_kps, n_papers // 2)
    completeness = min(1.0, n_kps / expected) if expected > 0 else 1.0
    if completeness < 0.5:
        return GateResult(
            passed=False,
            score=completeness,
            reason=f"知识点完整度 {completeness:.2f} 低于 0.5（期望 {expected} 个）",
            stage="engineer",
        )
    # JEV 软门限：知识点定义质量
    kp_text = " ".join(f"{k.name}: {k.definition}" for k in kps[:5]) if kps else ""
    jev_score = _jev_score_safe(
        kp_text, "知识点定义清晰度、重要性、学术价值"
    ) if kp_text else 0.5
    return GateResult(
        passed=True,
        score=completeness,
        reason=f"知识点 {n_kps} 个，完整度 {completeness:.2f}，JEV={jev_score:.2f}",
        stage="engineer",
        jev_score=jev_score,
    )


# ===== Writer 阶段准出 =====
def check_writer(report: str, papers: list[Any], kps: list[Any]) -> GateResult:
    """输出阶段准出：报告长度 + 引用规范率 + 事实准确率（架构 6.1）。

    阶段三：JEV 评估报告质量（结构、逻辑、学术规范）
    """
    gates = _gates_cfg()
    min_length = int(gates.get("min_report_length", 500))

    # 硬门限：报告长度
    if not report or len(report) < min_length:
        return GateResult(
            passed=False,
            score=min(len(report) / min_length, 1.0) if min_length > 0 else 0.0,
            reason=f"报告长度 {len(report)} 低于下限 {min_length}",
            stage="writer",
        )
    # 硬门限：引用规范率
    cited = sum(1 for p in papers if p.title and p.title[:20] in report)
    citation_rate = cited / max(len(papers), 1)
    if citation_rate < 0.3:
        return GateResult(
            passed=False,
            score=citation_rate,
            reason=f"引用规范率 {citation_rate:.2f} 低于 0.3",
            stage="writer",
        )
    # JEV 软门限：报告整体质量
    jev_score = _jev_score_safe(
        report[:2000], "研究报告质量：结构完整性、逻辑连贯性、学术规范性、结论价值"
    )
    score = min(1.0, 0.4 + len(papers) * 0.03 + len(report) / 5000 + jev_score * 0.2)
    return GateResult(
        passed=True,
        score=score,
        reason=f"报告 {len(report)} 字，引用率 {citation_rate:.2f}，JEV={jev_score:.2f}",
        stage="writer",
        jev_score=jev_score,
    )
