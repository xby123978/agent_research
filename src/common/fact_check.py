"""事实校验与幻觉防控（架构 6.4）。

四项核心能力：
1. 双重来源校验：核心结论必须至少有 2 篇独立文献支持，
   单篇文献的观点标注为"单源说法"
2. 引用溯源：所有引用观点必须标注对应文献 DOI 与页码
3. 幻觉检测：输出前调用 JEV 进行事实一致性评分，低于 0.9 触发重写
4. 人工复核入口：关键结论支持人工标记校验

接入点：WriterAgent 输出报告后调用 verify_report()，
不达标返回修改建议，由 Writer 重写或标记。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..common.data_models import PaperMeta
from ..common.logger import get_logger
from ..models.jev import get_jev

logger = get_logger(__name__)


@dataclass
class Claim:
    """报告中的单个论断。"""

    text: str  # 论断原文
    source_papers: list[str] = field(default_factory=list)  # 支持文献 paper_id
    has_doi: bool = False  # 是否标注了 DOI
    is_single_source: bool = False  # 是否单源说法
    needs_human_review: bool = False  # 是否需人工复核


@dataclass
class FactCheckResult:
    """事实校验结果。"""

    passed: bool
    consistency_score: float  # JEV 事实一致性评分
    claims: list[Claim] = field(default_factory=list)
    single_source_claims: list[Claim] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)  # 缺引用的论断
    suggestions: list[str] = field(default_factory=list)
    needs_rewrite: bool = False  # 是否需要重写

    def summary(self) -> str:
        return (
            f"事实校验: {'通过' if self.passed else '未通过'} | "
            f"一致性={self.consistency_score:.2f} | "
            f"论断数={len(self.claims)} | "
            f"单源={len(self.single_source_claims)} | "
            f"缺引用={len(self.missing_citations)}"
        )


def verify_report(report: str, papers: list[PaperMeta]) -> FactCheckResult:
    """报告事实校验主入口（架构 6.4）。

    Args:
        report: 生成的 Markdown 报告
        papers: 引用的文献列表
    Returns:
        FactCheckResult 含校验结果与修改建议
    """
    claims = _extract_claims(report, papers)

    # 1. 双重来源校验
    single_source = [c for c in claims if c.is_single_source]

    # 2. 引用溯源校验
    missing_citations = [c.text[:50] for c in claims if not c.has_doi]

    # 3. JEV 幻觉检测：事实一致性评分
    consistency_score = _jev_fact_consistency(report, papers)

    # 4. 生成修改建议
    suggestions: list[str] = []
    if single_source:
        suggestions.append(
            f"发现 {len(single_source)} 个单源说法，建议补充第二来源文献支持"
        )
    if missing_citations:
        suggestions.append(
            f"发现 {len(missing_citations)} 个论断缺 DOI 引用，建议补全引用溯源"
        )

    # 是否需要重写：一致性 < 0.9 或缺引用过多
    needs_rewrite = consistency_score < 0.9 or len(missing_citations) > len(claims) * 0.3

    passed = consistency_score >= 0.9 and not needs_rewrite

    return FactCheckResult(
        passed=passed,
        consistency_score=consistency_score,
        claims=claims,
        single_source_claims=single_source,
        missing_citations=missing_citations,
        suggestions=suggestions,
        needs_rewrite=needs_rewrite,
    )


def _extract_claims(report: str, papers: list[PaperMeta]) -> list[Claim]:
    """从报告中提取论断并匹配支持文献。

    简化实现：
    - 按句子切分（。.！!？?）
    - 过滤过短句子（<10 字）
    - 匹配文献标题关键词判断该句是否有文献支持
    """
    if not report:
        return []
    # 按句子切分
    import re

    sentences = re.split(r"[。.！!？?\n]+", report)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]

    paper_titles = {p.title[:20]: p for p in papers if p.title}
    paper_dois = {p.doi: p for p in papers if p.doi}

    claims: list[Claim] = []
    for sent in sentences:
        # 匹配文献：标题前 20 字出现在句子中，或 DOI 出现
        matched_papers: list[str] = []
        has_doi = False
        for title_prefix, p in paper_titles.items():
            if title_prefix and title_prefix in sent:
                matched_papers.append(p.paper_id)
                if p.doi and p.doi in sent:
                    has_doi = True
        for doi, p in paper_dois.items():
            if doi and doi in sent:
                has_doi = True
                if p.paper_id not in matched_papers:
                    matched_papers.append(p.paper_id)

        is_single = len(matched_papers) < 2
        claims.append(
            Claim(
                text=sent,
                source_papers=matched_papers,
                has_doi=has_doi,
                is_single_source=is_single,
            )
        )
    return claims


def _jev_fact_consistency(report: str, papers: list[PaperMeta]) -> float:
    """JEV 事实一致性评分（架构 6.4 幻觉检测）。

    用 JEV 评估报告与文献的一致性，<0.9 触发重写。
    """
    if not report or not papers:
        return 0.5
    try:
        # 构造上下文：文献标题与摘要
        ref_context = " | ".join(
            f"{p.title}: {p.abstract[:100]}" if p.abstract else p.title
            for p in papers[:5]
        )
        content = f"参考文献：\n{ref_context}\n\n报告内容：\n{report[:1500]}"
        score = get_jev().score(
            content,
            "事实一致性：报告论断是否与参考文献一致，是否存在幻觉或捏造内容",
        )
        logger.info(f"JEV 事实一致性评分: {score:.2f}")
        return score
    except Exception as e:
        logger.warning(f"JEV 事实校验失败，用默认 0.7: {e}")
        return 0.7


def mark_for_human_review(claim_text: str, reason: str = "") -> None:
    """人工复核入口：标记某论断需人工校验。

    架构 6.4.4：关键结论支持人工标记校验，校验结果反馈优化后续输出。
    """
    logger.info(
        f"人工复核标记: {claim_text[:50]}..."
        + (f" 原因: {reason}" if reason else "")
    )
