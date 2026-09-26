"""阶段三新增模块快速验证脚本。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 1. JEV 分级测试
from src.models.jev import get_jev
jev = get_jev()
print("JEV mode:", jev.mode)

r = jev.score_with_confidence("Transformer 架构核心原理与应用", "内容完整度")
print("score:", r.result, "conf:", r.confidence, "auto:", r.auto_execute, "review:", r.needs_review, "discard:", r.discarded)

# 6 类场景方法
tool = jev.select_tool("检索 Transformer 论文", ["search_crossref", "search_openalex"])
print("select_tool:", tool)

deepen = jev.should_deepen_subtopic("Transformer", "基础概念已覆盖")
print("should_deepen:", deepen)

sufficient = jev.is_recall_sufficient("Transformer", [{"content": "Transformer 是基于自注意力机制的神经网络架构"}])
print("is_recall_sufficient:", sufficient)

# 2. Skills 测试
from src.skills import ArgumentationSkill, ReviewSkill
print("ArgumentationSkill:", ArgumentationSkill.name)
print("ReviewSkill:", ReviewSkill.name)

# 3. CSL 工具测试
from src.tools.csl_tool import CSLTool
from src.common.data_models import PaperMeta

csl = CSLTool()
p = PaperMeta(
    paper_id="p1",
    title="Attention Is All You Need",
    authors=["Vaswani", "Shazeer"],
    publication_date="2017",
    doi="10.5555/3295222",
)
for style in ["apa", "mla", "chicago", "ieee", "gbt7714"]:
    r = csl.call(paper=p, style=style)
    citation = r.content[0]["citation"]
    print(style + ":", citation[:60])

# 4. 事实校验测试
from src.common.fact_check import verify_report
report = "Vaswani 等人在 2017 年提出了 Transformer 架构。该架构基于自注意力机制，彻底改变了 NLP 领域。"
result = verify_report(report, [p])
print("事实校验:", result.summary())
print("论断数:", len(result.claims), "单源:", len(result.single_source_claims), "一致性:", round(result.consistency_score, 2))

# 5. Obsidian 工具测试
from src.tools.obsidian_tool import ObsidianTool
obs = ObsidianTool()
r = obs.call(report=report, topic="Transformer", knowledge_points=[], papers=[p])
print("Obsidian:", r.success, "file:", r.content[0]["file"] if r.success else r.error)

print("\n[OK] 阶段三所有模块验证通过")
