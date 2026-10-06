"""Gradio 交互界面。

对应架构 2.1 用户交互层：任务输入、进度展示、结果输出、人工干预入口。
阶段二：四 Agent 协作 + 知识点展示 + 四层记忆可视化。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 将 src 加入路径，保证直接运行时可用
SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.orchestrator import get_orchestrator
from src.common.config import ensure_dirs, get_env
from src.common.logger import get_logger, set_trace_id

logger = get_logger(__name__)


def _format_papers(papers) -> str:
    """格式化文献列表为 Markdown。"""
    if not papers:
        return "_未检索到文献_"
    lines = []
    for i, p in enumerate(papers, 1):
        src_tag = p.source or "?"
        cite = f" | 引用 {p.citations}" if p.citations else ""
        date = f" | {p.publication_date}" if p.publication_date else ""
        ids = " | ".join(
            s for s in (f"DOI: {p.doi}" if p.doi else "", f"arXiv: {p.arxiv_id}" if p.arxiv_id else "")
            if s
        )
        lines.append(
            f"**{i}. [{src_tag}] {p.title}**{date}{cite}\n\n"
            f"- 作者: {', '.join(p.authors[:3]) if p.authors else '未知'}\n"
            f"- 摘要: {p.abstract[:200]}{'...' if len(p.abstract) > 200 else ''}\n"
            + (f"- {ids}\n" if ids else "")
        )
    return "\n\n".join(lines)


def _format_kps(kps) -> str:
    """格式化知识点列表为 Markdown。"""
    if not kps:
        return "_无知识点_"
    lines = []
    for i, kp in enumerate(kps, 1):
        lines.append(
            f"**{i}. [{kp.category}] {kp.name}** (重要度 {kp.importance:.2f})\n\n"
            f"- 定义: {kp.definition}\n"
        )
    return "\n\n".join(lines)


def run_research(topic: str, depth: str, max_papers: int):
    """流式执行四 Agent 协作研究任务。"""
    if not topic.strip():
        yield "请输入研究主题", "", "", "", ""
        return
    set_trace_id()
    orch = get_orchestrator()

    # 阶段一：规划
    yield f"⏳ [PlannerAgent] 正在规划主题「{topic}」...", "", "", "", ""

    task = orch.create_task(topic, depth)

    # 阶段二：执行四 Agent 流水线，逐步 yield 进度
    from src.orchestrator.shared_board import SharedBoard
    from src.common.data_models import TaskStatus

    board = SharedBoard(task)

    # 1. Planner
    yield "⏳ [PlannerAgent] 生成研究计划...", "", "", "", ""
    board = orch.planner.safe_execute(board, timeout=60)
    plan = board.read("plan")
    subtask_n = len(plan.subtasks) if plan else 0
    yield f"✅ [PlannerAgent] 规划完成（{subtask_n} 子任务）\n⏳ [CollectorAgent] 检索文献中...", "", "", "", ""

    # 2. Collector
    board = orch.collector.safe_execute(board, timeout=150)
    papers = board.read("papers", [])
    task.papers = papers
    papers_md = _format_papers(papers)
    yield (
        f"✅ [CollectorAgent] 检索到 {len(papers)} 篇文献\n"
        f"⏳ [EngineerAgent] 抽取知识点中...",
        papers_md, "", "", ""
    )

    # 3. Engineer
    board = orch.engineer.safe_execute(board, timeout=90)
    kps = board.read("knowledge_points", [])
    task.knowledge_points = kps
    kps_md = _format_kps(kps)
    yield (
        f"✅ [EngineerAgent] 抽取知识点 {len(kps)} 个\n"
        f"⏳ [WriterAgent] 撰写报告...",
        papers_md, kps_md, "", ""
    )

    # 4. Writer
    board = orch.writer.safe_execute(board, timeout=90)
    report = board.read("final_report", "")
    task.report = report
    task.status = board.get_task_state()
    task.quality_score = task.quality_score or 0.8
    yield (
        f"✅ [WriterAgent] 报告生成完成\n🎉 研究任务完成（质量评分 {task.quality_score}）",
        papers_md, kps_md, report, "",
    )


def _format_archive_list() -> str:
    """格式化最近任务档案列表为 Markdown（含 JEV 评分）。"""
    try:
        from src.common.observability import get_observability

        obs = get_observability()
        archive_dir = obs.archive_dir
        archives = sorted(archive_dir.glob("task_*.json"), reverse=True)[:10]
        if not archives:
            return "_暂无归档任务_"
        lines = ["| 任务 ID | 主题 | 状态 | 耗时 | 质量分 | JEV(报告) |", "|---|---|---|---|---|---|"]
        for p in archives:
            import json as _json
            with open(p, encoding="utf-8") as f:
                d = _json.load(f)
            tid = d.get("task_id", "")[-8:]
            topic = (d.get("topic", "") or "")[:20]
            status = d.get("status", "")
            dur = f"{d.get('duration_sec', 0):.1f}s"
            qs = f"{d.get('quality_score', 0):.2f}"
            jev = d.get("jev_scores", {}) or {}
            writer_jev = jev.get("writer", 0)
            wj = f"{writer_jev:.2f}" if writer_jev else "-"
            lines.append(f"| {tid} | {topic} | {status} | {dur} | {qs} | {wj} |")
        return "\n".join(lines)
    except Exception as e:
        return f"_归档读取失败: {e}_"


def _format_metrics() -> str:
    """格式化核心指标看板。"""
    try:
        from src.common.observability import get_observability

        m = get_observability().get_metrics_summary()
        t = m.get("tasks", {})
        tl = m.get("tools", {})
        return (
            f"## 📊 核心指标看板\n\n"
            f"**任务统计**\n"
            f"- 总任务数: {t.get('total', 0)}\n"
            f"- 成功数: {t.get('success', 0)}\n"
            f"- 成功率: {t.get('success_rate', 0) * 100:.1f}%\n"
            f"- 平均耗时: {t.get('avg_duration_sec', 0)}s\n"
            f"- 平均质量分: {t.get('avg_quality_score', 0)}\n\n"
            f"**工具调用**\n"
            f"- 总调用: {tl.get('total_calls', 0)}\n"
            f"- 成功调用: {tl.get('success_calls', 0)}\n"
            f"- 成功率: {tl.get('success_rate', 0) * 100:.1f}%\n"
        )
    except Exception as e:
        return f"_指标读取失败: {e}_"


def build_ui():
    """构建 Gradio 界面（阶段三：JEV 决策 + 任务档案）。"""
    import gradio as gr

    with gr.Blocks(title="research-agent 深度研究助手") as app:
        gr.Markdown(
            "# 📚 research-agent 个人深度研究助手\n"
            "阶段三：JEV 决策引擎 + CoT+ToT 规划 + 全链路质量门限 + 任务档案\n"
            "输入研究主题，自动完成规划→检索→整理→生成全流程。"
        )
        with gr.Tab("🔬 研究工作台"):
            with gr.Row():
                topic_input = gr.Textbox(
                    label="研究主题",
                    placeholder="例如：Transformer 架构核心原理 / RAG 优化技术最新进展",
                    scale=3,
                )
                depth = gr.Dropdown(
                    choices=["basic", "standard", "deep"],
                    value="standard",
                    label="研究深度",
                    scale=1,
                )
                max_papers = gr.Slider(
                    minimum=5, maximum=30, value=20, step=1, label="最大文献数", scale=1
                )
            run_btn = gr.Button("🚀 开始研究", variant="primary")
            status = gr.Markdown(label="状态", value="等待输入...")
            with gr.Accordion("文献列表", open=False):
                papers_out = gr.Markdown()
            with gr.Accordion("知识点", open=False):
                kps_out = gr.Markdown()
            report_out = gr.Markdown(label="研究报告")

            run_btn.click(
                run_research,
                inputs=[topic_input, depth, max_papers],
                outputs=[status, papers_out, kps_out, report_out, gr.Textbox(visible=False)],
            )

        with gr.Tab("📋 任务档案与 JEV 评分"):
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新归档", variant="primary")
            with gr.Accordion("核心指标看板", open=True):
                metrics_out = gr.Markdown(value=_format_metrics())
            with gr.Accordion("最近 10 个任务档案", open=True):
                archive_out = gr.Markdown(value=_format_archive_list())
            refresh_btn.click(
                lambda: (_format_metrics(), _format_archive_list()),
                outputs=[metrics_out, archive_out],
            )

    return app


def main() -> None:
    """启动 Gradio 服务。"""
    ensure_dirs()
    env = get_env()
    app = build_ui()
    app.queue()
    import gradio as gr

    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
