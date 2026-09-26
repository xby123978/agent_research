"""Obsidian 同步工具（架构 3.7）。

将研究报告与知识点同步到 Obsidian Vault：
- Markdown 文件格式
- 双向链接 [[知识点]]
- YAML frontmatter 元数据
- 自动创建 Vault 目录结构

降级策略：无 Obsidian 时写入本地 outputs/obsidian/
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..common.config import get_env, get_settings
from ..common.logger import get_logger
from .base import BaseTool, ToolResult, ToolSchema

logger = get_logger(__name__)


class ObsidianTool(BaseTool):
    """Obsidian 同步工具。"""

    schema = ToolSchema(
        name="sync_obsidian",
        description="将研究报告与知识点同步到 Obsidian Vault（Markdown + 双向链接）",
        input_schema={
            "type": "object",
            "properties": {
                "report": {"type": "string", "description": "研究报告内容"},
                "topic": {"type": "string", "description": "研究主题"},
                "knowledge_points": {
                    "type": "array",
                    "description": "知识点列表",
                },
                "papers": {"type": "array", "description": "文献列表"},
            },
            "required": ["report", "topic"],
        },
    )

    def call(self, **kwargs: Any) -> ToolResult:
        return self.execute(kwargs)

    def execute(self, args: dict[str, Any]) -> ToolResult:
        import time

        start = time.time()
        try:
            report = args.get("report", "")
            topic = args.get("topic", "untitled")
            kps = args.get("knowledge_points", [])
            papers = args.get("papers", [])

            vault_path = self._get_vault_path()
            notes_dir = vault_path / "research_notes"
            notes_dir.mkdir(parents=True, exist_ok=True)

            # 文件名：主题_日期
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(
                c for c in topic if c.isalnum() or c in (" ", "-", "_")
            ).strip()[:50]
            filename = f"{safe_topic}_{date_str}.md"
            filepath = notes_dir / filename

            # 生成 Markdown 内容
            content = self._build_markdown(topic, report, kps, papers)
            filepath.write_text(content, encoding="utf-8")

            # 同步知识点为独立笔记
            kp_count = self._sync_knowledge_points(kps, vault_path, topic)

            logger.info(f"Obsidian 同步完成: {filepath} + {kp_count} 个知识点笔记")

            return ToolResult(
                tool_name=self.name,
                success=True,
                content=[
                    {
                        "file": str(filepath),
                        "knowledge_point_notes": kp_count,
                        "vault": str(vault_path),
                    }
                ],
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"Obsidian 同步失败: {e}", exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _get_vault_path(self) -> Path:
        """获取 Obsidian Vault 路径，无配置时降级到 outputs/obsidian/。"""
        env = get_env()
        vault = (env.get("obsidian_vault") or "").strip()
        if vault:
            p = Path(vault)
            if p.exists():
                return p
            logger.warning(f"Obsidian Vault 路径不存在: {vault}，降级到 outputs/")
        # 降级到 outputs/obsidian/
        settings = get_settings().get("storage", {})
        output_dir = settings.get("output_dir", "outputs")
        return Path(output_dir) / "obsidian"

    def _build_markdown(
        self, topic: str, report: str, kps: list, papers: list
    ) -> str:
        """生成 Obsidian Markdown 内容（含 frontmatter 与双向链接）。"""
        # YAML frontmatter
        frontmatter = "---\n"
        frontmatter += f'title: "{topic}"\n'
        frontmatter += f"date: {datetime.now().isoformat()}\n"
        frontmatter += f'tags: [research, report]\n'
        frontmatter += f"knowledge_points: {len(kps)}\n"
        frontmatter += f"papers: {len(papers)}\n"
        frontmatter += "---\n\n"

        # 报告正文
        content = frontmatter + f"# {topic}\n\n{report}\n\n"

        # 知识点双向链接
        if kps:
            content += "## 关联知识点\n\n"
            for kp in kps:
                kp_name = getattr(kp, "name", str(kp))
                # Obsidian 双向链接格式
                content += f"- [[{kp_name}]]\n"
            content += "\n"

        # 文献引用
        if papers:
            content += "## 参考文献\n\n"
            for p in papers[:10]:
                title = getattr(p, "title", str(p))
                doi = getattr(p, "doi", "")
                authors = getattr(p, "authors", [])
                author_str = ", ".join(authors[:3]) if authors else "Unknown"
                ref = f"- {author_str}. {title}"
                if doi:
                    ref += f" [DOI: {doi}]"
                content += ref + "\n"

        return content

    def _sync_knowledge_points(
        self, kps: list, vault_path: Path, topic: str
    ) -> int:
        """将知识点同步为独立笔记。"""
        if not kps:
            return 0
        kp_dir = vault_path / "knowledge_points"
        kp_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for kp in kps:
            kp_name = getattr(kp, "name", str(kp))
            kp_def = getattr(kp, "definition", "")
            kp_cat = getattr(kp, "category", "general")
            kp_imp = getattr(kp, "importance", 0.5)
            # 文件名安全化
            safe_name = "".join(
                c for c in kp_name if c.isalnum() or c in (" ", "-", "_")
            ).strip()[:50]
            if not safe_name:
                continue
            filepath = kp_dir / f"{safe_name}.md"
            content = (
                "---\n"
                f'title: "{kp_name}"\n'
                f"date: {datetime.now().isoformat()}\n"
                f'tags: [knowledge, {kp_cat}]\n'
                f"importance: {kp_imp}\n"
                f'source_topic: "{topic}"\n'
                "---\n\n"
                f"# {kp_name}\n\n"
                f"**定义**: {kp_def}\n\n"
                f"**来源研究**: [[{topic}]]\n"
            )
            filepath.write_text(content, encoding="utf-8")
            count += 1
        return count
