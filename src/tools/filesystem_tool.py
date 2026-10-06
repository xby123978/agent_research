"""文件系统工具。

对应架构 3.7 输出同步类与 3.4.2 文件系统 Server。
本地实现（非 Docker），基于 pathlib 读写本地文件。
约束：仅在 OUTPUT_DIR/DATA_DIR 范围内操作，禁止越权访问。
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..common.config import get_env
from ..common.logger import get_logger
from .base import BaseTool, ToolResult, ToolSchema

logger = get_logger(__name__)


class FilesystemTool(BaseTool):
    """本地文件系统读写工具（MCP Filesystem Server 本地降级实现）。"""

    schema = ToolSchema(
        name="filesystem",
        description="本地文件读写（限 outputs/ 与 data/ 目录）",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "read | write | list | mkdir",
                    "default": "read",
                },
                "path": {"type": "string", "description": "相对路径（基于 outputs/）"},
                "content": {"type": "string", "description": "写入内容（write 用）"},
            },
            "required": ["action", "path"],
        },
    )

    def __init__(self) -> None:
        env = get_env()
        self.base_dir = os.path.abspath(env["output_dir"])
        self.data_dir = os.path.abspath(env["data_dir"])

    def call(self, **kwargs: Any) -> ToolResult:
        start = time.time()
        action = kwargs.get("action", "read")
        rel_path = kwargs.get("path", "")
        try:
            full = self._resolve(rel_path)
            if action == "read":
                return self._read(full, start)
            if action == "write":
                return self._write(full, kwargs.get("content", ""), start)
            if action == "list":
                return self._list(full, start)
            if action == "mkdir":
                os.makedirs(full, exist_ok=True)
                return self._ok(f"已创建目录: {rel_path}", start)
            return ToolResult(tool_name="filesystem", success=False, error=f"未知 action: {action}", duration_ms=(time.time() - start) * 1000)
        except Exception as e:
            logger.error(f"文件系统操作失败: {e}", exc_info=True)
            return ToolResult(tool_name="filesystem", success=False, error=str(e), duration_ms=(time.time() - start) * 1000)

    def _resolve(self, rel_path: str) -> str:
        """安全解析路径，禁止路径穿越。"""
        full = os.path.abspath(os.path.join(self.base_dir, rel_path))
        if not full.startswith(self.base_dir):
            # 允许 data_dir 下的访问
            full2 = os.path.abspath(os.path.join(self.data_dir, rel_path))
            if full2.startswith(self.data_dir):
                return full2
            raise PermissionError(f"路径越权访问: {rel_path}")
        return full

    def _read(self, full: str, start: float) -> ToolResult:
        if not os.path.exists(full):
            return ToolResult(tool_name="filesystem", success=False, error="文件不存在", duration_ms=(time.time() - start) * 1000)
        with open(full, encoding="utf-8") as f:
            content = f.read()
        return ToolResult(
            tool_name="filesystem",
            success=True,
            content=[{"type": "text", "text": content, "path": full}],
            duration_ms=(time.time() - start) * 1000,
        )

    def _write(self, full: str, content: str, start: float) -> ToolResult:
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return self._ok(f"已写入: {full}", start)

    def _list(self, full: str, start: float) -> ToolResult:
        if not os.path.isdir(full):
            return ToolResult(tool_name="filesystem", success=False, error="目录不存在", duration_ms=(time.time() - start) * 1000)
        entries = os.listdir(full)
        return ToolResult(
            tool_name="filesystem",
            success=True,
            content=[{"type": "list", "entries": entries, "path": full}],
            duration_ms=(time.time() - start) * 1000,
        )

    def _ok(self, msg: str, start: float) -> ToolResult:
        return ToolResult(
            tool_name="filesystem",
            success=True,
            content=[{"type": "text", "text": msg}],
            duration_ms=(time.time() - start) * 1000,
        )
