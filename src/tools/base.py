"""工具基类与数据结构。

对应架构 3.7 工具集层与 5.2 MCP 工具调用规范。
每个工具以 MCP 标准格式定义 schema，统一返回 ToolResult。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """工具元数据，对应 MCP tools/list 协议。"""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    # 别名兼容：servers.py 旧代码用了 .parameters
    parameters: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        if self.input_schema and not self.parameters:
            self.parameters = self.input_schema
        elif self.parameters and not self.input_schema:
            self.input_schema = self.parameters


class ToolResult(BaseModel):
    """工具调用结果，对应 MCP tools/call 响应。"""

    tool_name: str
    success: bool = True
    content: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0


class BaseTool(ABC):
    """工具抽象基类，所有工具实现 call 方法。"""

    schema: ToolSchema = ToolSchema(name="base", description="", input_schema={})

    @abstractmethod
    def call(self, **kwargs: Any) -> ToolResult:
        """执行工具调用。"""

    @property
    def name(self) -> str:
        return self.schema.name
