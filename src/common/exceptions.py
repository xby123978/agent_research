"""自定义异常定义。

对应架构 6.3 容错与降级机制，所有外部调用异常均在此定义，
便于上层统一捕获并执行指数退避重试与降级策略。
"""

from __future__ import annotations


class ResearchAgentError(Exception):
    """所有 research-agent 异常的基类。"""

    def __init__(self, message: str, *, recoverable: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.recoverable = recoverable

    def __str__(self) -> str:
        return self.message


class ConfigError(ResearchAgentError):
    """配置缺失或格式错误。"""


class LLMError(ResearchAgentError):
    """大模型调用失败（鉴权、限流、超时等）。"""


class EmbeddingError(ResearchAgentError):
    """嵌入模型调用失败。"""


class MemoryError(ResearchAgentError):
    """记忆模块读写失败。"""


class ToolError(ResearchAgentError):
    """工具调用失败（对应 MCP 工具层）。"""

    def __init__(self, message: str, *, tool_name: str = "", recoverable: bool = True) -> None:
        super().__init__(message, recoverable=recoverable)
        self.tool_name = tool_name


class MCPError(ResearchAgentError):
    """MCP 协议层错误（路由失败、Server 不可用等）。"""


class QualityGateError(ResearchAgentError):
    """质量门限未达标，对应架构 6.1。"""

    def __init__(self, message: str, *, stage: str = "", metric: str = "", value: float = 0.0) -> None:
        super().__init__(message, recoverable=False)
        self.stage = stage
        self.metric = metric
        self.value = value
