"""MCP Client 实现。

对应架构 3.4.3 能力发现与路由：
- 启动时自动拉取所有 MCP Server 的工具列表，动态注册到工具池
- Agent 调用工具时由 MCP Client 统一路由，对上层透明
- 统一错误处理：指数退避重试、超时控制、失败降级备选工具

MVP 阶段以「内置工具」方式注册，遵循 MCP 标准请求/响应格式（架构 5.2），
阶段二升级为真实的 stdio/HTTP MCP Server 连接。
"""

from __future__ import annotations

import time
from typing import Any

from ..common.config import get_settings
from ..common.exceptions import MCPError, ToolError
from ..common.logger import get_logger
from ..tools.base import BaseTool, ToolResult, ToolSchema
from ..tools.crossref_tool import CrossRefTool
from ..tools.filesystem_tool import FilesystemTool
from ..tools.knowledge_base_tool import KnowledgeBaseTool
from ..tools.openalex_tool import OpenAlexTool
from .servers import (
    AcademicSearchServer,
    DocumentProcessorServer,
    FilesystemServer,
    KnowledgeBaseServer,
)

logger = get_logger(__name__)


class MCPClient:
    """MCP 协议适配客户端。

    管理所有工具的注册、发现与路由，对上层 Agent 透明。
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._servers: dict[str, Any] = {}
        self._config = get_settings().get("mcp", {})
        self.max_retries = int(self._config.get("max_retries", 3))
        self.request_timeout = int(self._config.get("request_timeout", 30))
        self.retry_backoff = float(self._config.get("retry_backoff", 2.0))
        self._register_builtin_tools()
        self._register_mcp_servers()

    def _register_builtin_tools(self) -> None:
        """注册内置工具（MVP 阶段以本地工具模拟 MCP Server 能力）。

        主检索源为 OpenAlex（国内访问稳定、免费、覆盖 2.5 亿+ 文献）。
        arXiv / Semantic Scholar 作为备选保留源码，不在 MVP 默认启用，
        避免国内网络超时影响全链路。
        """
        builtin = [OpenAlexTool()]
        for tool in builtin:
            self.register_tool(tool)

    def _register_mcp_servers(self) -> None:
        """阶段二：注册自定义 MCP Server。

        AcademicSearchServer 聚合多源学术检索能力（OpenAlex + CrossRef）。
        DocumentProcessorServer 提供 PDF 解析能力（阶段二打桩降级）。
        FilesystemServer 本地文件读写（非 Docker）。
        KnowledgeBaseServer 对接记忆模块（非 Docker）。
        """
        try:
            academic = AcademicSearchServer()
            self._servers["academic_search"] = academic
            # 把 Server 内工具直接挂到本地工具池，供 call_tool 调用
            for tool in [academic._openalex, academic._crossref]:
                if tool.name not in self._tools:
                    self.register_tool(tool)
        except Exception as e:
            logger.warning(f"AcademicSearchServer 注册失败: {e}")
        try:
            doc_server = DocumentProcessorServer()
            self._servers["document_processor"] = doc_server
            logger.info("DocumentProcessorServer 注册（PDF 解析阶段二降级）")
        except Exception as e:
            logger.warning(f"DocumentProcessorServer 注册失败: {e}")
        # 文件系统 Server（本地实现，非 Docker）
        try:
            fs_server = FilesystemServer()
            self._servers["filesystem"] = fs_server
            fs_tool = fs_server._fs
            if fs_tool.name not in self._tools:
                self.register_tool(fs_tool)
            logger.info("FilesystemServer 注册（本地文件读写）")
        except Exception as e:
            logger.warning(f"FilesystemServer 注册失败: {e}")
        # 知识库 Server（对接记忆模块，非 Docker）
        try:
            kb_server = KnowledgeBaseServer()
            self._servers["knowledge_base"] = kb_server
            kb_tool = kb_server._kb
            if kb_tool.name not in self._tools:
                self.register_tool(kb_tool)
            logger.info("KnowledgeBaseServer 注册（对接记忆模块）")
        except Exception as e:
            logger.warning(f"KnowledgeBaseServer 注册失败: {e}")

    def get_server(self, name: str) -> Any | None:
        return self._servers.get(name)

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"MCP 工具注册: {tool.name}")

    def list_tools(self) -> list[ToolSchema]:
        """对应 MCP tools/list：返回所有可用工具 schema。"""
        return [t.schema for t in self._tools.values()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """对应 MCP tools/call：路由到对应工具并执行。

        带统一错误处理与重试。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"工具不存在: {name}",
            )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = tool.call(**arguments)
                if result.success:
                    return result
                # 工具内部已重试，仍失败则记录
                last_error = ToolError(result.error or "未知错误", tool_name=name)
                if attempt < self.max_retries:
                    wait = self.retry_backoff ** attempt
                    logger.warning(
                        f"工具 {name} 第 {attempt} 次失败: {result.error}，{wait:.1f}s 后重试"
                    )
                    time.sleep(wait)
            except Exception as e:
                last_error = e
                wait = self.retry_backoff ** attempt
                logger.warning(f"工具 {name} 第 {attempt} 次异常: {e}，{wait:.1f}s 后重试")
                time.sleep(wait)

        return ToolResult(
            tool_name=name,
            success=False,
            error=f"工具 {name} 重试 {self.max_retries} 次后仍失败: {last_error}",
        )

    def call_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """标准 JSON-RPC 2.0 入口，对应架构 5.2 MCP 工具调用规范。

        请求: {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": ..., "arguments": {}}}
        响应: {"jsonrpc": "2.0", "result": {"content": []}}
        """
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        try:
            if method == "tools/list":
                result = {"tools": [t.model_dump() for t in self.list_tools()]}
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                tr = self.call_tool(tool_name, arguments)
                result = {
                    "content": [{"type": "text", "text": str(tr.content)}],
                    "success": tr.success,
                    "error": tr.error,
                    "duration_ms": tr.duration_ms,
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"未知方法: {method}"},
                }
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            }


_mcp_instance: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    """获取全局 MCP 客户端单例。"""
    global _mcp_instance
    if _mcp_instance is None:
        _mcp_instance = MCPClient()
    return _mcp_instance
