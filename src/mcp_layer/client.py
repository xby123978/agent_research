"""MCP Client 实现（官方 mcp SDK）。

对应架构 3.4.3 能力发现与路由：
- 启动时通过 MCP 协议连接各 Server，拉取 tools/list 并建立工具索引
- Agent 调用工具时由 MCP Client 统一路由，对上层透明
- 统一错误处理：指数退避重试、超时控制

传输（mcp.transport）：
- memory：进程内会话（官方 ClientSession + 内存流），与 Agent 共享进程状态（默认）
- stdio：子进程 + stdio 管道，真正跨进程
- http：连接已启动的 streamable-http Server

对外保持同步 API（call_tool / list_tools / call_jsonrpc），
内部用后台事件循环桥接官方异步客户端。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from ..common.config import get_settings
from ..common.exceptions import ToolError
from ..common.logger import get_logger
from ..tools.base import BaseTool, ToolResult, ToolSchema
from .servers import PROJECT_ROOT, SERVER_BUILDERS

logger = get_logger(__name__)


class _AsyncLoop:
    """后台事件循环线程，供同步代码桥接异步 MCP 客户端。"""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="mcp-loop", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any, timeout: float) -> Any:
        """提交协程到后台循环并阻塞等待结果。"""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def shutdown(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


class MCPClient:
    """MCP 协议客户端。

    管理所有 Server 的连接、工具发现与路由，对上层 Agent 透明。
    """

    def __init__(self) -> None:
        cfg = get_settings().get("mcp", {})
        self.max_retries = int(cfg.get("max_retries", 3))
        self.request_timeout = int(cfg.get("request_timeout", 30))
        self.call_timeout = float(cfg.get("call_timeout", 120))
        self.retry_backoff = float(cfg.get("retry_backoff", 2.0))
        self.transport = str(cfg.get("transport", "memory"))
        self._transports: dict[str, str] = dict(cfg.get("transports", {}) or {})

        # 本地覆盖工具（register_tool 注入，优先级高于 Server）
        self._tools: dict[str, BaseTool] = {}
        # Server 名称 → FastMCP 实例（memory 传输时可用）
        self._servers: dict[str, Any] = {}
        # Server 名称 → ClientSession
        self._sessions: dict[str, ClientSession] = {}
        # Server 名称 → 退出栈（保持会话长驻）
        self._stacks: dict[str, AsyncExitStack] = {}
        # 工具名 → Server 名称
        self._tool_index: dict[str, str] = {}
        # Server 名称 → {工具名: ToolSchema}
        self._server_schemas: dict[str, dict[str, ToolSchema]] = {}

        self._loop = _AsyncLoop()
        self._connect_all()

    # ---------- 连接管理 ----------

    def _transport_for(self, server_name: str) -> str:
        return self._transports.get(server_name, self.transport)

    def _connect_all(self) -> None:
        """连接所有 Server，失败仅告警不阻断（离线可用）。"""
        for name, builder in SERVER_BUILDERS.items():
            try:
                self._connect(name, builder)
                logger.info(f"MCP Server 已连接: {name} transport={self._transport_for(name)}")
            except Exception as e:
                logger.warning(f"MCP Server 连接失败: {name}: {e}")

    def _connect(self, name: str, builder: Any) -> None:
        transport = self._transport_for(name)
        if transport == "stdio":
            session, stack, tools = self._loop.run(
                self._open_stdio(name), timeout=self.request_timeout
            )
        else:
            server = builder()
            self._servers[name] = server
            session, stack, tools = self._loop.run(
                self._open_memory(server), timeout=self.request_timeout
            )

        self._sessions[name] = session
        self._stacks[name] = stack
        schemas: dict[str, ToolSchema] = {}
        for t in tools.tools:
            schemas[t.name] = ToolSchema(
                name=t.name,
                description=t.description or "",
                input_schema=dict(t.inputSchema or {}),
            )
            self._tool_index[t.name] = name
        self._server_schemas[name] = schemas

    async def _open_memory(
        self, server: Any
    ) -> tuple[ClientSession, AsyncExitStack, Any]:
        """进程内会话（官方 ClientSession + 内存流）。"""
        stack = AsyncExitStack()
        session = await stack.enter_async_context(
            create_connected_server_and_client_session(server)
        )
        tools = await session.list_tools()
        return session, stack, tools

    async def _open_stdio(
        self, name: str
    ) -> tuple[ClientSession, AsyncExitStack, Any]:
        """子进程 stdio 会话。"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_layer.servers", name, "stdio"],
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = await session.list_tools()
        return session, stack, tools

    def get_server(self, name: str) -> Any | None:
        """返回 Server 实例（仅 memory 传输可用）。"""
        return self._servers.get(name)

    def register_tool(self, tool: BaseTool) -> None:
        """注册本地覆盖工具（优先级高于 Server 提供的同名工具）。"""
        self._tools[tool.name] = tool
        logger.info(f"MCP 工具注册（本地）: {tool.name}")

    # ---------- 能力发现 ----------

    def list_tools(self) -> list[ToolSchema]:
        """对应 MCP tools/list：返回所有可用工具 schema（本地覆盖优先）。"""
        result: dict[str, ToolSchema] = {}
        for schemas in self._server_schemas.values():
            result.update(schemas)
        for tool in self._tools.values():
            result[tool.name] = tool.schema
        return list(result.values())

    # ---------- 工具调用 ----------

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """对应 MCP tools/call：路由到本地工具或 Server 并执行，带统一重试。"""
        if name not in self._tools and name not in self._tool_index:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"工具不存在: {name}",
            )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._dispatch(name, arguments)
                if result.success:
                    return result
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

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name in self._tools:
            return self._tools[name].call(**arguments)
        return self._call_server(name, arguments)

    def _call_server(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        server_name = self._tool_index[name]
        session = self._sessions[server_name]
        response = self._loop.run(
            session.call_tool(name, arguments), timeout=self.call_timeout
        )
        if response.isError:
            text = response.content[0].text if response.content else "工具执行失败"
            return ToolResult(tool_name=name, success=False, error=text)
        if not response.content:
            return ToolResult(tool_name=name, success=False, error="空响应")
        text = getattr(response.content[0], "text", "")
        try:
            return ToolResult(**json.loads(text))
        except (json.JSONDecodeError, TypeError, ValueError):
            return ToolResult(tool_name=name, success=True, content=[{"text": text}])

    # ---------- JSON-RPC ----------

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

    # ---------- 关闭 ----------

    def shutdown(self) -> None:
        """关闭所有会话与后台循环（尽力而为）。"""
        for stack in list(self._stacks.values()):
            try:
                self._loop.run(stack.aclose(), timeout=5)
            except Exception:
                pass
        self._stacks.clear()
        self._sessions.clear()
        try:
            self._loop.shutdown()
        except Exception:
            pass


_mcp_instance: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    """获取全局 MCP 客户端单例。"""
    global _mcp_instance
    if _mcp_instance is None:
        _mcp_instance = MCPClient()
    return _mcp_instance
