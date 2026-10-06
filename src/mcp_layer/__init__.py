"""MCP 协议适配层（官方 mcp SDK）。

对应架构 3.4 MCP协议适配层：
- 统一工具接入标准、能力自动发现、请求路由、错误重试
- 基于官方 MCP SDK（FastMCP Server + ClientSession）

注：包名用 mcp_layer 而非 mcp，避免与官方 SDK 的顶层包名 `mcp` 冲突
（conftest/入口会把 src/ 加入 sys.path，届时顶层 `mcp` 会被 src/mcp 遮蔽）。
"""
from .client import MCPClient, get_mcp_client

__all__ = ["MCPClient", "get_mcp_client"]
