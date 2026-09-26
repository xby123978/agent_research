"""MCP 协议适配层。

对应架构 3.4 MCP协议适配层：
- 统一工具接入标准、能力自动发现、请求路由、错误重试
- 所有外部工具通过 MCP 协议接入，禁止在 Agent 中直接硬编码 API 调用
"""
from .client import MCPClient, get_mcp_client

__all__ = ["MCPClient", "get_mcp_client"]
