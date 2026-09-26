"""MCP 客户端与工具测试。"""

from __future__ import annotations

from src.mcp.client import MCPClient
from src.tools.base import ToolResult, ToolSchema


def test_mcp_list_tools():
    client = MCPClient()
    tools = client.list_tools()
    names = [t.name for t in tools]
    assert "search_openalex" in names


def test_mcp_call_nonexistent_tool():
    client = MCPClient()
    result = client.call_tool("not_a_tool", {})
    assert not result.success


def test_mcp_jsonrpc_list():
    client = MCPClient()
    resp = client.call_jsonrpc(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )
    assert resp["jsonrpc"] == "2.0"
    assert "tools" in resp["result"]


def test_mcp_jsonrpc_call_unknown():
    client = MCPClient()
    resp = client.call_jsonrpc(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}
    )
    # 工具不存在时返回 result 含 success=False 与 error，而非 JSON-RPC error
    assert resp["jsonrpc"] == "2.0"
    assert resp["result"]["success"] is False
    assert "error" in resp["result"]


def test_tool_result_model():
    r = ToolResult(tool_name="t", success=True, content=[{"a": 1}])
    assert r.success
    assert r.content[0]["a"] == 1


def test_tool_schema():
    s = ToolSchema(name="n", description="d")
    assert s.input_schema == {}
