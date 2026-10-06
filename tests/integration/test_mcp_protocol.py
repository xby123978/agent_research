"""MCP 真实协议端到端测试。

覆盖两种传输：
- memory：官方 ClientSession + 内存流（进程内会话）
- stdio：子进程 + stdio 管道（真正跨进程）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (str(ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.common import config as config_mod
from src.mcp_layer.client import MCPClient

_ALL_TOOLS = {"search_openalex", "search_crossref", "parse_pdf", "extract_metadata", "filesystem", "knowledge_base"}


def _make_client(**mcp_overrides) -> MCPClient:
    """按覆盖后的 mcp 配置构造客户端（用完恢复配置）。"""
    settings = config_mod.get_settings()
    old = settings.get("mcp", {})
    settings["mcp"] = {**old, **mcp_overrides}
    try:
        return MCPClient()
    finally:
        settings["mcp"] = old


def test_list_tools_exposes_all_servers():
    client = MCPClient()
    names = {t.name for t in client.list_tools()}
    assert _ALL_TOOLS.issubset(names), f"缺失工具: {_ALL_TOOLS - names}"
    for t in client.list_tools():
        assert t.description, f"工具 {t.name} 缺少 description"
        assert t.input_schema, f"工具 {t.name} 缺少 input_schema"


def test_unknown_tool_fails():
    client = MCPClient()
    assert not client.call_tool("not_a_tool", {}).success


def test_filesystem_roundtrip_memory_transport():
    """进程内会话：写入后读回。"""
    client = _make_client(transport="memory")
    path = "mcp_smoke_memory.txt"
    w = client.call_tool("filesystem", {"action": "write", "path": path, "content": "hello-mcp"})
    assert w.success, w.error
    r = client.call_tool("filesystem", {"action": "read", "path": path})
    assert r.success, r.error
    assert "hello-mcp" in r.content[0]["text"]


def test_filesystem_roundtrip_stdio_transport():
    """子进程 stdio：真正跨进程的 MCP 调用。"""
    client = _make_client(transport="stdio", transports={"filesystem": "stdio"})
    assert client._transport_for("filesystem") == "stdio"
    assert client.get_server("filesystem") is None  # stdio 无本地实例
    path = "mcp_smoke_stdio.txt"
    w = client.call_tool("filesystem", {"action": "write", "path": path, "content": "hello-stdio"})
    assert w.success, w.error
    r = client.call_tool("filesystem", {"action": "read", "path": path})
    assert r.success, r.error
    assert "hello-stdio" in r.content[0]["text"]
