"""FastAPI REST API 服务。

对应架构 5.1 核心服务接口：
- POST   /api/v1/tasks          创建研究任务
- GET    /api/v1/tasks/{id}     获取任务状态与结果
- GET    /api/v1/tasks          获取任务列表
- DELETE /api/v1/tasks/{id}     删除任务
- POST   /api/v1/memory/search  语义检索记忆
- GET    /api/v1/memory/{id}    获取指定知识点
- POST   /api/v1/memory/graph   查询知识图谱子图
- POST   /api/v1/skills/{name}/execute  执行 Skill
- GET    /api/v1/skills         获取 Skill 列表

启动：python -m src.api.server 或 uvicorn src.api.server:app
依赖：fastapi、uvicorn（轻量，非 Docker）
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

# 将 src 加入路径
SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.common.config import ensure_dirs
from src.common.logger import get_logger, set_trace_id
from src.memory.manager import get_memory
from src.mcp.client import get_mcp_client
from src.orchestrator import get_orchestrator
from src.skills.citation_manager import CitationManagerSkill
from src.skills.knowledge_card import KnowledgeCardSkill
from src.skills.literature_review import LiteratureReviewSkill

logger = get_logger(__name__)

app = FastAPI(
    title="research-agent API",
    version="0.1.0",
    description="个人深度研究助手 REST API（对应架构 5.1）",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 内存任务存储（MVP，单机单用户）=====
_tasks_store: dict[str, Any] = {}
_tasks_lock = threading.Lock()


# ===== 请求/响应模型 =====
class TaskCreateRequest(BaseModel):
    topic: str = Field(..., description="研究主题")
    depth: str = Field("standard", description="研究深度: basic/standard/deep")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="检索关键词")
    top_k: int = Field(5, description="返回数")


class GraphQueryRequest(BaseModel):
    entity_id: str = Field(..., description="起点实体 id")
    depth: int = Field(1, description="图谱查询深度")


class SkillExecuteRequest(BaseModel):
    inputs: dict[str, Any] = Field(..., description="Skill 输入参数")


# ===== 任务管理接口（架构 5.1.1）=====
@app.post("/api/v1/tasks")
def create_task(req: TaskCreateRequest) -> dict[str, Any]:
    """创建并异步执行研究任务。"""
    set_trace_id()
    orch = get_orchestrator()
    task = orch.create_task(req.topic, req.depth)

    def _run() -> None:
        try:
            result = orch.run(req.topic, req.depth)
            with _tasks_lock:
                _tasks_store[result.task_id] = result.model_dump()
        except Exception as e:
            logger.error(f"任务 {task.task_id} 执行失败: {e}", exc_info=True)
            task.status = "failed"
            with _tasks_lock:
                _tasks_store[task.task_id] = task.model_dump()

    with _tasks_lock:
        _tasks_store[task.task_id] = task.model_dump()
    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task.task_id, "status": task.status.value}


@app.get("/api/v1/tasks")
def list_tasks() -> dict[str, Any]:
    """获取任务列表。"""
    with _tasks_lock:
        tasks = list(_tasks_store.values())
    return {"tasks": tasks, "count": len(tasks)}


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    """获取任务状态与结果。"""
    with _tasks_lock:
        task = _tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return task


@app.delete("/api/v1/tasks/{task_id}")
def delete_task(task_id: str) -> dict[str, Any]:
    """删除任务。"""
    with _tasks_lock:
        removed = _tasks_store.pop(task_id, None)
    if not removed:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return {"deleted": task_id}


# ===== 记忆查询接口（架构 5.1.2）=====
@app.post("/api/v1/memory/search")
def search_memory(req: MemorySearchRequest) -> dict[str, Any]:
    """语义检索记忆。"""
    memory = get_memory()
    items = memory.search(req.query, top_k=req.top_k)
    return {
        "results": [i.model_dump() for i in items],
        "count": len(items),
    }


@app.get("/api/v1/memory/{item_id}")
def get_memory_item(item_id: str) -> dict[str, Any]:
    """获取指定记忆条目。"""
    memory = get_memory()
    item = memory.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"记忆条目不存在: {item_id}")
    return item.model_dump()


@app.post("/api/v1/memory/graph")
def query_graph(req: GraphQueryRequest) -> dict[str, Any]:
    """查询知识图谱子图（基于工作记忆）。"""
    memory = get_memory()
    if memory.working is None:
        raise HTTPException(status_code=503, detail="工作记忆未初始化")
    related_ids = memory.working.graph_query(req.entity_id, req.depth)
    nodes: list[dict[str, Any]] = []
    for rid in related_ids:
        item = memory.working.get(rid)
        if item:
            nodes.append(item.model_dump())
    return {"entity_id": req.entity_id, "depth": req.depth, "nodes": nodes}


# ===== Skill 执行接口（架构 5.1.3）=====
_SKILLS = {
    "literature_review": LiteratureReviewSkill,
    "knowledge_card": KnowledgeCardSkill,
    "citation_manager": CitationManagerSkill,
}


@app.get("/api/v1/skills")
def list_skills() -> dict[str, Any]:
    """获取可用 Skill 列表。"""
    return {
        "skills": [
            {"name": n, "description": s.description, "version": s.version}
            for n, s in _SKILLS.items()
        ]
    }


@app.post("/api/v1/skills/{name}/execute")
def execute_skill(name: str, req: SkillExecuteRequest) -> dict[str, Any]:
    """执行指定 Skill。"""
    skill_cls = _SKILLS.get(name)
    if skill_cls is None:
        raise HTTPException(status_code=404, detail=f"Skill 不存在: {name}")
    skill = skill_cls()
    result = skill.run(req.inputs)
    return result


# ===== MCP 工具列表（架构 5.2）=====
@app.get("/api/v1/tools")
def list_tools() -> dict[str, Any]:
    """列出所有 MCP 工具（对应 tools/list）。"""
    client = get_mcp_client()
    return {"tools": [t.model_dump() for t in client.list_tools()]}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    """启动 API 服务。"""
    ensure_dirs()
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(__import__("os").environ.get("API_PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
