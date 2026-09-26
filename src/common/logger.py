"""统一日志体系。

对应架构 6.5 可观测性设计：
- 分级日志 DEBUG/INFO/WARN/ERROR
- 全链路 trace_id 追踪
- 单机部署使用 loguru，支持文件轮转
"""

from __future__ import annotations

import os
import sys
import uuid
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from loguru import logger

# 全链路 trace_id，通过 contextvar 在协程间隔离
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def set_trace_id(trace_id: str | None = None) -> str:
    """设置当前上下文的 trace_id，未指定则随机生成。

    所有日志输出会自动带上该 id，便于全链路追踪。
    """
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """获取当前上下文的 trace_id。"""
    return _trace_id.get()


def _format(record: dict) -> str:  # type: ignore[type-arg]
    """自定义日志格式，注入 trace_id。"""
    # 绑定到 record.extra 的 trace_id（默认回退）
    trace = record["extra"].get("trace_id", get_trace_id())
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        f"<cyan>[{trace}]</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>\n"
    )


@lru_cache(maxsize=1)
def _configure_logger() -> None:
    """初始化日志配置（仅执行一次）。"""
    logger.remove()

    # 控制台输出
    logger.add(
        sys.stderr,
        format=_format,  # type: ignore[arg-type]
        level=os.environ.get("LOG_LEVEL", "INFO"),
        colorize=True,
    )

    # 文件输出，轮转 + 保留
    log_dir = Path(os.environ.get("LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_dir / "research-agent.log"),
        format=_format,  # type: ignore[arg-type]
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )


def get_logger(name: str | None = None):
    """获取一个绑定 trace_id 的 logger 实例。

    使用方在调用前通过 set_trace_id 设置上下文，
    日志输出会自动带上对应 trace_id。
    """
    _configure_logger()
    if name:
        return logger.bind(name=name, trace_id=get_trace_id())
    return logger.bind(trace_id=get_trace_id())
