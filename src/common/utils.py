"""通用工具函数。

对应架构第 8 节目录结构中的 src/common/utils。
当前提供带超时的函数执行封装，用于保证每个阶段都有硬超时上限，
避免任一阶段（规划/检索/整理/生成）因网络或模型异常而无限挂死。
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, TypeVar

from .exceptions import ResearchAgentError
from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def run_with_timeout(func: Callable[..., T], timeout_seconds: float, *args: Any, **kwargs: Any) -> T:
    """在子线程中执行 func，超过 timeout_seconds 则抛 ResearchAgentError。

    用于给无原生超时控制的同步调用加硬超时上限。
    注意：超时后子线程仍在后台运行直至结束（Python 无法强杀线程），
    但主流程会立即返回，避免 UI 长期无响应。
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def _run() -> None:
        try:
            result.append(func(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001
            error.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)
    if t.is_alive():
        # 超时：线程仍在跑，主流程抛异常
        raise ResearchAgentError(
            f"阶段超时（{timeout_seconds:.0f}s）: {func.__name__}",
            recoverable=True,
        )
    if error:
        raise error[0]
    return result[0]  # type: ignore[no-any-return]


def stage_timeout(seconds: float, fallback: Any = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """阶段超时装饰器：超时则记录告警并返回 fallback，不中断主流程。

    用法：
        @stage_timeout(60, fallback="")
        def generate(self, task): ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return run_with_timeout(func, seconds, *args, **kwargs)
            except ResearchAgentError as e:
                logger.warning(f"阶段 {func.__name__} 超时降级: {e}")
                if fallback is not None:
                    return fallback
                raise

        return wrapper

    return decorator
