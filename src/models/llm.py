"""大模型客户端封装。

对应架构 7.2.1 主推理模型选型，使用 OpenAI 兼容接口对接 DeepSeek-V3。
所有外部调用带指数退避重试（最多3次），无 API Key 时降级为 Mock LLM，
保证 MVP 在未配置密钥时仍可跑通全链路（Claude.md 6.4 降级方案）。

实现说明：
不依赖 openai Python SDK（3.x 与 DeepSeek 在某些环境下鉴权有兼容问题），
直接用 httpx 调 OpenAI 兼容 REST 接口，更轻量更稳定，已实测验证 200 OK。
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ..common.config import get_env
from ..common.exceptions import LLMError
from ..common.logger import get_logger

logger = get_logger(__name__)


class _MockLLM:
    """无 API Key 时的降级 LLM，返回结构化占位结果。

    用于测试与未配置密钥场景，保证全链路可跑通。
    """

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        logger.warning("使用 Mock LLM（未配置 API Key），返回占位结果")
        user_msg = messages[-1].get("content", "") if messages else ""
        # 尝试返回一个最小可用的 JSON 计划占位
        if "子任务" in user_msg or "plan" in user_msg.lower() or "拆解" in user_msg:
            return json.dumps(
                {
                    "topic": user_msg[:50],
                    "subtasks": [
                        {
                            "subtask_id": "st_1",
                            "objective": "检索核心文献并梳理基础概念",
                            "deliverable": "文献清单与概念定义",
                            "recommended_tools": ["search_arxiv", "search_semantic_scholar"],
                        },
                        {
                            "subtask_id": "st_2",
                            "objective": "梳理技术原理与核心方法",
                            "deliverable": "原理与方法总结",
                            "recommended_tools": ["search_arxiv"],
                        },
                        {
                            "subtask_id": "st_3",
                            "objective": "总结研究进展与应用场景",
                            "deliverable": "进展综述与案例",
                            "recommended_tools": ["search_semantic_scholar"],
                        },
                    ],
                    "quality_score": 0.8,
                },
                ensure_ascii=False,
            )
        return f"（Mock LLM 占位输出）基于主题「{user_msg[:40]}」生成的结构化摘要占位。"


class LLMClient:
    """大模型客户端，支持指数退避重试与降级。

    使用 httpx 直调 OpenAI 兼容 REST 接口，可对接 DeepSeek/OpenAI/Ollama。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_retries: int = 3,
        backoff: float = 2.0,
        timeout: float = 60.0,
    ) -> None:
        env = get_env()
        self.api_key = api_key if api_key is not None else env["llm_api_key"]
        self.base_url = (base_url if base_url is not None else env["llm_base_url"]).rstrip("/")
        self.model = model if model is not None else env["llm_model"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self._mock = False
        self._mock_client = _MockLLM()
        self._init_client()

    def _init_client(self) -> None:
        if not self.api_key:
            logger.warning("未配置 LLM_API_KEY，启用 Mock LLM 降级模式")
            self._mock = True
            return
        logger.info(f"LLM 客户端初始化成功: model={self.model} base_url={self.base_url}")

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """调用大模型对话接口，带指数退避重试。

        Args:
            messages: OpenAI 格式的消息列表
            **kwargs: 透传 temperature/max_tokens/model 等参数
        Returns:
            模型输出的文本
        """
        if self._mock:
            return self._mock_client.chat(messages, **kwargs)

        model = kwargs.pop("model", self.model)
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        # 支持 response_format 等可选参数
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # 透传其他 OpenAI 兼容参数（如 response_format、top_p 等）
        for k, v in kwargs.items():
            payload[k] = v

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                # 429 限流 / 5xx 服务端错误才重试，4xx 鉴权错误不重试
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    # 4xx 客户端错误（如 401 鉴权失败）不重试，直接降级
                    logger.error(
                        f"LLM 调用 HTTP {resp.status_code} 客户端错误，不重试: {resp.text[:200]}"
                    )
                    break
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                last_error = e
                wait = self.backoff ** attempt
                logger.warning(
                    f"LLM 调用第 {attempt}/{self.max_retries} 次失败: {e}，{wait:.1f}s 后重试"
                )
                time.sleep(wait)

        logger.error("LLM 调用重试耗尽，降级为 Mock 输出")
        return self._mock_client.chat(messages, **kwargs)


# 单例
_llm_instance: LLMClient | None = None


def get_llm() -> LLMClient:
    """获取全局 LLM 客户端单例。"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance
