"""嵌入模型客户端封装。

对应架构 7.2.3 嵌入模型选型，优先使用 Ollama BGE-M3，
无 Ollama 时自动降级到 sentence-transformers（all-MiniLM-L6-v2），
保证无 GPU/Docker 环境下 MVP 仍可运行。
"""

from __future__ import annotations

import time
from typing import Any

from ..common.config import get_env
from ..common.exceptions import EmbeddingError
from ..common.logger import get_logger

logger = get_logger(__name__)

# 常见嵌入模型维度，用于初始化向量库
EMBEDDING_DIMS: dict[str, int] = {
    "bge-m3": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "text-embedding-3-small": 1536,
}


class _SentenceTransformerEmbedder:
    """基于 sentence-transformers 的本地嵌入降级方案。"""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "未安装 sentence-transformers，请 pip install sentence-transformers", recoverable=False
            ) from e
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        logger.info(f"sentence-transformers 嵌入模型加载成功: {model_name} dim={self._dim}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    @property
    def dim(self) -> int:
        return self._dim


class _HashingEmbedder:
    """无任何模型依赖时的最终降级嵌入器。

    基于 word n-gram 哈希，虽非语义向量但可支撑关键词重叠召回，
    保证无 GPU/Docker/模型依赖时两层记忆仍可写入与召回（架构 6.3 模型级降级）。
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        logger.warning("使用 Hashing 降级嵌入器（无语义能力，仅关键词召回）")

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        import math
        import re

        results: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            tokens = re.findall(r"\w+", text.lower())
            # unigram + bigram
            grams = tokens + ["_".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
            for g in grams:
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
                vec[h % self._dim] += 1.0
            # L2 归一化
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            results.append([v / norm for v in vec])
        return results

    @property
    def dim(self) -> int:
        return self._dim



class EmbeddingClient:
    """嵌入模型客户端，带降级策略。

    优先尝试 Ollama（OpenAI 兼容接口），
    失败则降级到 sentence-transformers 本地模型。
    """

    def __init__(self, timeout: float = 30.0) -> None:
        env = get_env()
        self.provider = env["embedding_provider"]
        self.model = env["embedding_model"]
        self.base_url = env["embedding_base_url"].rstrip("/")
        self.api_key = env["embedding_api_key"]
        self.fallback_model = env["embedding_fallback_model"]
        self.timeout = timeout  # Ollama / 远端嵌入调用超时
        self._client: Any = None
        self._st: _SentenceTransformerEmbedder | None = None
        self._hasher: _HashingEmbedder | None = None
        self._dim: int = 0
        self._init_client()

    def _init_client(self) -> None:
        if self.provider == "ollama":
            self._try_ollama()
        else:
            self._init_sentence_transformers(self.fallback_model)

        if self._dim == 0:
            # 兜底从已知维度表取
            self._dim = EMBEDDING_DIMS.get(
                self.model, EMBEDDING_DIMS.get(self.fallback_model, 384)
            )

    def _try_ollama(self) -> None:
        """用 httpx 直调 Ollama 嵌入接口（与 LLM 一致，避免 openai SDK 兼容问题）。"""
        try:
            import httpx

            url = f"{self.base_url}/embeddings"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {"model": self.model, "input": ["probe"]}
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            self._dim = len(data["data"][0]["embedding"])
            logger.info(f"Ollama 嵌入就绪: model={self.model} dim={self._dim}")
        except Exception as e:
            logger.warning(
                f"Ollama 嵌入不可用（{e}），降级到 sentence-transformers"
            )
            self._init_sentence_transformers(self.fallback_model)

    def _init_sentence_transformers(self, model_name: str) -> None:
        try:
            self._st = _SentenceTransformerEmbedder(model_name)
            self._dim = self._st.dim
        except Exception as e:
            logger.warning(f"sentence-transformers 不可用（{e}），降级到 Hashing 嵌入器")
            self._hasher = _HashingEmbedder()
            self._dim = self._hasher.dim

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """生成嵌入向量，支持单条或批量。"""
        single = isinstance(texts, str)
        inputs = [texts] if single else texts
        if self._st is not None:
            return self._st.embed(inputs)
        if self._hasher is not None:
            return self._hasher.embed(inputs)
        # Ollama 路径（httpx 直调 + 超时控制）
        import httpx

        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": inputs}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return [d["embedding"] for d in data["data"]]
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(f"嵌入调用第 {attempt+1}/3 次失败: {e}，{wait}s 后重试")
                time.sleep(wait)
        raise EmbeddingError(f"嵌入生成失败: {last_error}")

    @property
    def dim(self) -> int:
        return self._dim


# 单例
_embedding_instance: EmbeddingClient | None = None


def get_embedding() -> EmbeddingClient:
    """获取全局嵌入客户端单例。"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingClient()
    return _embedding_instance
