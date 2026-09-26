"""模型封装层。

对应架构第 2.1 节智能决策层（大模型 + JEV）的封装入口，
统一对外提供 LLM 调用与嵌入向量生成能力。
"""
from .llm import LLMClient, get_llm
from .embedding import EmbeddingClient, get_embedding

__all__ = ["LLMClient", "get_llm", "EmbeddingClient", "get_embedding"]
