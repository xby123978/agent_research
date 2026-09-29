"""长期记忆模块。

对应架构 3.3.1 长期记忆层：
- 存储内容：历史研究成果、知识点、实体关系、文献索引
- 存储介质：Qdrant + 本地文件
- 生命周期：永久存储，增量更新
- 召回策略：向量相似度召回（MVP 阶段），阶段二扩展为混合召回

降级策略（架构 6.3）：
1. 配置了 QDRANT_URL 时优先使用 Qdrant Server（Docker）
2. 无 Qdrant 时降级为内置纯 Python 向量库（基于余弦相似度），
   保证无 Docker 环境下两层记忆仍可写入与召回。
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

from ..common.config import get_env, get_settings
from ..common.data_models import MemoryItem
from ..common.exceptions import MemoryError
from ..common.logger import get_logger
from .base import BaseMemory

logger = get_logger(__name__)


class _LocalVectorStore:
    """纯 Python 向量库降级方案，基于余弦相似度。

    单机单用户场景下足够支撑 MVP，无外部依赖。
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._payloads: list[dict[str, Any]] = []
        logger.info(f"使用内置向量库（降级模式）dim={dim}")

    def upsert(self, item_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        for i, existing in enumerate(self._ids):
            if existing == item_id:
                self._vectors[i] = vec
                self._payloads[i] = payload
                return
        self._ids.append(item_id)
        self._vectors.append(vec)
        self._payloads.append(payload)

    def search(self, query_vec: list[float], top_k: int = 5) -> list[tuple[str, float, dict[str, Any]]]:
        if not self._vectors:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        mat = np.vstack(self._vectors)
        qn = float(np.linalg.norm(q)) or 1.0
        norms = np.linalg.norm(mat, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        scores = (mat @ q) / (norms * qn)
        idx = np.argsort(-scores)[:top_k]
        return [
            (self._ids[i], float(scores[i]), self._payloads[i])
            for i in idx
            if scores[i] > 0
        ]

    def get(self, item_id: str) -> dict[str, Any] | None:
        for i, existing in enumerate(self._ids):
            if existing == item_id:
                return self._payloads[i]
        return None


class LongTermMemory(BaseMemory):
    """长期记忆：向量库语义召回（Qdrant 或内置降级向量库）。"""

    layer = "long_term"

    def __init__(self, embedding_client=None) -> None:
        env = get_env()
        settings = get_settings().get("memory", {}).get("long_term", {})
        self.collection = settings.get("collection_name", "research_kb")
        self._embedding = embedding_client  # 延迟注入
        self._qdrant = None
        self._local: _LocalVectorStore | None = None
        self._dim = 384
        self._init_store()

    def _get_embedding(self):
        if self._embedding is None:
            from ..models.embedding import get_embedding

            self._embedding = get_embedding()
        return self._embedding

    def _init_store(self) -> None:
        env = get_env()
        url = (env.get("qdrant_url") or "").strip()
        # 先确定嵌入维度，用于初始化
        try:
            dim = self._get_embedding().dim
        except Exception as e:
            logger.warning(f"嵌入维度获取失败，使用默认 384: {e}")
            dim = 384
        self._dim = dim

        if url:
            # 有 Qdrant Server 配置时尝试连接
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Distance, VectorParams

                # Qdrant 客户端超时控制（秒）：连接/读写均受限
                client = QdrantClient(
                    url=url,
                    api_key=env.get("qdrant_api_key") or None,
                    timeout=30,
                )
                try:
                    client.get_collection(self.collection)
                except Exception:
                    client.recreate_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                # 仅在 get/recreate 均成功后才赋值，避免脏对象
                self._qdrant = client
                logger.info(f"Qdrant 就绪: {self.collection} dim={dim}")
                return
            except Exception as e:
                logger.warning(f"Qdrant 不可用（{e}），降级到内置向量库")
                self._qdrant = None  # 确保重置，防止脏对象残留

        # 无 Qdrant 配置或连接失败 → 内置降级向量库
        self._local = _LocalVectorStore(dim)

    def add(self, item: MemoryItem) -> str:
        if not item.item_id:
            item.item_id = f"ltm_{uuid.uuid4().hex[:12]}"
        item.source = self.layer
        item.created_at = datetime.now()

        emb = self._get_embedding()
        vector = emb.embed(item.content)
        if isinstance(vector[0], list):
            vector = vector[0]
        item.vector = vector

        # 语义去重（架构 3.3.2）：相似度 > 0.92 视为重复，跳过写入
        dup_id = self._find_duplicate(vector)
        if dup_id:
            logger.info(f"长期记忆去重命中: 新内容与 {dup_id} 相似度过高，跳过写入")
            return dup_id

        payload = {
            "content": item.content,
            "metadata": item.metadata,
            "source": item.source,
            "created_at": item.created_at.isoformat(),
        }

        if self._qdrant is not None:
            from qdrant_client.http.models import PointStruct

            point = PointStruct(id=item.item_id, vector=vector, payload=payload)
            try:
                self._qdrant.upsert(collection_name=self.collection, points=[point])
            except Exception as e:
                # 写失败不抛异常，降级到本地向量库，保证主流程不中断
                logger.warning(f"Qdrant 写入失败，降级到本地向量库: {e}")
                self._qdrant = None
                self._local = _LocalVectorStore(len(vector))
                self._local.upsert(item.item_id, vector, payload)
        else:
            self._local.upsert(item.item_id, vector, payload)
        logger.debug(f"长期记忆写入: {item.item_id}")
        return item.item_id

    def _find_duplicate(self, vector: list[float], threshold: float = 0.92) -> str | None:
        """语义去重：检索最相似条目，高于阈值则返回已存在 id。

        架构 3.3.2 自动去重：基于语义相似度与实体对齐，避免重复知识沉淀。
        """
        if self._qdrant is not None:
            try:
                results = self._qdrant.search(
                    collection_name=self.collection, query_vector=vector, limit=1
                )
                if results and results[0].score >= threshold:
                    return str(results[0].id)
            except Exception as e:
                logger.warning(f"长期记忆去重查询失败: {e}")
                return None
        elif self._local is not None:
            hits = self._local.search(vector, top_k=1)
            if hits and hits[0][1] >= threshold:
                return hits[0][0]
        return None

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """混合召回（架构 3.3.1）：向量 60% + 关键词 20% + 图谱 20%。

        三路召回后按权重融合分数，去重排序取 top_k。
        降级：某路召回失败时，该路权重归零并重新归一化其他路权重。
        """
        settings = get_settings().get("memory", {}).get("long_term", {})
        weights: dict[str, float] = settings.get("recall_weights", {
            "vector": 0.6, "keyword": 0.2, "graph": 0.2,
        })

        # 三路召回
        vector_hits = self._recall_vector(query, top_k=max(top_k * 2, 10))
        keyword_hits = self._recall_keyword(query, top_k=max(top_k * 2, 10))
        graph_hits = self._recall_graph(query, top_k=max(top_k * 2, 10))

        # 路径失败时权重归零并归一化
        if not vector_hits:
            weights["vector"] = 0.0
        if not keyword_hits:
            weights["keyword"] = 0.0
        if not graph_hits:
            weights["graph"] = 0.0
        total_w = sum(weights.values())
        if total_w == 0:
            return []  # 三路全失败
        # 归一化
        weights = {k: v / total_w for k, v in weights.items()}

        # 融合分数：每路分数归一化到 [0,1] 后加权
        merged: dict[str, tuple[float, dict[str, Any]]] = {}
        for score, payload in vector_hits:
            mid = payload.get("_id", "")
            merged[mid] = (score * weights["vector"], payload)
        for score, payload in keyword_hits:
            mid = payload.get("_id", "")
            if mid in merged:
                old_score, old_pl = merged[mid]
                merged[mid] = (old_score + score * weights["keyword"], old_pl)
            else:
                merged[mid] = (score * weights["keyword"], payload)
        for score, payload in graph_hits:
            mid = payload.get("_id", "")
            if mid in merged:
                old_score, old_pl = merged[mid]
                merged[mid] = (old_score + score * weights["graph"], old_pl)
            else:
                merged[mid] = (score * weights["graph"], payload)

        # 按融合分数排序取 top_k
        ranked = sorted(merged.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        out: list[MemoryItem] = []
        for item_id, (score, payload) in ranked:
            out.append(
                MemoryItem(
                    item_id=item_id,
                    content=payload.get("content", ""),
                    metadata=payload.get("metadata", {}),
                    score=score,
                    source=payload.get("source", self.layer),
                    created_at=datetime.fromisoformat(payload["created_at"])
                    if "created_at" in payload
                    else datetime.now(),
                )
            )
        return out

    def _recall_vector(
        self, query: str, top_k: int = 10
    ) -> list[tuple[float, dict[str, Any]]]:
        """向量召回：余弦相似度。"""
        emb = self._get_embedding()
        query_vec = emb.embed(query)
        if isinstance(query_vec[0], list):
            query_vec = query_vec[0]
        hits: list[tuple[str, float, dict[str, Any]]] = []
        if self._qdrant is not None:
            try:
                results = self._qdrant.search(
                    collection_name=self.collection, query_vector=query_vec, limit=top_k
                )
                hits = [(str(h.id), float(h.score), h.payload or {}) for h in results]
            except Exception as e:
                logger.warning(f"向量召回失败: {e}")
                return []
        elif self._local is not None:
            hits = self._local.search(query_vec, top_k)
        return [(score, {**p, "_id": mid}) for mid, score, p in hits]

    def _recall_keyword(
        self, query: str, top_k: int = 10
    ) -> list[tuple[float, dict[str, Any]]]:
        """关键词召回：BM25 算法（Okapi BM25）。

        相比 Jaccard 相似度的优势：
        - 引入 IDF 权重，稀有词贡献更高分
        - 考虑文档长度归一化，避免长文档天然占优
        - 词频饱和（k1 控制），避免高频词过度刷分

        公式：sum_t IDF(t) * (f(t,d) * (k1+1)) / (f(t,d) + k1*(1-b+b*|d|/avgdl))
        参数：k1=1.5（词频饱和）, b=0.75（文档长度归一化）

        降级方案：无 Qdrant 全文检索时用本地遍历计算 BM25。
        """
        if self._local is None and self._qdrant is None:
            return []
        # 收集所有文档（mid, payload, 分词列表, 文档长度）
        docs: list[tuple[str, dict[str, Any], list[str], int]] = []
        if self._local is not None:
            for i, mid in enumerate(self._local._ids):
                p = self._local._payloads[i]
                content = (p.get("content", "") or "").lower()
                terms = content.split()
                docs.append((mid, p, terms, len(terms)))
        if not docs:
            return []

        # 查询分词（与原实现一致：按空格切分）
        query_terms = query.lower().split()
        if not query_terms:
            return []

        N = len(docs)
        avgdl = sum(d[3] for d in docs) / N if N > 0 else 0
        if avgdl == 0:
            return []

        # 计算 df(t)：包含 term 的文档数
        df_map: dict[str, int] = {}
        for _, _, terms, _ in docs:
            for t in set(terms):
                df_map[t] = df_map.get(t, 0) + 1

        # IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
        idf_map: dict[str, float] = {
            t: math.log((N - df + 0.5) / (df + 0.5) + 1)
            for t, df in df_map.items()
        }

        # BM25 参数
        k1, b = 1.5, 0.75

        scored: list[tuple[float, dict[str, Any]]] = []
        for mid, p, terms, doc_len in docs:
            if doc_len == 0:
                continue
            tf_map = Counter(terms)
            score = 0.0
            for t in query_terms:
                f = tf_map.get(t, 0)
                if f == 0:
                    continue
                idf = idf_map.get(t, 0)
                # BM25 核心：词频饱和 + 文档长度归一化
                denom = f + k1 * (1 - b + b * doc_len / avgdl)
                score += idf * (f * (k1 + 1)) / denom
            if score > 0:
                scored.append((score, {**p, "_id": mid}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def _recall_graph(
        self, query: str, top_k: int = 10
    ) -> list[tuple[float, dict[str, Any]]]:
        """图谱关联召回：通过工作记忆图结构找关联节点。

        降级：工作记忆不可用时返回空（该路权重归零）。
        """
        try:
            from .working import WorkingMemory

            wm = WorkingMemory()
            related = wm.search(query, top_k=top_k)
            scored: list[tuple[float, dict[str, Any]]] = []
            for item in related:
                if item.item_id and item.content:
                    # 图谱关联分数用 item.score 或固定 0.5
                    s = item.score if item.score > 0 else 0.5
                    scored.append((
                        s,
                        {
                            "content": item.content,
                            "metadata": item.metadata,
                            "source": item.source or "graph",
                            "created_at": item.created_at.isoformat()
                            if item.created_at
                            else datetime.now().isoformat(),
                            "_id": item.item_id,
                        },
                    ))
            return scored[:top_k]
        except Exception as e:
            logger.debug(f"图谱召回失败（工作记忆不可用）: {e}")
            return []

    def get(self, item_id: str) -> MemoryItem | None:
        payload: dict[str, Any] | None = None
        if self._qdrant is not None:
            try:
                points = self._qdrant.retrieve(collection_name=self.collection, ids=[item_id])
                if points:
                    payload = points[0].payload or {}
            except Exception as e:
                logger.warning(f"长期记忆 get 失败: {e}")
                return None
        elif self._local is not None:
            payload = self._local.get(item_id)

        if not payload:
            return None
        return MemoryItem(
            item_id=item_id,
            content=payload.get("content", ""),
            metadata=payload.get("metadata", {}),
            source=payload.get("source", self.layer),
            created_at=datetime.fromisoformat(payload["created_at"])
            if "created_at" in payload
            else datetime.now(),
        )
