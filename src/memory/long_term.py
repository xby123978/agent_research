"""长期记忆模块。

对应架构 3.3.1 长期记忆层：
- 存储内容：历史研究成果、知识点、实体关系、文献索引
- 存储介质：Qdrant（本地持久化 / Server）+ 内置向量库兜底
- 生命周期：永久存储，增量更新
- 召回策略：混合召回：向量 60% + 关键词(BM25) 20% + 图谱 20%（加权 RRF 融合）

存储后端选择（架构 6.3，三级降级）：
1. 配置 QDRANT_URL  → Qdrant Server 模式（需 Docker/远程服务，支持多进程）
2. 配置 QDRANT_PATH → Qdrant 本地持久化模式（默认，无需 Docker，数据落盘）
                      特殊值 ":memory:" 表示纯内存模式（测试用，不持久化）
3. 以上均不可用     → 内置纯 Python 向量库（内存，无持久化，最后兜底）

注意：Qdrant 本地持久化模式为单进程独占（数据目录加锁）。
     多进程场景（如同时运行 API 服务与 Web UI）请改用 Server 模式。
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..common.config import get_env, get_settings
from ..common.data_models import MemoryItem
from ..common.exceptions import MemoryError
from ..common.logger import get_logger
from .base import BaseMemory

logger = get_logger(__name__)

# Qdrant 点 id 必须为 UUID，用固定命名空间把业务 id 确定性映射为 UUID
_QDRANT_NS = uuid.UUID("6f2e6c1a-0000-4000-8000-000000000001")


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
        self._qdrant_local = False  # 是否本地持久化模式
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
        path = (env.get("qdrant_path") or "").strip()
        if not path:
            # 未显式配置时，默认落到 data_dir/qdrant（本地持久化，无需 Docker）
            path = str(Path(env.get("data_dir", "./data")) / "qdrant")

        # 先确定嵌入维度，用于初始化集合
        try:
            dim = self._get_embedding().dim
        except Exception as e:
            logger.warning(f"嵌入维度获取失败，使用默认 384: {e}")
            dim = 384
        self._dim = dim

        # 1. Server 模式（多进程/大规模，需 Docker 或远程服务）
        if url and self._init_qdrant_server(url, env, dim):
            return
        # 2. 本地持久化模式（单机默认，无需 Docker，数据落盘）
        if path and self._init_qdrant_local(path, dim):
            return
        # 3. 兜底：内置内存向量库（无持久化）
        self._local = _LocalVectorStore(dim)

    def _init_qdrant_server(self, url: str, env: dict, dim: int) -> bool:
        """连接 Qdrant Server（需 Docker/远程服务）。"""
        try:
            from qdrant_client import QdrantClient

            # Qdrant 客户端超时控制（秒）：连接/读写均受限
            client = QdrantClient(
                url=url,
                api_key=env.get("qdrant_api_key") or None,
                timeout=30,
            )
            self._ensure_collection(client, dim)
            self._qdrant = client
            self._qdrant_local = False
            logger.info(
                f"Qdrant Server 就绪: {url} collection={self.collection} dim={dim}"
            )
            return True
        except Exception as e:
            logger.warning(f"Qdrant Server 不可用（{e}），尝试本地持久化模式")
            self._qdrant = None  # 确保重置，防止脏对象残留
            return False

    def _init_qdrant_local(self, path: str, dim: int) -> bool:
        """启动 Qdrant 本地持久化模式（无需 Docker，数据落盘）。

        特殊值 ":memory:" 使用纯内存模式（测试用，不持久化）。
        """
        try:
            from qdrant_client import QdrantClient

            if path == ":memory:":
                client = QdrantClient(":memory:")
                target = "内存模式（不持久化）"
            else:
                Path(path).mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=path)
                target = path
            self._ensure_collection(client, dim)
            self._qdrant = client
            self._qdrant_local = True
            logger.info(
                f"Qdrant 本地模式就绪: {target} collection={self.collection} dim={dim}"
            )
            return True
        except Exception as e:
            logger.warning(f"Qdrant 本地模式不可用（{e}），降级到内置向量库（无持久化）")
            self._qdrant = None
            return False

    def _ensure_collection(self, client: Any, dim: int) -> None:
        """确保集合存在（兼容新旧 qdrant-client API）。"""
        from qdrant_client.http.models import Distance, VectorParams

        vectors_config = VectorParams(size=dim, distance=Distance.COSINE)
        try:
            exists = client.collection_exists(self.collection)
        except Exception:
            exists = None
        if exists:
            return
        if exists is False:
            client.create_collection(
                collection_name=self.collection, vectors_config=vectors_config
            )
            return
        # 旧版本无 collection_exists：get 失败则重建
        try:
            client.get_collection(self.collection)
        except Exception:
            client.recreate_collection(
                collection_name=self.collection, vectors_config=vectors_config
            )

    @staticmethod
    def _point_id(item_id: str) -> str:
        """业务 item_id → Qdrant 合法点 id（UUID，确定性映射）。"""
        return str(uuid.uuid5(_QDRANT_NS, item_id))

    def _qdrant_query(self, vector: list[float], limit: int) -> list[Any]:
        """Qdrant 向量检索（兼容 query_points 新 API 与旧版 search）。"""
        if hasattr(self._qdrant, "query_points"):
            res = self._qdrant.query_points(
                collection_name=self.collection, query=vector, limit=limit
            )
            return list(res.points)
        return list(
            self._qdrant.search(
                collection_name=self.collection, query_vector=vector, limit=limit
            )
        )

    def _all_docs(self) -> list[tuple[str, dict[str, Any]]]:
        """拉取全部文档 (业务id, payload)，供 BM25 关键词召回使用。

        统一后端差异：Qdrant 用 scroll 拉取，兜底库直接读内存列表。
        """
        if self._qdrant is not None:
            try:
                records, _next = self._qdrant.scroll(
                    collection_name=self.collection,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False,
                )
                return [
                    (str((r.payload or {}).get("_id") or r.id), r.payload or {})
                    for r in records
                ]
            except Exception as e:
                logger.warning(f"长期记忆全量拉取失败: {e}")
                return []
        if self._local is not None:
            return list(zip(self._local._ids, self._local._payloads))
        return []

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
            "_id": item.item_id,  # 业务 id（Qdrant 点 id 必须为 UUID，故另存业务 id）
        }

        if self._qdrant is not None:
            from qdrant_client.http.models import PointStruct

            point = PointStruct(
                id=self._point_id(item.item_id), vector=vector, payload=payload
            )
            try:
                self._qdrant.upsert(collection_name=self.collection, points=[point])
            except Exception as e:
                # 不切换后端（避免丢弃已持久化数据），仅记录并跳过本条
                logger.warning(f"Qdrant 写入失败（本条跳过，后端保持）: {e}")
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
                points = self._qdrant_query(vector, 1)
                if points and float(points[0].score) >= threshold:
                    return str((points[0].payload or {}).get("_id") or points[0].id)
            except Exception as e:
                logger.warning(f"长期记忆去重查询失败: {e}")
                return None
        elif self._local is not None:
            hits = self._local.search(vector, top_k=1)
            if hits and hits[0][1] >= threshold:
                return hits[0][0]
        return None

    # RRF 平滑常数：业界标准 k=60，缓解头部排名差异
    _RRF_K = 60

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """混合召回（架构 3.3.1）：向量 60% + 关键词 20% + 图谱 20%。

        融合方法：加权 RRF (Reciprocal Rank Fusion)
            fused_score(mid) = Σ_path  w_path / (k + rank_path(mid))

        优势（相比 CombSUM 加权求和）：
        - 只看每路排名，完全无视分数量纲，天然抗 BM25/余弦/图谱分异构
        - 加权系数 w_path 真正起作用，不会被某路高分污染
        - k=60 平滑头部排名差异（rank=1 与 rank=2 差距不至于过大）

        降级：某路召回为空时该路权重归零并重新归一化其他路权重。
        """
        settings = get_settings().get("memory", {}).get("long_term", {})
        weights: dict[str, float] = settings.get("recall_weights", {
            "vector": 0.6, "keyword": 0.2, "graph": 0.2,
        })

        # 三路召回（取 2*top_k 扩大候选池，保证 RRF 有足够候选）
        candidate_k = max(top_k * 2, 10)
        vector_hits = self._recall_vector(query, top_k=candidate_k)
        keyword_hits = self._recall_keyword(query, top_k=candidate_k)
        graph_hits = self._recall_graph(query, top_k=candidate_k)

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
        weights = {k: v / total_w for k, v in weights.items()}

        # 构建每路的 rank 表：mid → rank(从 1 开始)
        # 命中越靠前 rank 越小，RRF 分数越大
        def _to_rank(
            hits: list[tuple[float, dict[str, Any]]],
        ) -> dict[str, int]:
            return {
                payload.get("_id", ""): idx + 1
                for idx, (_s, payload) in enumerate(hits)
                if payload.get("_id", "")
            }

        rank_vector = _to_rank(vector_hits) if weights["vector"] > 0 else {}
        rank_keyword = _to_rank(keyword_hits) if weights["keyword"] > 0 else {}
        rank_graph = _to_rank(graph_hits) if weights["graph"] > 0 else {}

        # 收集所有候选 mid（取并集）
        all_mids: set[str] = set()
        all_mids.update(rank_vector.keys())
        all_mids.update(rank_keyword.keys())
        all_mids.update(rank_graph.keys())

        # 构建 mid → payload 映射（取首个出现的 payload，优先级 vector > keyword > graph）
        payload_map: dict[str, dict[str, Any]] = {}
        for _s, p in vector_hits:
            mid = p.get("_id", "")
            if mid and mid not in payload_map:
                payload_map[mid] = p
        for _s, p in keyword_hits:
            mid = p.get("_id", "")
            if mid and mid not in payload_map:
                payload_map[mid] = p
        for _s, p in graph_hits:
            mid = p.get("_id", "")
            if mid and mid not in payload_map:
                payload_map[mid] = p

        # RRF 融合：fused = Σ w_path / (k + rank_path)
        k = self._RRF_K
        fused: dict[str, float] = {}
        for mid in all_mids:
            score = 0.0
            if (r := rank_vector.get(mid)) is not None:
                score += weights["vector"] / (k + r)
            if (r := rank_keyword.get(mid)) is not None:
                score += weights["keyword"] / (k + r)
            if (r := rank_graph.get(mid)) is not None:
                score += weights["graph"] / (k + r)
            fused[mid] = score

        # 按 RRF 分数降序取 top_k
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out: list[MemoryItem] = []
        for item_id, score in ranked:
            payload = payload_map.get(item_id, {})
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
                points = self._qdrant_query(query_vec, top_k)
                hits = [
                    (
                        str((p.payload or {}).get("_id") or p.id),
                        float(p.score),
                        p.payload or {},
                    )
                    for p in points
                ]
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

        文档来源统一走 _all_docs()：Qdrant 用 scroll 拉取，兜底库读内存列表。
        """
        # 收集所有文档（mid, payload, 分词列表, 文档长度）
        docs: list[tuple[str, dict[str, Any], list[str], int]] = []
        for mid, p in self._all_docs():
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
        """图谱关联召回：多因子评分 = 路径分 × 节点重要性 × 查询相关性。

        五个因子（详见 WorkingMemory.graph_search 文档）：
        1. 边权重 — 建边时的实例强度
        2. 路径长度衰减 — decay^hop
        3. 节点重要性 — 度中心性
        4. 关系类型权重 — cites/supports/contradicts/similar/related
        5. 查询相关性 — 由本方法注入语义相似度回调（embedding 余弦）

        降级：工作记忆不可用时返回空（该路权重归零）。
        """
        try:
            from .working import WorkingMemory

            wm = WorkingMemory()

            # 构建语义相关性回调：用 embedding 余弦相似度
            # 降级：embedding 不可用时返回 None，graph_search 内部用关键词重叠率兜底
            relevance_fn: Any = None
            try:
                emb = self._get_embedding()
                query_vec = emb.embed(query)
                if isinstance(query_vec[0], list):
                    query_vec = query_vec[0]
                import numpy as np

                qv = np.asarray(query_vec, dtype=np.float32)
                qn = float(np.linalg.norm(qv)) or 1.0

                def _semantic_relevance(_nid: str, content: str) -> float:
                    try:
                        cv = emb.embed(content)
                        if isinstance(cv[0], list):
                            cv = cv[0]
                        cv_arr = np.asarray(cv, dtype=np.float32)
                        cn = float(np.linalg.norm(cv_arr)) or 1.0
                        return float(np.dot(qv, cv_arr) / (qn * cn))
                    except Exception:
                        return 0.0

                relevance_fn = _semantic_relevance
            except Exception as e:
                logger.debug(f"图谱召回语义回调降级为关键词匹配: {e}")

            related = wm.graph_search(
                query,
                top_k=top_k,
                max_depth=2,
                query_relevance_fn=relevance_fn,
            )
            scored: list[tuple[float, dict[str, Any]]] = []
            for score, item in related:
                if item.item_id and item.content:
                    scored.append((
                        score,
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
                points = self._qdrant.retrieve(
                    collection_name=self.collection, ids=[self._point_id(item_id)]
                )
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
