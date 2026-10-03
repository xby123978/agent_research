"""短期记忆模块。

对应架构 3.3.1 短期记忆层：
- 存储内容：会话历史、推理过程、工具调用日志
- 存储介质：内存 + SQLite
- 召回策略：滑动窗口 + 摘要压缩
- 生命周期：会话结束后保留 7 天

阶段三升级（方案 B）：
- 最近 N 轮保留原文（滑动窗口）
- 超出窗口的旧轮次 LLM 摘成 1-2 段摘要常驻
- 召回时返回 (recent_items, summary) 二元组
- token 预算感知：按剩余 token 数动态决定塞多少条
- 内存/SQLite 优先级修复：内存不足时回填 SQLite
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from ..common.config import get_env, get_settings
from ..common.data_models import MemoryItem
from ..common.exceptions import MemoryError
from ..common.logger import get_logger
from .base import BaseMemory

logger = get_logger(__name__)


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token。"""
    if not text:
        return 0
    cn_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_count = len(text) - cn_count
    return cn_count + other_count // 4 + 1


class ShortTermMemory(BaseMemory):
    """短期记忆：滑动窗口 + 摘要压缩 + token 预算感知。

    数据结构：
    - _buffer: 最近 N 轮原文（滑动窗口，默认 20）
    - _summary: 旧轮次累积摘要（LLM 压缩生成）
    - _summary_covered_ids: 摘要已覆盖的 item_id 集合，避免重复摘取
    - SQLite: 全量持久化，仅按 retention_days 过期
    """

    layer = "short_term"

    def __init__(self, db_path: str | None = None) -> None:
        env = get_env()
        settings = get_settings().get("memory", {}).get("short_term", {})
        self.retention_days = int(settings.get("retention_days", 7))
        # 滑动窗口：内存保留最近 N 轮原文
        self.max_window = int(settings.get("max_window", 20))
        # 触发摘要的轮次阈值：窗口满后再积累 N 条触发一次摘要压缩
        self.summary_trigger = int(settings.get("summary_trigger", 5))
        # 摘要 token 预算（单次召回中摘要部分最多占用的 token 数）
        self.summary_token_budget = int(settings.get("summary_token_budget", 800))
        # 摘要最大字符数（防止单次摘要过长）
        self.summary_max_chars = int(settings.get("summary_max_chars", 1200))

        data_dir = Path(env["data_dir"]) / "sqlite"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(data_dir / "short_term.db")
        self._lock = threading.Lock()
        self._buffer: list[MemoryItem] = []
        self._summary: str = ""
        self._summary_covered_ids: set[str] = set()
        self._init_db()
        self._load_persisted_state()

    # ===== 数据库 =====

    def _init_db(self) -> None:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS memory_items (
                        item_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        source TEXT,
                        created_at TEXT
                    )"""
                )
                # 摘要表：单行表，存最新累积摘要
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS short_term_summary (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        summary TEXT NOT NULL,
                        covered_ids TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )"""
                )
        except Exception as e:
            raise MemoryError(f"短期记忆数据库初始化失败: {e}") from e

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _load_persisted_state(self) -> None:
        """重启后从 SQLite 恢复摘要 + 最近窗口。"""
        try:
            with self._get_conn() as conn:
                # 恢复摘要
                row = conn.execute(
                    "SELECT summary, covered_ids FROM short_term_summary WHERE id = 1"
                ).fetchone()
                if row:
                    self._summary = row[0] or ""
                    try:
                        self._summary_covered_ids = set(eval(row[1]))  # noqa: S307
                    except Exception:
                        self._summary_covered_ids = set()

                # 恢复最近窗口（按时间倒序取 max_window 条）
                rows = conn.execute(
                    "SELECT item_id, content, metadata, source, created_at "
                    "FROM memory_items ORDER BY created_at DESC LIMIT ?",
                    (self.max_window,),
                ).fetchall()
                if rows:
                    self._buffer = [
                        MemoryItem(
                            item_id=r[0],
                            content=r[1],
                            metadata=eval(r[2]) if r[2] else {},  # noqa: S307
                            source=r[3],
                            created_at=datetime.fromisoformat(r[4]),
                        )
                        for r in reversed(rows)
                    ]
                    logger.debug(
                        f"短期记忆恢复: {len(self._buffer)} 条原文, "
                        f"摘要 {len(self._summary)} 字符"
                    )
        except Exception as e:
            logger.warning(f"短期记忆状态恢复失败: {e}")

    def _persist_summary(self) -> None:
        """持久化摘要到 SQLite。"""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO short_term_summary "
                    "(id, summary, covered_ids, updated_at) VALUES (1, ?, ?, ?)",
                    (
                        self._summary,
                        str(list(self._summary_covered_ids)),
                        datetime.now().isoformat(),
                    ),
                )
        except Exception as e:
            logger.warning(f"短期记忆摘要持久化失败: {e}")

    # ===== 写入 =====

    def add(self, item: MemoryItem) -> str:
        if not item.item_id:
            item.item_id = f"stm_{uuid.uuid4().hex[:12]}"
        item.source = self.layer
        item.created_at = datetime.now()
        with self._lock:
            self._buffer.append(item)
            # 窗口溢出触发摘要压缩
            overflow = len(self._buffer) - self.max_window
            if overflow >= self.summary_trigger:
                self._compress_old(overflow)
            elif len(self._buffer) > self.max_window:
                # 未达摘要阈值但已超窗口，先截断保留最新
                self._buffer = self._buffer[-self.max_window:]
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_items VALUES (?,?,?,?,?)",
                    (
                        item.item_id,
                        item.content,
                        str(item.metadata),
                        item.source,
                        item.created_at.isoformat(),
                    ),
                )
        except Exception as e:
            logger.warning(f"短期记忆持久化失败: {e}")
        logger.debug(f"短期记忆写入: {item.item_id}")
        return item.item_id

    def _compress_old(self, overflow: int) -> None:
        """将窗口溢出的旧轮次 LLM 摘要压缩进 _summary。

        Args:
            overflow: 超出 max_window 的条数，这些条目会被摘要吸收后从 buffer 移除
        """
        if overflow <= 0 or not self._buffer:
            return

        # 取最早的 overflow 条（已经在窗口外的）
        to_compress = self._buffer[:overflow]
        new_items_text = "\n".join(
            f"- [{item.created_at.strftime('%m-%d %H:%M')}] {item.content[:200]}"
            for item in to_compress
            if item.item_id not in self._summary_covered_ids
        )
        if not new_items_text:
            # 都已被覆盖过，直接截断
            self._buffer = self._buffer[overflow:]
            return

        # 调 LLM 压缩：旧摘要 + 新溢出条目 → 新摘要
        try:
            from ..models.llm import get_llm
            llm = get_llm()
            old_summary = f"已有摘要：\n{self._summary}\n" if self._summary else ""
            prompt = (
                f"{old_summary}以下是新增的对话历史片段，请合并进一份紧凑的累积摘要。"
                f"保留关键事实、决策、实体、工具调用结果，丢弃寒暄与冗余。"
                f"输出不超过 {self.summary_max_chars} 字符，直接输出摘要文本：\n\n"
                f"{new_items_text}"
            )
            new_summary = llm.chat(
                [
                    {"role": "system", "content": "你是会话摘要助手，输出紧凑的累积摘要。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            ).strip()
            if new_summary:
                self._summary = new_summary[: self.summary_max_chars]
                self._summary_covered_ids.update(item.item_id for item in to_compress)
                self._persist_summary()
                logger.info(
                    f"短期记忆摘要压缩: 吸收 {len(to_compress)} 条, "
                    f"摘要长度 {len(self._summary)} 字符"
                )
        except Exception as e:
            logger.warning(f"短期记忆摘要压缩失败（保留原文）: {e}")
            # 降级：不摘要，直接截断（原文会丢失但 SQLite 仍有全量）
            pass

        # 无论摘要是否成功，从内存窗口移除已处理的旧条目
        self._buffer = self._buffer[overflow:]

    # ===== 召回 =====

    def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """全量窗口召回：按时间倒序返回最近 top_k 条。

        短期记忆不做语义检索，仅按时间窗口返回。
        """
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        # 内存窗口 + SQLite 回填（修复优先级：内存不足时查 SQLite 补齐）
        items = [x for x in self._buffer if x.created_at >= cutoff]
        if len(items) < top_k:
            try:
                with self._get_conn() as conn:
                    rows = conn.execute(
                        "SELECT item_id, content, metadata, source, created_at "
                        "FROM memory_items WHERE created_at >= ? "
                        "ORDER BY created_at DESC LIMIT ?",
                        (cutoff.isoformat(), self.max_window),
                    ).fetchall()
                    sqlite_items = [
                        MemoryItem(
                            item_id=r[0],
                            content=r[1],
                            metadata=eval(r[2]) if r[2] else {},  # noqa: S307
                            source=r[3],
                            created_at=datetime.fromisoformat(r[4]),
                        )
                        for r in rows
                    ]
                    # 合并去重：内存已有 + SQLite 补齐
                    seen_ids = {x.item_id for x in items}
                    for item in sqlite_items:
                        if item.item_id not in seen_ids:
                            items.append(item)
                            seen_ids.add(item.item_id)
            except Exception as e:
                logger.warning(f"短期记忆查询失败: {e}")
        # 按时间倒序
        items.sort(key=lambda x: x.created_at, reverse=True)
        return items[:top_k]

    def search_with_summary(
        self, query: str, top_k: int = 5, max_tokens: int | None = None
    ) -> tuple[list[MemoryItem], str]:
        """滑动窗口 + 摘要召回（方案 B 核心接口）。

        Args:
            query: 查询文本（短期记忆不语义检索，仅用于接口兼容）
            top_k: 最近窗口最多返回的条数
            max_tokens: token 预算上限，按此动态裁剪原文条数。
                        None 表示不限制。摘要占用 summary_token_budget 预算。

        Returns:
            (recent_items, summary): 最近窗口原文 + 旧轮次累积摘要
        """
        recent = self.search(query, top_k=top_k)
        summary = self._summary

        # token 预算裁剪：摘要先扣减，剩余给原文
        if max_tokens is not None:
            summary_budget = min(self.summary_token_budget, max_tokens // 3)
            if _estimate_tokens(summary) > summary_budget:
                # 摘要超预算，硬截断（保留头部）
                summary = summary[: summary_budget * 3]  # 粗估 3 字符/token

            remaining = max_tokens - _estimate_tokens(summary)
            kept: list[MemoryItem] = []
            for item in recent:
                t = _estimate_tokens(item.content)
                if remaining - t < 0:
                    break
                kept.append(item)
                remaining -= t
            recent = kept

        return recent, summary

    def get(self, item_id: str) -> MemoryItem | None:
        for x in self._buffer:
            if x.item_id == item_id:
                return x
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT item_id, content, metadata, source, created_at "
                    "FROM memory_items WHERE item_id = ?",
                    (item_id,),
                ).fetchone()
            if row:
                return MemoryItem(
                    item_id=row[0],
                    content=row[1],
                    metadata=eval(row[2]) if row[2] else {},  # noqa: S307
                    source=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                )
        except Exception as e:
            logger.warning(f"短期记忆 get 失败: {e}")
        return None

    def cleanup_expired(self) -> int:
        """清理过期记忆，返回删除条数。"""
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
        try:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "DELETE FROM memory_items WHERE created_at < ?", (cutoff,)
                )
                return cur.rowcount
        except Exception as e:
            logger.warning(f"短期记忆清理失败: {e}")
            return 0

    # ===== 状态查询 =====

    @property
    def summary(self) -> str:
        """当前累积摘要（只读）。"""
        return self._summary

    def stats(self) -> dict:
        """返回短期记忆状态统计，便于调试与可观测性。"""
        return {
            "buffer_size": len(self._buffer),
            "max_window": self.max_window,
            "summary_length": len(self._summary),
            "summary_covered_count": len(self._summary_covered_ids),
            "summary_token_budget": self.summary_token_budget,
        }
