"""JEV 决策引擎（接入 OpenJev 真实判别模型）。

对应架构 3.6 JEV 决策模块：
- 只做判断/评分/分类任务，绝不承担生成类工作
- 三类任务：scoring、classification、judgment

分级执行策略（架构 3.6.1）：
- confidence ≥ 0.8：自动执行
- 0.5 ≤ confidence < 0.8：提交大模型复核
- confidence < 0.5：丢弃结果

模式优先级（架构要求独立判别模型，非 LLM 兼任）：
1. OpenJev 本地模式（主）：Qwen3.5-0.8B NLI 交叉编码器，0 token 生成
2. DeepSeek API（降级）：本地不可用时回退
3. 规则评分（终极兜底）
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from ..common.config import get_env
from ..common.logger import get_logger
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logger = get_logger(__name__)

# 本地模型路径优先(避免 Windows symlink 问题)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OPENJEV = _PROJECT_ROOT / "models" / "openjev" / "qwen3.5-0.8b-nli-v2s-long"

TaskType = Literal["scoring", "classification", "judgment"]


@dataclass
class JEVResult:
    """JEV 结果（架构 3.6.1）：含 result/confidence/reason 三字段。"""

    result: Any
    confidence: float
    reason: str
    task_type: TaskType
    escalated: bool = False

    @property
    def auto_execute(self) -> bool:
        return self.confidence >= 0.8

    @property
    def needs_review(self) -> bool:
        return 0.5 <= self.confidence < 0.8

    @property
    def discarded(self) -> bool:
        return self.confidence < 0.5


class _RuleBasedFallback:
    """规则兜底评分。"""

    @staticmethod
    def score(content: str, rubric: str = "") -> JEVResult:
        if not content or len(content) < 10:
            return JEVResult(0.1, 0.4, "内容过短", "scoring")
        base = 0.3
        length_bonus = min(0.4, len(content) / 200 * 0.05)
        keywords = ["方法", "原理", "结论", "应用", "案例", "对比", "局限", "未来"]
        kw_bonus = min(0.2, sum(0.025 for k in keywords if k in content))
        v = min(0.8, base + length_bonus + kw_bonus)
        return JEVResult(v, 0.6, "规则评分", "scoring")

    @staticmethod
    def classify(content: str, categories: list[str]) -> JEVResult:
        if not categories:
            return JEVResult("unknown", 0.4, "无候选类别", "classification")
        scores = {c: 0 for c in categories}
        for c in categories:
            if c.lower() in content.lower():
                scores[c] += 1
        best = max(scores.items(), key=lambda x: x[1])
        if best[1] > 0:
            return JEVResult(best[0], 0.6, "规则分类", "classification")
        return JEVResult(categories[0], 0.4, "无匹配取首类", "classification")

    @staticmethod
    def judge(content: str, criteria: str = "") -> JEVResult:
        passed = bool(content and len(content) > 50)
        conf = 0.6 if passed else 0.4
        return JEVResult(passed, conf, "规则判断", "judgment")


class _OpenJevBackend:
    """OpenJev 真实判别模型后端（Qwen3.5-0.8B NLI 交叉编码器）。

    单次 forward pass 输出蕴含/矛盾/中立三类概率，0 token 生成。
    与主推理 DeepSeek 完全独立，符合架构"独立审查官"意图。
    """

    def __init__(self) -> None:
        self._loaded = False
        self._tok = None
        self._model = None
        self._label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}

    def _ensure_loaded(self) -> bool:
        """惰性加载模型。优先本地路径,fallback HF Hub。"""
        if self._loaded:
            return True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("加载 OpenJev 0.8B 判别模型...")

            # 优先策略 1:本地路径(避免 Windows symlink + 跳过网络)
            model_path = str(_LOCAL_OPENJEV) if _LOCAL_OPENJEV.exists() else None

            if model_path:
                logger.info(f"使用本地模型路径: {model_path}")
                self._tok = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, local_files_only=True
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, trust_remote_code=True, local_files_only=True
                )
            else:
                # 策略 2:从 HF Hub 加载(网络下载到 cache)
                logger.info("本地路径不存在,从 HF Hub 加载...")
                self._tok = AutoTokenizer.from_pretrained(
                    "AlexWortega/openjev",
                    subfolder="qwen3.5-0.8b-nli-v2s-long",
                    trust_remote_code=True,
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    "AlexWortega/openjev",
                    subfolder="qwen3.5-0.8b-nli-v2s-long",
                    trust_remote_code=True,
                )

            self._model.eval()
            if hasattr(self._model.config, "id2label"):
                self._label_map = {
                    int(k): v.lower()
                    for k, v in self._model.config.id2label.items()
                }
            self._loaded = True
            logger.info(f"OpenJev 加载成功，labels={self._label_map}")
            return True
        except Exception as e:
            logger.warning(f"OpenJev 加载失败: {e}")
            # 缺少 sentencepiece/tiktoken 时给出明确诊断
            err_msg = str(e).lower()
            if "sentencepiece" in err_msg or "tiktoken" in err_msg:
                logger.error(
                    "缺少 tokenizer 依赖,请执行: pip install sentencepiece tiktoken"
                )
            return False

    def _nli_probs(self, premise: str, hypothesis: str) -> dict[str, float]:
        """单次 NLI 推理：返回 {label: prob}。"""
        if not self._ensure_loaded():
            return {"entailment": 0.5, "contradiction": 0.0, "neutral": 0.5}
        import torch

        inputs = self._tok(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        return {self._label_map.get(i, "unknown"): float(p) for i, p in enumerate(probs)}

    def score(self, content: str, rubric: str) -> JEVResult:
        """评分：rubric 作 premise，content 作 hypothesis，entailment 概率即分数。"""
        probs = self._nli_probs(rubric or "内容质量", content)
        ent = probs.get("entailment", 0.5)
        contra = probs.get("contradiction", 0.0)
        s = max(0.0, min(1.0, ent - contra * 0.5))
        conf = min(1.0, abs(ent - probs.get("neutral", 0.5)) + 0.3)
        return JEVResult(s, conf, f"OpenJev NLI ent={ent:.2f}", "scoring")

    def classify(self, content: str, categories: list[str]) -> JEVResult:
        """分类：每类别构造 hypothesis，选 entailment 最高者。"""
        if not categories:
            return JEVResult("unknown", 0.4, "无类别", "classification")
        best_cat = categories[0]
        best_ent = -1.0
        for cat in categories:
            hyp = f"该内容属于 {cat} 类别"
            probs = self._nli_probs(content, hyp)
            ent = probs.get("entailment", 0.0)
            if ent > best_ent:
                best_ent = ent
                best_cat = cat
        conf = min(1.0, best_ent + 0.2)
        return JEVResult(best_cat, conf, f"OpenJev ent={best_ent:.2f}", "classification")

    def judge(self, content: str, criteria: str) -> JEVResult:
        """判断：criteria 作 premise，content 作 hypothesis，entailment > 0.5 即通过。"""
        probs = self._nli_probs(criteria or "内容满足要求", content)
        ent = probs.get("entailment", 0.0)
        contra = probs.get("contradiction", 0.0)
        passed = ent > 0.5 and ent > contra
        conf = abs(ent - contra)
        return JEVResult(passed, conf, f"OpenJev ent={ent:.2f}", "judgment")


class JEVEngine:
    """JEV 决策引擎（OpenJev 主导 + DeepSeek 降级 + 规则兜底）。"""

    def __init__(self) -> None:
        env = get_env()
        self.api_key = env["llm_api_key"]
        self.api_base = env["llm_base_url"].rstrip("/")
        self.api_model = env["llm_model"]
        self.timeout = 15.0
        self._fallback = _RuleBasedFallback()
        self._openjev = _OpenJevBackend()
        self._init_mode()

    def _init_mode(self) -> None:
        """确定主模式：OpenJev > DeepSeek API > 规则。"""
        self.mode = "rule"
        try:
            if self._openjev._ensure_loaded():
                self.mode = "openjev"
                logger.info("JEV 使用 OpenJev 本地判别模型（架构主推，独立于 DeepSeek）")
                return
        except Exception as e:
            logger.warning(f"OpenJev 不可用: {e}")
        if self.api_key:
            self.mode = "api"
            logger.warning("JEV 降级到 DeepSeek API（非独立判别，架构降级方案）")
            return
        logger.warning("JEV 降级到规则评分")

    def score(self, content: str, rubric: str, max_score: float = 1.0) -> float:
        """评分任务（兼容旧接口）：返回纯数值。"""
        r = self.score_with_confidence(content, rubric, max_score)
        return r.result

    def score_with_confidence(
        self, content: str, rubric: str, max_score: float = 1.0
    ) -> JEVResult:
        """评分任务：返回含 confidence 的 JEVResult。"""
        if self.mode == "openjev":
            r = self._openjev.score(content, rubric)
            r.result = r.result * max_score
            return self._apply_grading(r)
        prompt = self._build_prompt("scoring", content, rubric=rubric, max_score=max_score)
        raw = self._invoke_api(prompt)
        try:
            data = self._parse_json(raw)
            v = float(data.get("score", data.get("result", 0)))
            v = max(0.0, min(max_score, v))
            conf = float(data.get("confidence", 0.8))
            reason = str(data.get("reason", ""))
            return self._apply_grading(JEVResult(v, conf, reason, "scoring"))
        except Exception as e:
            logger.warning(f"JEV score 解析失败，规则兜底: {e}")
            r = self._fallback.score(content, rubric)
            r.result = r.result * max_score
            return r

    def classify(self, content: str, categories: list[str]) -> str:
        """分类任务（兼容旧接口）。"""
        r = self.classify_with_confidence(content, categories)
        return r.result

    def classify_with_confidence(
        self, content: str, categories: list[str]
    ) -> JEVResult:
        """分类任务：返回 JEVResult。"""
        if self.mode == "openjev":
            return self._apply_grading(self._openjev.classify(content, categories))
        prompt = self._build_prompt("classification", content, categories=categories)
        raw = self._invoke_api(prompt)
        try:
            data = self._parse_json(raw)
            cat = str(data.get("category", data.get("result", "")))
            conf = float(data.get("confidence", 0.8))
            reason = str(data.get("reason", ""))
            if cat in categories:
                return self._apply_grading(JEVResult(cat, conf, reason, "classification"))
            for c in categories:
                if c.lower() in cat.lower():
                    return self._apply_grading(JEVResult(c, conf, reason, "classification"))
            return self._fallback.classify(content, categories)
        except Exception as e:
            logger.warning(f"JEV classify 解析失败，规则兜底: {e}")
            return self._fallback.classify(content, categories)

    def judge(self, content: str, criteria: str) -> bool:
        """判断任务（兼容旧接口）。"""
        r = self.judge_with_confidence(content, criteria)
        return r.result

    def judge_with_confidence(self, content: str, criteria: str) -> JEVResult:
        """判断任务：返回 JEVResult。"""
        if self.mode == "openjev":
            return self._apply_grading(self._openjev.judge(content, criteria))
        prompt = self._build_prompt("judgment", content, criteria=criteria)
        raw = self._invoke_api(prompt)
        try:
            data = self._parse_json(raw)
            passed = bool(data.get("pass", data.get("result", data.get("value", False))))
            conf = float(data.get("confidence", 0.8))
            reason = str(data.get("reason", ""))
            return self._apply_grading(JEVResult(passed, conf, reason, "judgment"))
        except Exception as e:
            logger.warning(f"JEV judge 解析失败，规则兜底: {e}")
            return self._fallback.judge(content, criteria)

    # ===== 架构 3.6.2 六类落地场景 =====

    def select_tool(self, query: str, tools: list[str]) -> str:
        """场景 1：检索工具路由选择。"""
        r = self.classify_with_confidence(query, tools)
        if r.auto_execute:
            logger.info(f"JEV 工具路由自动选择: {r.result} (conf={r.confidence:.2f})")
            return r.result
        return tools[0] if tools else r.result

    def should_deepen_subtopic(self, topic: str, current_coverage: str) -> bool:
        """场景 5：是否需要深入子主题。"""
        r = self.judge_with_confidence(
            current_coverage,
            f"判断当前内容对「{topic}」的覆盖是否充分",
        )
        if r.auto_execute:
            return not r.result
        return False

    def is_recall_sufficient(self, query: str, recall_results: list) -> bool:
        """场景 6：记忆召回结果是否充分。"""
        if not recall_results:
            return False
        content = " ".join(
            getattr(item, "content", str(item))[:100] for item in recall_results[:5]
        )
        r = self.judge_with_confidence(content, f"判断召回结果对查询「{query}」是否充分")
        if r.auto_execute:
            return r.result
        return False

    def _apply_grading(self, r: JEVResult) -> JEVResult:
        """应用分级执行策略（架构 3.6.1）。"""
        if r.discarded:
            logger.info(f"JEV confidence={r.confidence:.2f} < 0.5，丢弃，需大模型重新决策")
        elif r.needs_review:
            r.escalated = True
            logger.info(f"JEV confidence={r.confidence:.2f} 在 [0.5, 0.8)，提交大模型复核")
        elif r.auto_execute:
            logger.debug(f"JEV confidence={r.confidence:.2f} ≥ 0.8，自动执行")
        return r

    # ===== DeepSeek API 降级实现 =====

    def _build_prompt(
        self,
        task_type: TaskType,
        content: str,
        rubric: str = "",
        criteria: str = "",
        categories: list[str] | None = None,
        max_score: float = 1.0,
    ) -> list[dict[str, str]]:
        if task_type == "scoring":
            sys_msg = (
                "你是质量评分官（JEV 降级模式）。只输出 JSON："
                '{"score": float, "confidence": float[0,1], "reason": str}'
                f" score 范围 [0,{max_score}]。"
            )
            user_msg = f"评分标准：{rubric}\n\n被评分内容：\n{content[:2000]}"
        elif task_type == "classification":
            cats = ", ".join(categories or [])
            sys_msg = (
                "你是分类器（JEV 降级模式）。"
                f"可选类别：[{cats}]。只输出 JSON："
                '{"category": str, "confidence": float, "reason": str}'
            )
            user_msg = f"待分类内容：\n{content[:2000]}"
        else:
            sys_msg = (
                "你是审查官（JEV 降级模式）。只输出 JSON："
                '{"pass": bool, "confidence": float, "reason": str}'
            )
            user_msg = f"判断标准：{criteria}\n\n待判断内容：\n{content[:2000]}"
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

    def _invoke_api(self, messages: list[dict[str, str]]) -> str:
        if self.mode not in ("api", "rule"):
            return ""
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.api_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                return ""
            return resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            logger.warning(f"JEV API 调用失败：{e}，规则兜底")
            return ""

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"\{[^{}]*\}", raw)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"无法从输出提取 JSON: {raw[:100]}")


_jev_instance: JEVEngine | None = None


def get_jev() -> JEVEngine:
    """获取全局 JEV 决策引擎单例。"""
    global _jev_instance
    if _jev_instance is None:
        _jev_instance = JEVEngine()
    return _jev_instance
