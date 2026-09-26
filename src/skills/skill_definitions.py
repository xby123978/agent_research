"""Skill YAML 定义加载器。

对应架构 3.5.1 Skill 标准定义：每个 Skill 应含
name/version/description/input_schema/output_schema/flow/quality_gates。

现有 Skill 子类用 Python 方法实现 flow 逻辑，本模块把
flow 元数据与 quality_gates 外置到 YAML，便于声明式配置。
运行时仍由 Python 类执行，YAML 仅作元数据源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..common.logger import get_logger

logger = get_logger(__name__)

# Skill 定义目录
SKILLS_DIR = Path(__file__).resolve().parent / "definitions"


def load_skill_definition(name: str) -> dict[str, Any]:
    """加载指定 Skill 的 YAML 定义。

    Args:
        name: skill 名称（如 literature_review）
    Returns:
        含 name/version/flow/quality_gates 的字典；加载失败返回空 dict
    """
    path = SKILLS_DIR / f"{name}.yaml"
    if not path.exists():
        logger.debug(f"Skill 定义不存在: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("skill", data)


def list_skill_definitions() -> list[dict[str, Any]]:
    """列出所有已定义 Skill。"""
    if not SKILLS_DIR.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in SKILLS_DIR.glob("*.yaml"):
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        skill = data.get("skill", data)
        if skill.get("name"):
            out.append(skill)
    return out
