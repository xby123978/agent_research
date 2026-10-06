"""配置加载器。

统一从 .env、settings.yaml、models.yaml、mcp_servers.yaml 加载配置，
对应架构第 3 节环境约定：所有阈值、密钥、路径、开关全部从配置文件读取。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .exceptions import ConfigError

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """加载全部配置，合并 .env 与 yaml。

    返回一个聚合字典：
      - settings: settings.yaml 内容
      - models: models.yaml 内容
      - mcp_servers: mcp_servers.yaml 内容
      - env: 环境变量子集（密钥、URL 等）
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    settings = _load_yaml(CONFIG_DIR / "settings.yaml")
    models_cfg = _load_yaml(CONFIG_DIR / "models.yaml")
    mcp_cfg = _load_yaml(CONFIG_DIR / "mcp_servers.yaml")

    env = {
        "llm_api_key": os.getenv("LLM_API_KEY", ""),
        "llm_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "llm_model": os.getenv("LLM_MODEL", "deepseek-chat"),
        "jev_base_url": os.getenv("JEV_BASE_URL", "http://localhost:11434/v1"),
        "jev_model": os.getenv("JEV_MODEL", "qwen2.5:7b"),
        "jev_api_key": os.getenv("JEV_API_KEY", "ollama"),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "ollama"),
        "embedding_base_url": os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "bge-m3"),
        "embedding_api_key": os.getenv("EMBEDDING_API_KEY", "ollama"),
        "embedding_fallback_model": os.getenv("EMBEDDING_FALLBACK_MODEL", "all-MiniLM-L6-v2"),
        "qdrant_url": os.getenv("QDRANT_URL", ""),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
        "qdrant_path": os.getenv("QDRANT_PATH", ""),  # 空则默认 DATA_DIR/qdrant
        "s2_api_key": os.getenv("S2_API_KEY", ""),
        "data_dir": os.getenv("DATA_DIR", "./data"),
        "log_dir": os.getenv("LOG_DIR", "./logs"),
        "output_dir": os.getenv("OUTPUT_DIR", "./outputs"),
        "default_depth": os.getenv("DEFAULT_DEPTH", "standard"),
        "default_max_papers": int(os.getenv("DEFAULT_MAX_PAPERS", "20")),
    }

    return {
        "settings": settings,
        "models": models_cfg,
        "mcp_servers": mcp_cfg,
        "env": env,
    }


def get_settings() -> dict[str, Any]:
    """获取 settings.yaml 内容。"""
    return load_config()["settings"]


def get_env() -> dict[str, Any]:
    """获取环境变量配置。"""
    return load_config()["env"]


def ensure_dirs() -> None:
    """确保数据、日志、输出目录存在。"""
    env = get_env()
    for key in ("data_dir", "log_dir", "output_dir"):
        path = Path(env[key])
        path.mkdir(parents=True, exist_ok=True)
