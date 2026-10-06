"""共享 pytest fixtures。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 将 src 加入路径
SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 确保使用纯内存 Qdrant + 无 API Key 的测试环境（不落盘、不污染 data_dir）
os.environ.setdefault("QDRANT_URL", "")
os.environ.setdefault("QDRANT_PATH", ":memory:")
os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("EMBEDDING_PROVIDER", "sentence_transformers")
os.environ.setdefault("DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """临时数据目录。"""
    return tmp_path
