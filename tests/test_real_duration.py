"""真实端到端耗时测试。

使用真实 DeepSeek API + 真实文献 API,测量端到端耗时。
不使用 Mock,数据可作为简历项目经历佐证。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 项目根目录(让 src 作为包可导入) - 从 cwd 解析
import os
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.orchestrator import get_orchestrator  # noqa: E402


def test_real_e2e_duration():
    """真实端到端耗时测试 - 单任务。"""
    print("\n" + "=" * 60)
    print("真实端到端耗时测试")
    print("=" * 60)

    orch = get_orchestrator()
    topic = "Transformer 架构核心原理"
    print(f"[主题] {topic}")
    print(f"[深度] basic")
    print("-" * 60)

    start = time.time()
    task = orch.run(topic, depth="basic")
    elapsed = time.time() - start

    print("-" * 60)
    print(f"[结果] 状态={task.status.value}")
    print(f"[结果] 文献数={len(task.papers)}")
    print(f"[结果] 知识点数={len(task.knowledge_points)}")
    print(f"[结果] 报告长度={len(task.report) if task.report else 0} 字")
    print(f"[结果] 质量评分={task.quality_score:.3f}")
    print(f"[耗时] 端到端 {elapsed:.1f}s ({elapsed/60:.2f}min)")
    print("=" * 60)

    # 写入测试结果到文件,便于后续读取
    result = {
        "topic": topic,
        "status": task.status.value,
        "papers": len(task.papers),
        "knowledge_points": len(task.knowledge_points),
        "report_length": len(task.report) if task.report else 0,
        "quality_score": round(task.quality_score, 3),
        "duration_sec": round(elapsed, 1),
    }

    # 输出为简单格式,便于解析
    print("\n[JSON]", result)

    assert task.status.value == "completed", f"任务未完成: {task.status.value}"
    assert elapsed > 5, "真实端到端耗时应大于 5s (排除 Mock 模式)"
    return result


if __name__ == "__main__":
    test_real_e2e_duration()
