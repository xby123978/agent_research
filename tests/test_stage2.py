"""阶段二：四 Agent 协作架构测试。"""

import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.common.data_models import ResearchDepth, ResearchTask, TaskStatus
from src.orchestrator.shared_board import SharedBoard
from src.orchestrator.state_machine import StateMachine


class TestStateMachine:
    """状态机转移合法性测试。"""

    def test_valid_transition(self):
        assert StateMachine.can_transition(TaskStatus.PENDING_PLANNING, TaskStatus.PENDING_SEARCH)
        assert StateMachine.can_transition(TaskStatus.PENDING_SEARCH, TaskStatus.PENDING_FILTER)
        assert StateMachine.can_transition(TaskStatus.PENDING_OUTPUT, TaskStatus.COMPLETED)

    def test_invalid_transition(self):
        # 不能从规划直接跳到输出
        assert not StateMachine.can_transition(TaskStatus.PENDING_PLANNING, TaskStatus.PENDING_OUTPUT)
        # 不能从完成回退
        assert not StateMachine.can_transition(TaskStatus.COMPLETED, TaskStatus.PENDING_PLANNING)

    def test_next_state_raises_on_invalid(self):
        import pytest
        with pytest.raises(ValueError):
            StateMachine.next_state(TaskStatus.COMPLETED, TaskStatus.PENDING_PLANNING)

    def test_terminal_state(self):
        assert StateMachine.is_terminal(TaskStatus.COMPLETED)
        assert StateMachine.is_terminal(TaskStatus.FAILED)
        assert not StateMachine.is_terminal(TaskStatus.PENDING_PLANNING)

    def test_agent_for_state(self):
        assert StateMachine.agent_for_state(TaskStatus.PENDING_PLANNING) == "PlannerAgent"
        assert StateMachine.agent_for_state(TaskStatus.PENDING_SEARCH) == "CollectorAgent"
        assert StateMachine.agent_for_state(TaskStatus.PENDING_OUTPUT) == "WriterAgent"


class TestSharedBoard:
    """共享黑板测试。"""

    def test_read_write(self):
        task = ResearchTask(task_id="t1", topic="测试")
        board = SharedBoard(task)
        assert board.read("plan") is None
        board.write("plan", {"subtasks": ["a", "b"]})
        assert board.read("plan") == {"subtasks": ["a", "b"]}

    def test_append_extend(self):
        task = ResearchTask(task_id="t1", topic="测试")
        board = SharedBoard(task)
        board.append("papers", "p1")
        board.extend("papers", ["p2", "p3"])
        assert board.read("papers") == ["p1", "p2", "p3"]

    def test_state_update(self):
        task = ResearchTask(task_id="t1", topic="测试")
        board = SharedBoard(task)
        board.update_task_state(TaskStatus.PENDING_SEARCH)
        assert board.get_task_state() == TaskStatus.PENDING_SEARCH

    def test_sync_to_task(self):
        task = ResearchTask(task_id="t1", topic="测试")
        board = SharedBoard(task)
        board.write("final_report", "测试报告内容")
        result = board.sync_to_task()
        assert result.report == "测试报告内容"

    def test_log_stage(self):
        task = ResearchTask(task_id="t1", topic="测试")
        board = SharedBoard(task)
        board.log_stage("PlannerAgent", "规划完成")
        logs = board.read("stage_logs")
        assert len(logs) == 1
        assert logs[0]["stage"] == "PlannerAgent"
        assert logs[0]["message"] == "规划完成"


class TestPreferenceMemory:
    """偏好记忆测试。"""

    def test_default_prefs_loaded(self, tmp_path):
        from src.memory.preference import PreferenceMemory, DEFAULT_PREFS

        pm = PreferenceMemory(prefs_path=str(tmp_path / "prefs.json"))
        assert pm.get_pref("research_style") == DEFAULT_PREFS["research_style"]
        assert pm.get_pref("max_papers_per_topic") == 20
        assert pm.get_pref("citation_style") == "apa"

    def test_set_pref_persists(self, tmp_path):
        from src.memory.preference import PreferenceMemory
        prefs_file = tmp_path / "prefs.json"
        pm = PreferenceMemory(prefs_path=str(prefs_file))
        pm.set_pref("research_style", "detailed")
        # 重新加载，验证持久化
        pm2 = PreferenceMemory(prefs_path=str(prefs_file))
        assert pm2.get_pref("research_style") == "detailed"

    def test_search_prefs(self, tmp_path):
        from src.memory.preference import PreferenceMemory
        pm = PreferenceMemory(prefs_path=str(tmp_path / "prefs.json"))
        results = pm.search("apa")
        assert len(results) >= 1
