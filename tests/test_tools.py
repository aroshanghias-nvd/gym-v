"""Tests for tools and ToolWrapper."""

import json

import pytest

from gym_v import Env, Observation
from gym_v.tools import IPythonTool, Tool
from gym_v.wrappers.tool_wrapper import ToolWrapper


class DummyTool(Tool):
    """A simple tool for testing."""

    name = "dummy"
    description = "A dummy tool for testing"

    def execute(self, message: str = "hello") -> Observation:
        return Observation(text=f"dummy: {message}", image=None)


class DummyEnv(Env):
    """A minimal env for testing ToolWrapper."""

    _agent_ids = {"agent_0"}
    num_players = 1

    def __init__(self):
        super().__init__()
        self._step_count = 0

    @property
    def description(self) -> str:
        return "A dummy environment"

    def inner_step(self, action):
        self._step_count += 1
        obs = {
            "agent_0": Observation(text=f"env step: {action['agent_0']}", image=None)
        }
        reward = {"agent_0": 1.0}
        terminated = {"agent_0": False, "__all__": False}
        truncated = {"agent_0": False, "__all__": False}
        info = {"agent_0": {}}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._step_count = 0
        obs = {"agent_0": Observation(text="reset", image=None)}
        info = {"agent_0": {}}
        return obs, info

    def close(self):
        pass


class TestToolWrapper:
    """Tests for ToolWrapper."""

    def test_tool_call(self):
        """Test that tool calls are intercepted and executed."""
        env = ToolWrapper(DummyEnv(), tools=[DummyTool()])
        env.reset()

        action = {
            "agent_0": json.dumps({"tool": "dummy", "args": {"message": "world"}})
        }
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs["agent_0"].text == "dummy: world"
        assert reward["agent_0"] == 0.0
        assert terminated["agent_0"] is False
        assert info["agent_0"]["tool_call"] is True
        # Tool call should not increment env step count
        assert env.env._step_count == 0

    def test_non_tool_action_passes_through(self):
        """Test that non-tool actions are passed to the underlying env."""
        env = ToolWrapper(DummyEnv(), tools=[DummyTool()])
        env.reset()

        action = {"agent_0": "regular action"}
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs["agent_0"].text == "env step: regular action"
        assert reward["agent_0"] == 1.0
        assert env.env._step_count == 1

    def test_invalid_json_passes_through(self):
        """Test that invalid JSON is passed to the underlying env."""
        env = ToolWrapper(DummyEnv(), tools=[DummyTool()])
        env.reset()

        action = {"agent_0": "not json {"}
        obs, reward, terminated, truncated, info = env.step(action)

        assert "env step:" in obs["agent_0"].text
        assert env.env._step_count == 1

    def test_unknown_tool_passes_through(self):
        """Test that unknown tool names are passed to the underlying env."""
        env = ToolWrapper(DummyEnv(), tools=[DummyTool()])
        env.reset()

        action = {"agent_0": json.dumps({"tool": "unknown", "args": {}})}
        obs, reward, terminated, truncated, info = env.step(action)

        assert "env step:" in obs["agent_0"].text
        assert env.env._step_count == 1

    def test_reset_resets_tools(self):
        """Test that reset() calls reset on all tools."""

        class StatefulTool(Tool):
            name = "stateful"
            description = "A stateful tool"

            def __init__(self):
                self.state = 0

            def execute(self) -> Observation:
                self.state += 1
                return Observation(text=str(self.state), image=None)

            def reset(self):
                self.state = 0

        tool = StatefulTool()
        env = ToolWrapper(DummyEnv(), tools=[tool])
        env.reset()

        # Execute tool twice
        env.step({"agent_0": json.dumps({"tool": "stateful"})})
        env.step({"agent_0": json.dumps({"tool": "stateful"})})
        assert tool.state == 2

        # Reset should clear state
        env.reset()
        assert tool.state == 0


@pytest.mark.slow
class TestIPythonTool:
    """Tests for IPythonTool. Marked slow as they start a kernel."""

    def test_simple_execution(self):
        """Test simple code execution."""
        tool = IPythonTool()
        try:
            obs = tool.execute(code="print('hello')")
            assert "hello" in obs.text
        finally:
            tool.close()

    def test_expression_result(self):
        """Test that expression results are captured."""
        tool = IPythonTool()
        try:
            obs = tool.execute(code="1 + 1")
            assert "2" in obs.text
        finally:
            tool.close()

    def test_variable_persistence(self):
        """Test that variables persist across executions."""
        tool = IPythonTool()
        try:
            tool.execute(code="x = 42")
            obs = tool.execute(code="x * 2")
            assert "84" in obs.text
        finally:
            tool.close()

    def test_error_handling(self):
        """Test that errors are captured."""
        tool = IPythonTool()
        try:
            obs = tool.execute(code="1/0")
            assert "ZeroDivisionError" in obs.text
        finally:
            tool.close()

    def test_matplotlib_capture(self):
        """Test that matplotlib figures are captured."""
        tool = IPythonTool()
        try:
            obs = tool.execute(
                code="import matplotlib.pyplot as plt\nplt.plot([1,2,3])"
            )
            assert obs.image is not None
            assert len(obs.image) > 0
        finally:
            tool.close()

    def test_reset_clears_variables(self):
        """Test that reset clears the kernel state."""
        tool = IPythonTool()
        try:
            tool.execute(code="y = 100")
            tool.reset()
            obs = tool.execute(code="y")
            assert "NameError" in obs.text
        finally:
            tool.close()
