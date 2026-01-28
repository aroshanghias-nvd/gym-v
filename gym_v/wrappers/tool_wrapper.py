"""Tool wrapper for adding tool capabilities to environments."""

from __future__ import annotations

import json
from typing import Any

from gym_v.core import Env, Observation, Wrapper
from gym_v.tools.base import Tool


class ToolWrapper(Wrapper):
    """Wrapper that adds tool calling capabilities to any environment.

    This wrapper intercepts actions and checks if they contain tool calls.
    If a tool call is detected, the tool is executed and its result is returned.
    Otherwise, the action is passed through to the underlying environment.

    Tool calls do not count towards the episode step limit.

    Action format for tool calls:
        {"tool": "tool_name", "args": {"arg1": "value1", ...}}

    Example:
        >>> from gym_v.tools import IPythonTool
        >>> env = ToolWrapper(base_env, tools=[IPythonTool()])
        >>> obs, info = env.reset()
        >>> # Call a tool
        >>> action = '{"tool": "ipython", "args": {"code": "print(1+1)"}}'
        >>> obs, reward, term, trunc, info = env.step({"agent_0": action})
        >>> # obs["agent_0"].text == "2"
    """

    def __init__(self, env: Env, tools: list[Tool]):
        """Initialize the tool wrapper.

        Args:
            env: The environment to wrap (must be single-agent).
            tools: List of Tool instances to make available.
        """
        super().__init__(env)
        assert env.num_players == 1, (
            f"ToolWrapper only supports single-agent environments, "
            f"got num_players={env.num_players}"
        )
        self._tools = {tool.name: tool for tool in tools}

    @property
    def tools(self) -> dict[str, Tool]:
        """Return the available tools."""
        return self._tools

    def step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Execute a step, either calling a tool or stepping the environment.

        Args:
            action: Dictionary with single agent's action string.
                    If action is a valid tool call JSON, the tool is executed.
                    Otherwise, the action is passed to the underlying env.

        Returns:
            Standard gym_v step return tuple.
        """
        agent_id = next(iter(self.env._agent_ids))
        act = action[agent_id]
        parsed = self._parse_tool_call(act)

        if parsed is not None:
            # Execute tool and return result
            tool_name, args = parsed
            obs = {agent_id: self._tools[tool_name].execute(**args)}
            reward = {agent_id: 0.0}
            terminated = {agent_id: False, "__all__": False}
            truncated = {agent_id: False, "__all__": False}
            info = {agent_id: {"tool_call": True}}
            return obs, reward, terminated, truncated, info

        # No tool call, pass through to env
        return self.env.step(action)

    def _parse_tool_call(self, action: str) -> tuple[str, dict] | None:
        """Parse an action string to check if it's a tool call.

        Args:
            action: Action string, potentially a JSON tool call.

        Returns:
            Tuple of (tool_name, args) if valid tool call, None otherwise.
        """
        try:
            data = json.loads(action)
            if isinstance(data, dict) and "tool" in data:
                tool_name = data["tool"]
                if tool_name in self._tools:
                    return tool_name, data.get("args", {})
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset the environment and all tools.

        Args:
            seed: Random seed for the environment.
            options: Additional reset options.

        Returns:
            Initial observation and info from the environment.
        """
        # Reset all tools
        for tool in self._tools.values():
            tool.reset()
        return self.env.reset(seed=seed, options=options)

    def close(self) -> None:
        """Close all tools and the environment."""
        for tool in self._tools.values():
            tool.close()
        return self.env.close()
