"""Base class for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gym_v.core import Observation


class Tool(ABC):
    """Base class for tools that can be used by agents.

    Tools provide additional capabilities to agents, such as code execution,
    web search, file operations, etc. Each tool defines an `execute` method
    that takes arguments and returns an Observation.

    Attributes:
        name: Tool name, used to match the "tool" field in action.
        description: Tool description, can be used to generate prompts for LLMs.
    """

    name: str
    description: str

    @abstractmethod
    def execute(self, **args) -> Observation:
        """Execute the tool with given arguments.

        Args:
            **args: Arguments parsed from the "args" field in action.

        Returns:
            Observation containing the execution result.
            - text: stdout, return value, or error message
            - image: any generated images (e.g., matplotlib figures)
        """
        raise NotImplementedError

    def reset(self) -> None:  # noqa: B027
        """Reset tool state. Called when env.reset() is invoked.

        Override this method to clear any persistent state,
        such as IPython kernel variables.
        """

    def close(self) -> None:  # noqa: B027
        """Clean up resources. Called when env.close() is invoked.

        Override this method to release any resources,
        such as stopping a kernel process.
        """
