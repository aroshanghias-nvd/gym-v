"""FourRooms environment using Minigrid."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

import gymnasium as gym
import minigrid
from PIL import Image

from gym_v import Env, Observation, get_logger

logger = get_logger()


class MinigridFourRoomsEnv(Env):
    """FourRooms environment using Minigrid.

    Classic four rooms environment. The agent must navigate through four rooms
    connected by gaps in the walls to reach the goal.

    Args:
        tile_size: Size of each tile in pixels for rendering
    """

    def __init__(
        self,
        tile_size: int = 32,
        max_episode_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tile_size = tile_size

        if max_episode_steps is None:
            max_episode_steps = 100
        self._max_episode_steps = max_episode_steps

        self._agent_ids = {"agent_0"}

        self._minigrid_env = gym.make(
            "MiniGrid-FourRooms-v0",
            render_mode="rgb_array",
            max_steps=max_episode_steps,
            tile_size=tile_size,
        )

        self._action_map = {
            "left": 0,
            "right": 1,
            "forward": 2,
            "pickup": 3,
            "drop": 4,
            "toggle": 5,
            "done": 6,
        }

    @property
    def description(self) -> str:
        return dedent("""
            You are in a classic four rooms environment. The rooms are connected by gaps in the walls.
            Your goal is to navigate through the rooms to reach the green goal square.

            Available actions:
            - left: Turn left
            - right: Turn right
            - forward: Move forward
            - toggle: Toggle/activate an object
            - done: End the episode

            You need to find the passages between rooms and navigate efficiently to the goal.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._minigrid_env.reset(seed=seed)

        logger.info("Reset Minigrid FourRooms environment.")

        obs = Observation(image=self.render(), text=self._get_observation_text())
        info = {}
        return {"agent_0": obs}, {"agent_0": info}

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        agent_id = "agent_0"
        action_str = action[agent_id]

        if action_str not in self._action_map:
            raise ValueError(
                f"Invalid action '{action_str}'. Valid actions: {list(self._action_map.keys())}"
            )

        action_int = self._action_map[action_str]
        _, reward, terminated, truncated, info = self._minigrid_env.step(action_int)

        obs = Observation(image=self.render(), text=self._get_observation_text())

        return (
            {"agent_0": obs},
            {"agent_0": float(reward)},
            {"agent_0": terminated, "__all__": terminated},
            {"agent_0": truncated, "__all__": truncated},
            {"agent_0": info},
        )

    def render(self) -> Image.Image | None:
        img_array = self._minigrid_env.render()
        return Image.fromarray(img_array)

    def _get_observation_text(self) -> str:
        unwrapped_env = self._minigrid_env.unwrapped
        obs = unwrapped_env.gen_obs()
        mission = obs.get("mission", "")
        direction = obs.get("direction", 0)
        direction_names = ["right", "down", "left", "up"]
        direction_str = direction_names[direction] if 0 <= direction < 4 else "unknown"

        return f"{mission}\nYou are facing {direction_str}."

    def close(self):
        self._minigrid_env.close()
