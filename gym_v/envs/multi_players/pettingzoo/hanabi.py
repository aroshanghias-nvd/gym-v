"""Hanabi game using PettingZoo."""

from __future__ import annotations

from collections import defaultdict
from textwrap import dedent
from typing import Any

from PIL import Image
from typing_extensions import override

from gym_v import Env, Observation, get_logger
from gym_v.envs.multi_players.pettingzoo.utils import TerminateIllegalOutOfBoundsWrapper
from pettingzoo.classic import hanabi_v5

logger = get_logger()


class PettingZooHanabi(Env):
    """
    Hanabi cooperative card game using PettingZoo's hanabi environment.

    Players work together to play cards in the correct order without seeing their own hands.
    """

    is_deterministic = False

    def __init__(
        self,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_players < 2 or num_players > 5:
            raise ValueError(
                f"{self.__class__.__name__} supports 2-5 players, got {num_players}"
            )

        # Hanabi doesn't support rgb_array rendering, use None
        env = hanabi_v5.raw_env(players=num_players, render_mode=None)
        env = TerminateIllegalOutOfBoundsWrapper(env)
        self._pz_env = env

        self._agent_ids = {f"player_{i}" for i in range(num_players)}
        self._num_players = num_players

    @override
    @property
    def description(self) -> dict[str, str]:
        hand_size = 5 if self._num_players <= 3 else 4
        base_description = dedent("""
            You are playing <<Hanabi>> as Player {player_id}.

            This is a cooperative game. Work with other players to play cards 1-5 in order
            for each color. You can see other players' cards but not your own.

            Action format:
            - "play <pos>" - Play card at position (1-{hand_size})
            - "discard <pos>" - Discard card at position (1-{hand_size})
            - "hint <player> color <color>" - Hint a color to another player
            - "hint <player> rank <rank>" - Hint a rank to another player

            Colors: red, yellow, green, white, blue
            Ranks: 1, 2, 3, 4, 5
            Players: player_0, player_1, ...
        """).strip()
        return {
            f"player_{i}": base_description.format(
                player_id=str(i),
                hand_size=hand_size,
            )
            for i in range(self._num_players)
        }

    def _get_current_observation(self) -> Observation:
        """Get observation for the current player.

        Note: Hanabi doesn't support image rendering, so we provide text observation.
        """
        # Build text description of game state
        current_player = self._pz_env.agent_selection
        text = f"Current player: {current_player}\n"
        text += "(Hanabi is a text-based game - image rendering not available)"

        return Observation(image=None, text=text)

    def _get_pz_action(self, action: str) -> int:
        """Convert action string to PettingZoo action."""
        action = action.strip().lower()
        hand_size = 5 if self._num_players <= 3 else 4
        colors = ["red", "yellow", "green", "white", "blue"]
        ranks = ["1", "2", "3", "4", "5"]

        if action.startswith("play "):
            pos = int(action[5:]) - 1  # Convert 1-indexed to 0-indexed
            return pos
        elif action.startswith("discard "):
            pos = int(action[8:]) - 1
            return hand_size + pos
        elif action.startswith("hint "):
            parts = action[5:].split()
            target_player = int(parts[0].replace("player_", ""))
            hint_type = parts[1]
            hint_value = parts[2]

            # Calculate base offset for hints
            base = hand_size * 2

            # For multi-player, hints are organized by target player
            current_player_idx = int(
                self._pz_env.agent_selection.replace("player_", "")
            )

            # Adjust target index relative to current player
            relative_target = (target_player - current_player_idx - 1) % (
                self._num_players - 1
            )

            if hint_type == "color":
                color_idx = colors.index(hint_value)
                return base + relative_target * 5 + color_idx
            else:  # rank
                rank_idx = ranks.index(hint_value)
                # Rank hints come after all color hints
                num_other_players = self._num_players - 1
                return base + num_other_players * 5 + relative_target * 5 + rank_idx
        else:
            raise ValueError(f"Invalid action: {action}")

    @override
    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        obs_map: dict[str, Observation] = {}
        reward_map: dict[str, float] = defaultdict(float)
        terminated_map: dict[str, bool] = defaultdict(bool)
        truncated_map: dict[str, bool] = defaultdict(bool)
        info_map: dict[str, dict[str, Any]] = defaultdict(dict)
        info_map["__all__"] = {}

        acting_player = self._pz_env.agent_selection
        action_str = action[acting_player]
        pz_action = self._get_pz_action(action_str)

        self._pz_env.step(pz_action)

        while self._pz_env.agents:
            _, reward, terminated, truncated, info = self._pz_env.last()
            current_player = self._pz_env.agent_selection
            obs_map[current_player] = self._get_current_observation()
            reward_map[current_player] = reward
            terminated_map[current_player] = terminated
            truncated_map[current_player] = truncated
            info_map[current_player] = info

            if terminated or truncated:
                self._pz_env.step(None)
            else:
                break

        all_gone = not self._pz_env.agents
        terminated_map["__all__"] = all_gone and all(terminated_map.values())
        truncated_map["__all__"] = all_gone and all(truncated_map.values())

        # Add text feedback based on game state
        made_invalid_action = False
        made_invalid_action_players = []
        for pid in obs_map:
            if info_map.get(pid, {}).get("invalid_action", False):
                made_invalid_action = True
                made_invalid_action_players.append(pid)

        for pid in obs_map:
            if made_invalid_action:
                if pid in made_invalid_action_players:
                    obs_map[pid] = Observation(
                        image=obs_map[pid].image,
                        text="You made an invalid action.",
                        metadata=obs_map[pid].metadata,
                    )
                elif terminated_map.get(pid, False) or truncated_map.get(pid, False):
                    obs_map[pid] = Observation(
                        image=obs_map[pid].image,
                        text=f"Game terminated due to {', '.join(made_invalid_action_players)} made invalid actions.",
                        metadata=obs_map[pid].metadata,
                    )
            else:
                if terminated_map.get(pid, False) or truncated_map.get(pid, False):
                    final_score = reward_map[pid]
                    obs_map[pid] = Observation(
                        image=obs_map[pid].image,
                        text=f"Game over! Final score: {final_score}",
                        metadata=obs_map[pid].metadata,
                    )

        return obs_map, reward_map, terminated_map, truncated_map, info_map

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._pz_env.reset(seed=seed, options=options)
        _, _, _, _, info = self._pz_env.last()

        obs_map = {self._pz_env.agent_selection: self._get_current_observation()}
        info_map = {self._pz_env.agent_selection: info}
        info_map["__all__"] = {}

        logger.info("Reset PettingZoo Hanabi")

        return obs_map, info_map

    @override
    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the Hanabi game state.

        Note: Hanabi doesn't support rgb_array rendering, returns None.
        """
        return None

    @override
    def close(self):
        """Clean up resources."""
        self._pz_env.close()
