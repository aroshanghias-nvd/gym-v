"""Chess game using PettingZoo."""

from __future__ import annotations

from collections import defaultdict
from textwrap import dedent
from typing import Any

import chess
from PIL import Image
from typing_extensions import override

from gym_v import Env, Observation, get_logger
from gym_v.envs.multi_players.pettingzoo.utils import TerminateIllegalOutOfBoundsWrapper
from pettingzoo.classic import chess_v6

logger = get_logger()

# Maps for converting between UCI moves and PettingZoo action indices
MOVE_TYPES = {
    # Queen-type moves: direction * 7 distances
    "N": list(range(7)),  # North
    "NE": list(range(7, 14)),  # Northeast
    "E": list(range(14, 21)),  # East
    "SE": list(range(21, 28)),  # Southeast
    "S": list(range(28, 35)),  # South
    "SW": list(range(35, 42)),  # Southwest
    "W": list(range(42, 49)),  # West
    "NW": list(range(49, 56)),  # Northwest
    # Knight moves
    "KNIGHT": list(range(56, 64)),
    # Underpromotions (to rook, bishop, knight)
    "UNDERPROMO": list(range(64, 73)),
}

KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]


class PettingZooChess(Env):
    """
    Chess game using PettingZoo's chess environment.

    Two players take turns moving pieces according to standard chess rules.
    """

    is_deterministic = False

    def __init__(
        self,
        num_players: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_players != 2:
            raise ValueError(
                f"{self.__class__.__name__} only supports 2 players, got {num_players}"
            )

        env = chess_v6.raw_env(render_mode="rgb_array")
        env = TerminateIllegalOutOfBoundsWrapper(env)
        self._pz_env = env

        self._agent_ids = {"player_0", "player_1"}

    @override
    @property
    def description(self) -> dict[str, str]:
        base_description = dedent("""
            You are playing <<Chess>> as Player {player_id} with {color} pieces.

            Standard chess rules apply. Move your pieces strategically to checkmate
            your opponent's king.

            Action format: Provide a move in UCI notation (e.g., "e2e4", "g1f3").
            For pawn promotion, append the piece letter (e.g., "e7e8q" for queen).
        """).strip()
        return {
            "player_0": base_description.format(player_id="0", color="white"),
            "player_1": base_description.format(player_id="1", color="black"),
        }

    def _get_current_observation(self) -> Observation:
        """Get observation for the current player."""
        return Observation(image=self.render(), text=None)

    def _get_pz_action(self, action: str) -> int:
        """Convert UCI notation to PettingZoo action index."""
        action = action.strip().lower()
        move = chess.Move.from_uci(action)

        from_square = move.from_square
        to_square = move.to_square

        from_row, from_col = from_square // 8, from_square % 8
        to_row, to_col = to_square // 8, to_square % 8

        delta_row = to_row - from_row
        delta_col = to_col - from_col

        # Check for knight move
        if (abs(delta_row), abs(delta_col)) in [(2, 1), (1, 2)]:
            for i, (dr, dc) in enumerate(KNIGHT_MOVES):
                if delta_row == dr and delta_col == dc:
                    move_type = 56 + i
                    break
        # Check for underpromotion
        elif move.promotion and move.promotion != chess.QUEEN:
            promo_piece = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}[
                move.promotion
            ]
            if delta_col == -1:
                direction = 0  # left capture
            elif delta_col == 0:
                direction = 1  # straight
            else:
                direction = 2  # right capture
            move_type = 64 + promo_piece * 3 + direction
        # Queen-type move (including queen promotion)
        else:
            distance = max(abs(delta_row), abs(delta_col)) - 1
            if delta_row > 0 and delta_col == 0:
                direction = 0  # N
            elif delta_row > 0 and delta_col > 0:
                direction = 1  # NE
            elif delta_row == 0 and delta_col > 0:
                direction = 2  # E
            elif delta_row < 0 and delta_col > 0:
                direction = 3  # SE
            elif delta_row < 0 and delta_col == 0:
                direction = 4  # S
            elif delta_row < 0 and delta_col < 0:
                direction = 5  # SW
            elif delta_row == 0 and delta_col < 0:
                direction = 6  # W
            else:
                direction = 7  # NW
            move_type = direction * 7 + distance

        return from_row * 8 * 73 + from_col * 73 + move_type

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
                    if reward_map[pid] > 0:
                        obs_map[pid] = Observation(
                            image=obs_map[pid].image,
                            text="You win!",
                            metadata=obs_map[pid].metadata,
                        )
                    elif reward_map[pid] < 0:
                        obs_map[pid] = Observation(
                            image=obs_map[pid].image,
                            text="You lose!",
                            metadata=obs_map[pid].metadata,
                        )
                    else:
                        obs_map[pid] = Observation(
                            image=obs_map[pid].image,
                            text="Draw!",
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

        logger.info("Reset PettingZoo Chess")

        return obs_map, info_map

    @override
    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the Chess board."""
        return Image.fromarray(self._pz_env.render())

    @override
    def close(self):
        """Clean up resources."""
        self._pz_env.close()
