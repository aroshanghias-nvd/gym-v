"""Interactive gym-v environment viewer for multiplayer turn-based games."""

from __future__ import annotations

import ast
import shutil
import sys
from textwrap import dedent
import tkinter as tk
from typing import Any

import click
from PIL import ImageTk
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

import gym_v
from gym_v.logger import get_logger

logger = get_logger()


def parse_env_args(env_args: tuple[str, ...]) -> dict[str, Any]:
    """Parse key=value arguments."""
    if not env_args:
        return {}

    kwargs = {}
    for arg in env_args:
        if "=" not in arg:
            continue
        key, raw_value = arg.split("=", 1)

        try:
            value = ast.literal_eval(raw_value.strip())
        except (ValueError, SyntaxError):
            value = raw_value.strip()

        kwargs[key.strip()] = value
    return kwargs


@click.command()
@click.option("--id", "env_id", default="TextArena/Othello-v0", show_default=True)
@click.option("--env-args", "env_args", multiple=True)
def main(env_id: str, env_args: tuple[str, ...]):
    root = tk.Tk()
    root.title(f"gym-v: {env_id}")
    root.attributes("-topmost", True)

    image_label = tk.Label(root)
    image_label.pack()

    session = PromptSession(history=InMemoryHistory())

    kwargs = parse_env_args(env_args)
    try:
        env = gym_v.make(env_id, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create environment({env_id}): {e}")
        sys.exit(1)

    is_game_over = False

    logger.info(f"Environment {env_id} created.")
    logger.info("Controls: Type command and Enter.")
    logger.info("  - 'reset' or 'r': Reset environment")
    logger.info("  - 'quit' or 'q': Exit")

    # Initial Reset
    obs_dict, info_dict = env.reset()

    # Check if obs_dict is empty (unexpected for initial state)
    if not obs_dict:
        logger.error("No observations returned after reset!")
        sys.exit(1)

    # Determine initial player (usually the one in obs_dict for turn-based games)
    current_agent_id = list(obs_dict.keys())[0]

    obs = obs_dict[current_agent_id]
    info = info_dict.get(current_agent_id, {})

    width = shutil.get_terminal_size(fallback=(80, 20)).columns

    # Print environment description
    desc = env.description
    if isinstance(desc, dict):
        # If description is a dict (like in Othello), print specific player description
        player_desc = desc.get(current_agent_id, str(desc))
        logger.info(f"\nEnv Description for {current_agent_id}:\n{player_desc}")
    else:
        logger.info(f"\nEnv Description: {desc}")

    logger.info(
        dedent(f"""
        {f" Step info(step_count={env.get_wrapper_attr('_current_episode_steps')}) ".center(width, "=")}
        Current Player: {current_agent_id}
        observation (text): {obs.text}
        info: {info}
        {"=" * width}""")
    )

    running = True
    while running:
        try:
            # Render image for the current player
            if obs.image:
                tk_image = ImageTk.PhotoImage(obs.image)
                image_label.configure(image=tk_image)
                image_label.image = tk_image
                root.update()
        except tk.TclError:
            break

        prompt = (
            "[Game Over] Type 'r' to reset >>> "
            if is_game_over
            else f"[{current_agent_id}] >>> "
        )

        try:
            action = session.prompt(prompt)
        except (EOFError, KeyboardInterrupt):
            break

        if not action:
            continue

        action_lower = action.lower()

        if action_lower in ("quit", "q"):
            running = False
            break

        elif action_lower in ("reset", "r"):
            obs_dict, info_dict = env.reset()
            current_agent_id = list(obs_dict.keys())[0]
            obs = obs_dict[current_agent_id]
            is_game_over = False
            logger.info(f"Environment({env_id}) Reset")

            # Re-print description if needed
            desc = env.description
            if isinstance(desc, dict):
                logger.info(
                    f"\nEnv Description for {current_agent_id}:\n{desc.get(current_agent_id, str(desc))}"
                )

            continue

        if is_game_over:
            logger.info("Game over. Type 'r' to reset.")
            continue

        # --- Step Execution ---
        action_dict = {current_agent_id: action}

        try:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = (
                env.step(action_dict)
            )
        except Exception as e:
            logger.error(f"Step failed: {e}")
            continue

        env_done = terminated_dict.get("__all__", False) or truncated_dict.get(
            "__all__", False
        )

        if env_done:
            is_game_over = True
            logger.info("Game over!")

            # Print final rewards and info
            for pid, r in reward_dict.items():
                reason = info_dict.get(pid, {}).get("reward_reason", "")
                logger.info(
                    f"Player {pid}: Reward={r} {f'({reason})' if reason else ''}"
                )

            # Keep showing last observation (image) but text might be irrelevant now
            # Usually we don't switch players on termination

        else:
            # --- Turn Switching ---
            # In turn-based games like Othello, env.step returns observation for the NEXT player.
            if not obs_dict:
                logger.error("No observations returned, cannot determine next player!")
                # This might happen if env returns empty dict on intermediate steps?
                # For now assume Othello behavior: always returns next player's obs.
                break

            next_agent_id = list(obs_dict.keys())[0]

            # Update state variables
            current_agent_id = next_agent_id
            obs = obs_dict[current_agent_id]
            info = info_dict.get(current_agent_id, {})
            reward = reward_dict.get(current_agent_id, 0.0)  # Reward from previous move

            logger.info(
                dedent(f"""
                {f" Step info(step_count={env.get_wrapper_attr('_current_episode_steps')}) ".center(width, "=")}
                Next Player: {current_agent_id}
                observation (text): {obs.text}
                reward (from last move): {reward}
                info: {info}
                {"=" * width}""")
            )

    try:
        root.destroy()
    except tk.TclError:
        pass
    env.close()


if __name__ == "__main__":
    main()
