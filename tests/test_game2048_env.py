import math

import pytest

import gym_v

pytest.importorskip("textarena")


def test_game2048_reset_exposes_progress_metadata() -> None:
    env = gym_v.make("Games/Game2048-v0", target_tile=2048)

    obs, info = env.reset(seed=123)

    metadata = obs["agent_0"].metadata
    assert metadata["board"] == info["agent_0"]["board"]
    assert metadata["max_tile"] == info["agent_0"]["max_tile"]
    assert metadata["target_tile"] == 2048
    assert metadata["target_progress"] == pytest.approx(
        math.log2(metadata["max_tile"]) / math.log2(2048)
    )
    assert metadata["max_tile"] == max(max(row) for row in metadata["board"])


def test_game2048_step_exposes_progress_info_and_metadata() -> None:
    env = gym_v.make("Games/Game2048-v0", target_tile=2048)
    env.reset(seed=123)

    obs, _reward, _terminated, _truncated, info = env.step({"agent_0": "[Up]"})

    metadata = obs["agent_0"].metadata
    step_info = info["agent_0"]
    assert step_info["board"] == metadata["board"]
    assert step_info["max_tile"] == metadata["max_tile"]
    assert step_info["target_tile"] == 2048
    assert step_info["target_progress"] == metadata["target_progress"]
    assert step_info["max_tile"] == max(max(row) for row in step_info["board"])
    assert "invalid_action" in step_info
