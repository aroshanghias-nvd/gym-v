# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import pytest

import gym_v
from gym_v import Env


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _snapshot(env_id: str, seed: int) -> tuple[str, Any, Any]:
    env = gym_v.make(env_id)
    try:
        observations, infos = env.reset(seed=seed)
        observation = observations["agent_0"]
        image = observation.image.convert("RGBA")
        return (
            hashlib.sha256(image.tobytes()).hexdigest(),
            _jsonable(observation.metadata),
            _jsonable(infos["agent_0"].get("oracle_answer")),
        )
    finally:
        env.close()


def test_python_rng_uses_reset_seed() -> None:
    env = Env()
    env.reset(seed=424242)
    first = [env.py_random.random() for _ in range(5)]

    random.seed(7)
    for _ in range(20):
        random.random()

    env.reset(seed=424242)
    assert [env.py_random.random() for _ in range(5)] == first


@pytest.mark.parametrize(
    "env_id",
    [
        "Cognition/Hue-QA-v0",
        "Geometry/Tangram-QA-v0",
        "Logic/Binairo-v0",
        "Puzzles/Freecell-QA-v0",
        "Puzzles/Jewel2-QA-v0",
    ],
)
def test_seeded_generation_is_independent_of_global_rng(
    env_id: str,
) -> None:
    random.seed(17)
    np.random.seed(19)
    first = _snapshot(env_id, 424242)

    random.seed(104729)
    np.random.seed(130363)
    for _ in range(100):
        random.random()
        np.random.random()

    assert _snapshot(env_id, 424242) == first


@pytest.mark.parametrize(
    "env_id",
    [
        "Algorithmic/Lifegame-QA-v0",
        "Cognition/FlowNetwork-v0",
        "Cognition/TreeToTraversal-v0",
        "Puzzles/Maze-QA-v0",
        "Puzzles/Pacman-QA-v0",
        "Puzzles/Snake-QA-v0",
        "Puzzles/SpaceInvaders-QA-v0",
        "Puzzles/Tetris-QA-v0",
    ],
)
def test_seeded_generation_does_not_mutate_global_python_rng(env_id: str) -> None:
    random.seed(271828)
    expected_next = random.Random(271828).random()

    _snapshot(env_id, 424242)

    assert random.random() == expected_next
