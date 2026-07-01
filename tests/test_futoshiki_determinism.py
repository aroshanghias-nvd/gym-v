# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import hashlib
import random

import numpy as np

import gym_v


def _snapshot(seed: int) -> tuple[str, tuple[int, int], str, str]:
    env = gym_v.make("Logic/Futoshiki-v0", size=4)
    try:
        observations, infos = env.reset(seed=seed)
        image = observations["agent_0"].image.convert("RGBA")
        return (
            image.mode,
            image.size,
            hashlib.sha256(image.tobytes()).hexdigest(),
            infos["agent_0"]["oracle_answer"],
        )
    finally:
        env.close()


def test_futoshiki_seed_is_independent_of_global_random_state() -> None:
    random.seed(17)
    np.random.seed(19)
    first = _snapshot(300001)

    random.seed(104729)
    np.random.seed(130363)
    random.shuffle(list(range(100)))
    np.random.shuffle(np.arange(100))
    second = _snapshot(300001)

    assert second == first


def test_futoshiki_reset_is_order_independent() -> None:
    first = _snapshot(300001)
    _snapshot(300002)
    assert _snapshot(300001) == first


def test_futoshiki_oracle_and_invalid_answer_rewards() -> None:
    oracle_env = gym_v.make("Logic/Futoshiki-v0", size=4)
    invalid_env = gym_v.make("Logic/Futoshiki-v0", size=4)
    try:
        _, infos = oracle_env.reset(seed=300001)
        oracle = infos["agent_0"]["oracle_answer"]
        _, rewards, _, _, _ = oracle_env.step({"agent_0": oracle})
        assert rewards["agent_0"] == 1.0

        invalid_env.reset(seed=300001)
        _, rewards, _, _, _ = invalid_env.step({"agent_0": "0"})
        assert rewards["agent_0"] == 0.0
    finally:
        oracle_env.close()
        invalid_env.close()
