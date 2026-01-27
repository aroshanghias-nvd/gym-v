"""Tests for BrowserGym MiniWoB environment integration.

This module tests the integration of MiniWoB environments from BrowserGym into gym-v.
Tests verify environment creation, observation format, action execution, and the
multi-agent dictionary interface for interactive web tasks.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import gym_v
from gym_v.core import Observation


# Representative sample of MiniWoB tasks to test
# These cover different task types and complexities
TEST_ENVS = {
    "MiniWoB/ClickTest-v0": "click_test",  # Simple click task
    "MiniWoB/ClickButton-v0": "click_button",  # Multiple buttons
    "MiniWoB/LoginUser-v0": "login_user",  # Form filling
    "MiniWoB/ChooseList-v0": "choose_list",  # Dropdown selection
}


class TestMiniWoBEnvironments(unittest.TestCase):
    """Test suite for MiniWoB environment integration.

    Tests verify:
    1. Environment creation and reset
    2. Observation format (image, text, metadata)
    3. Action space and execution
    4. Multi-agent dictionary interface
    5. Episode termination
    """

    def _test_env(self, env_id: str) -> None:
        """Test a single MiniWoB environment.

        Args:
            env_id: Environment ID for gym_v.make()
        """
        print(f"\nTesting {env_id}")

        # Create environment
        env = gym_v.make(env_id, headless=True)

        # Test 1: Reset returns correct format
        obs_dict, info_dict = env.reset(seed=42)

        agent_id = "agent_0"
        self.assertIn(agent_id, obs_dict, f"{env_id}: agent_0 not in obs_dict")
        self.assertIn(agent_id, info_dict, f"{env_id}: agent_0 not in info_dict")

        obs: Observation = obs_dict[agent_id]

        # Test 2: Observation has required components
        self.assertIsNotNone(obs.image, f"{env_id}: obs.image is None")
        self.assertIsNotNone(obs.text, f"{env_id}: obs.text is None")
        self.assertIsInstance(obs.text, str, f"{env_id}: obs.text is not string")
        self.assertGreater(len(obs.text), 0, f"{env_id}: obs.text is empty")

        # Verify image dimensions are reasonable
        width, height = obs.image.size
        self.assertGreater(width, 0, f"{env_id}: image width is 0")
        self.assertGreater(height, 0, f"{env_id}: image height is 0")

        # Test 3: Environment has description
        desc = env.description
        self.assertIsInstance(desc, str, f"{env_id}: description is not string")
        self.assertGreater(len(desc), 0, f"{env_id}: description is empty")

        print(f"  Task goal: {obs.text[:100]}...")

        # Test 4: Step function works
        action_dict = {agent_id: "noop()"}
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(
            action_dict
        )

        # Verify step returns correct types
        self.assertIn(agent_id, obs_dict)
        self.assertIn(agent_id, reward_dict)
        self.assertIn(agent_id, terminated_dict)
        self.assertIn(agent_id, truncated_dict)

        self.assertIsInstance(obs_dict[agent_id], Observation)
        self.assertIsInstance(reward_dict[agent_id], (int, float))
        self.assertIsInstance(terminated_dict[agent_id], bool)
        self.assertIsInstance(truncated_dict[agent_id], bool)

        # Test 5: Episode terminates (run until done or max steps)
        step_count = 1
        max_steps = 20

        while (
            not terminated_dict[agent_id]
            and not truncated_dict[agent_id]
            and step_count < max_steps
        ):
            action_dict = {agent_id: "noop()"}
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = (
                env.step(action_dict)
            )
            step_count += 1

        # Episode should terminate or truncate
        self.assertTrue(
            terminated_dict[agent_id] or truncated_dict[agent_id],
            f"{env_id}: Episode did not terminate after {max_steps} steps",
        )

        print(f"  Episode finished in {step_count} steps")

        # Test 6: Environment can be reset multiple times
        for i in range(2):
            obs_dict, info_dict = env.reset(seed=42 + i)
            self.assertIn(agent_id, obs_dict)
            self.assertIsNotNone(obs_dict[agent_id].image)
            self.assertIsNotNone(obs_dict[agent_id].text)

        env.close()
        print(f"  ✅ All tests passed for {env_id}")


def _make_test_method(env_id: str, env_name: str):
    """Factory function to dynamically create test methods.

    Args:
        env_id: Environment ID
        env_name: Environment name for test method naming

    Returns:
        Test method function
    """

    def test_method(self):
        self._test_env(env_id)

    test_method.__name__ = f"test_{env_name}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


# Dynamically generate test methods for sample MiniWoB environments
for _env_id, _env_name in TEST_ENVS.items():
    _test_method = _make_test_method(_env_id, _env_name)
    setattr(TestMiniWoBEnvironments, _test_method.__name__, _test_method)


if __name__ == "__main__":
    unittest.main()
