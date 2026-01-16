"""Unified tests for Perception environments."""

from __future__ import annotations

from pathlib import Path
import random
import unittest

try:
    import gym_v
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Failed to import `gym_v`. Run tests from the `gym-v_perception/` directory "
        "(e.g. `cd gym-v_perception && python -m unittest ...`) or install it with "
        "`pip install -e gym-v_perception`."
    ) from e


PERCEPTION_ENVS = {
    "Perception/ChartToTable-v0": "chart_to_table",
    "Perception/GraphToAdjacency-v0": "graph_to_adjacency",
    "Perception/TreeToTraversal-v0": "tree_to_traversal",
    "Perception/DAGToTopoOrder-v0": "dag_to_topo_order",
    "Perception/GraphToMST-v0": "graph_to_mst",
    "Perception/FlowNetwork-v0": "flow_network",
    "Perception/FunctionGraph-v0": "function_graph",
    "Perception/ContourPlot-v0": "contour_plot",
    "Perception/PolarPlot-v0": "polar_plot",
    "Perception/VectorField-v0": "vector_field",
    "Perception/ParametricCurve-v0": "parametric_curve",
}


class TestPerception(unittest.TestCase):
    """Test Perception environments.

    Note: Perception environments return reward=0.0 from inner_step() by design.
    The reward calculation is left to external evaluators that can properly
    compare the agent's extracted data with the oracle answer.

    These tests verify:
    1. Environment can be created and reset
    2. Observations contain valid images
    3. Oracle answers are provided and non-empty
    4. Step function works correctly
    5. Multiple seeds generate valid puzzles
    """

    def _get_output_dir(self, env_id: str) -> Path:
        env_name = env_id.split("/")[1].replace("-v0", "")
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in env_name
        ).lstrip("_")
        return Path(__file__).resolve().parent / f"test_output_perception_{snake_name}"

    def _setup_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _test_env(self, env_id: str, env_name: str) -> None:
        output_dir = self._get_output_dir(env_id)
        self._setup_output_dir(output_dir)

        test_seed = random.randint(0, 9999)
        print(f"\n[{env_id}] Using random seed: {test_seed}")

        env = gym_v.make(env_id)

        # 1. Reset - Perception envs return (Observation, info) directly
        obs, info = env.reset(seed=test_seed)

        # Check Observation structure
        self.assertIsNotNone(obs.image, f"{env_id}: obs.image should not be None")
        obs.image.save(output_dir / "0_reset.png")

        oracle = info.get("oracle_answer")
        self.assertIsInstance(oracle, str, f"{env_id}: oracle_answer should be a string")
        self.assertGreater(len(oracle), 0, f"{env_id}: oracle_answer should not be empty")

        print("\n" + "=" * 80)
        print(f"[{env_id}] SEED: {test_seed}")
        print(f"[{env_id}] DESCRIPTION:\n")
        print(env.description[:500] if len(env.description) > 500 else env.description)
        print(f"\n[{env_id}] OBS.TEXT:\n")
        text = obs.text or "No text"
        print(text[:500] if len(text) > 500 else text)
        print(f"\n[{env_id}] ORACLE ANSWER:\n")
        print(oracle[:300] + "..." if len(oracle) > 300 else oracle)
        print("=" * 80 + "\n")

        # 2. Step - verify step function works with correct answer
        obs2, reward, terminated, truncated, info2 = env.step(oracle)

        self.assertTrue(terminated, f"{env_id}: terminated should be True after step")
        self.assertIsInstance(reward, float, f"{env_id}: reward should be a float")
        self.assertEqual(reward, 1.0, f"{env_id}: Expected reward 1.0 for oracle answer, got {reward}")

        # 3. Verify info still contains oracle_answer after step
        oracle2 = info2.get("oracle_answer")
        self.assertIsNotNone(oracle2, f"{env_id}: info should contain oracle_answer after step")

        # 4. Test with wrong answer
        env.reset(seed=test_seed)
        obs_wrong, reward_wrong, terminated_wrong, truncated_wrong, info_wrong = env.step("")
        self.assertTrue(terminated_wrong, f"{env_id}: terminated should be True for wrong answer")
        self.assertEqual(reward_wrong, 0.0, f"{env_id}: Expected reward 0.0 for wrong answer, got {reward_wrong}")

        # 5. Test with multiple seeds
        print(f"[{env_id}] Testing with 3 additional seeds...")
        for i in range(3):
            seed = random.randint(0, 9999)
            obs_test, info_test = env.reset(seed=seed)

            oracle_test = info_test.get("oracle_answer")

            self.assertIsNotNone(obs_test.image, f"{env_id}: obs.image should not be None (seed={seed})")
            obs_test.image.save(output_dir / f"{i + 1}_seed_{seed}.png")

            self.assertIsNotNone(oracle_test, f"{env_id}: oracle_answer should not be None (seed={seed})")
            self.assertIsInstance(oracle_test, str, f"{env_id}: oracle_answer should be string (seed={seed})")
            self.assertGreater(len(oracle_test), 0, f"{env_id}: oracle_answer should not be empty (seed={seed})")

            # Verify step works
            obs_step, reward_step, term_step, trunc_step, info_step = env.step(oracle_test)
            self.assertTrue(term_step, f"{env_id}: terminated should be True (seed={seed})")

            print(f"  ✓ Seed {seed}: Generated valid puzzle with oracle answer")

        env.close()
        print(f"✅ {env_id}: All tests passed (primary_seed={test_seed})")


def _make_test_method(env_id: str, env_name: str):
    def test_method(self):
        self._test_env(env_id, env_name)

    test_method.__name__ = f"test_{env_name.lower()}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


for _env_id, _env_name in PERCEPTION_ENVS.items():
    _test_method = _make_test_method(_env_id, _env_name)
    setattr(TestPerception, _test_method.__name__, _test_method)


if __name__ == "__main__":
    unittest.main()
