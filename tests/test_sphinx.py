"""Tests for Sphinx visual reasoning environments."""

from pathlib import Path
import random
import unittest

try:
    import gym_v
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Failed to import `gym_v`. Run tests from the `gym-v/` directory "
        "(e.g. `cd gym-v && python -m unittest ...`) or install it with "
        "`pip install -e gym-v`."
    ) from e


# Sphinx environment IDs
SPHINX_ENVS = [
    "Sphinx/TransformResult-v0",
    "Sphinx/TransformResultPoly-v0",
    "Sphinx/SymmetryFill-v0",
    "Sphinx/SymmetryFillPoly-v0",
    "Sphinx/OddOneOut-v0",
    "Sphinx/OddOneOutPoly-v0",
    "Sphinx/SequenceCompletion-v0",
    "Sphinx/SequenceCompletionPoly-v0",
]


class TestSphinx(unittest.TestCase):
    """Test all Sphinx visual reasoning environments."""

    def _get_output_dir(self, env_id: str) -> Path:
        """Get output directory for a given environment."""
        # Convert "Sphinx/TransformResult-v0" -> "test_output_sphinx_transform_result"
        env_name = env_id.split("/")[1].replace("-v0", "")
        # CamelCase to snake_case
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in env_name
        ).lstrip("_")
        return Path(__file__).resolve().parent / f"test_output_sphinx_{snake_name}"

    def _setup_output_dir(self, output_dir: Path) -> None:
        """Create or clean output directory."""
        if output_dir.exists():
            for p in output_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

    def _test_env(self, env_id: str) -> None:
        """Test a single Sphinx environment."""
        output_dir = self._get_output_dir(env_id)
        self._setup_output_dir(output_dir)

        # Use random seed for each test
        test_seed = random.randint(0, 9999)
        print(f"\n[{env_id}] Using random seed: {test_seed}")

        env = gym_v.make(env_id)
        obs, info = env.reset(seed=test_seed)

        # 1. Save image
        self.assertIsNotNone(obs.image)
        obs.image.save(output_dir / "0_reset.png")

        # 2. Verify oracle answer exists
        oracle = info.get("oracle_answer")
        self.assertIsInstance(oracle, str)
        self.assertGreater(len(oracle), 0)

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

        # 3. Verify reward with correct answer
        obs2, reward, terminated, truncated, info2 = env.step(oracle)
        self.assertTrue(terminated)
        self.assertIsInstance(reward, float)
        self.assertEqual(
            reward, 1.0, f"{env_id}: Expected reward 1.0 for oracle answer"
        )

        # 4. Verify reward with wrong answer
        env.reset(seed=test_seed)
        # Use a clearly wrong answer
        wrong_answer = "(z)"
        _, reward2, terminated2, truncated2, _ = env.step(wrong_answer)
        self.assertTrue(terminated2)
        self.assertIsInstance(reward2, float)
        self.assertEqual(
            reward2, 0.0, f"{env_id}: Expected reward 0.0 for wrong answer"
        )

        print(
            f"✅ {env_id}: Oracle answer verified (seed={test_seed}, answer={oracle})"
        )

    def _test_deterministic(self, env_id: str) -> None:
        """Test that same seed produces same output."""
        test_seed = 42

        env = gym_v.make(env_id)

        # Reset twice with same seed
        obs1, info1 = env.reset(seed=test_seed)
        oracle1 = info1.get("oracle_answer")

        obs2, info2 = env.reset(seed=test_seed)
        oracle2 = info2.get("oracle_answer")

        # Answers must match
        self.assertEqual(
            oracle1,
            oracle2,
            f"{env_id}: Same seed must produce same oracle answer",
        )

        # Images must match (convert to bytes and compare)
        img1_bytes = obs1.image.tobytes()
        img2_bytes = obs2.image.tobytes()
        self.assertEqual(
            img1_bytes,
            img2_bytes,
            f"{env_id}: Same seed must produce same image",
        )

        # Reset with different seed should produce different result
        obs3, info3 = env.reset(seed=test_seed + 1)
        oracle3 = info3.get("oracle_answer")
        img3_bytes = obs3.image.tobytes()

        # At least one of answer or image should differ
        answer_differs = oracle1 != oracle3
        image_differs = img1_bytes != img3_bytes
        self.assertTrue(
            answer_differs or image_differs,
            f"{env_id}: Different seeds should produce different outputs",
        )

        print(f"✅ {env_id}: Deterministic generation verified (seed={test_seed})")

    def _test_multiple_resets(self, env_id: str, num_resets: int = 5) -> None:
        """Test multiple resets produce valid outputs."""
        env = gym_v.make(env_id)

        for i in range(num_resets):
            obs, info = env.reset(seed=i * 100)

            # Image must exist
            self.assertIsNotNone(obs.image)

            # Oracle answer must exist and be valid format (a)-(h)
            oracle = info.get("oracle_answer")
            self.assertIsInstance(oracle, str)
            self.assertIn(
                oracle,
                ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"],
                f"{env_id}: Oracle answer must be (a)-(h), got {oracle}",
            )

            # Step with oracle answer must give reward 1.0
            _, reward, _, _, _ = env.step(oracle)
            self.assertEqual(
                reward,
                1.0,
                f"{env_id}: Reset {i} - Oracle answer should give reward 1.0",
            )

        print(f"✅ {env_id}: Multiple resets verified ({num_resets} resets)")


def _make_test_method(env_id: str):
    """Factory function to create test methods for each environment."""

    def test_method(self):
        self._test_env(env_id)

    # Set a descriptive name for the test
    env_name = env_id.split("/")[1].replace("-v0", "")
    test_method.__name__ = f"test_{env_name.lower()}"
    test_method.__doc__ = f"Test {env_id} environment."
    return test_method


def _make_deterministic_test(env_id: str):
    """Factory function to create deterministic test methods."""

    def test_method(self):
        self._test_deterministic(env_id)

    env_name = env_id.split("/")[1].replace("-v0", "")
    test_method.__name__ = f"test_{env_name.lower()}_deterministic"
    test_method.__doc__ = f"Test {env_id} deterministic generation."
    return test_method


def _make_multiple_resets_test(env_id: str):
    """Factory function to create multiple resets test methods."""

    def test_method(self):
        self._test_multiple_resets(env_id)

    env_name = env_id.split("/")[1].replace("-v0", "")
    test_method.__name__ = f"test_{env_name.lower()}_multiple_resets"
    test_method.__doc__ = f"Test {env_id} multiple resets."
    return test_method


# Dynamically add test methods for each environment
for _env_id in SPHINX_ENVS:
    _test_method = _make_test_method(_env_id)
    setattr(TestSphinx, _test_method.__name__, _test_method)

    _det_test = _make_deterministic_test(_env_id)
    setattr(TestSphinx, _det_test.__name__, _det_test)

    _multi_test = _make_multiple_resets_test(_env_id)
    setattr(TestSphinx, _multi_test.__name__, _multi_test)


if __name__ == "__main__":
    unittest.main()
