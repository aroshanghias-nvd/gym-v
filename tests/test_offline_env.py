"""Tests for offline dataset-backed envs."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import gym_v
import gym_v.envs  # noqa: F401  # register built-in envs


def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string (OpenAI API format)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class TestOfflineSingleTurnEnv(unittest.TestCase):
    def _write_small_dataset(self, root: Path) -> Path:
        """Create a small 2-sample dataset for basic tests."""
        img = Image.new("RGB", (32, 32), (255, 0, 0))
        img_b64 = _image_to_base64(img)

        jsonl_path = root / "dataset.jsonl"
        rows = [
            {
                "text": "Q1: What is 2+2?",
                "image": img_b64,
                "answer": "4",
                "metadata": {"id": 1},
            },
            {
                "text": "Q2: Capital of France?",
                "image": img_b64,
                "answer": "Paris",
                "metadata": {"id": 2},
            },
        ]
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return jsonl_path

    def _write_large_dataset(self, root: Path, num_samples: int) -> Path:
        """Create a dataset with num_samples distinct samples."""
        img = Image.new("RGB", (32, 32), (0, 255, 0))
        img_b64 = _image_to_base64(img)

        jsonl_path = root / "dataset.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for i in range(num_samples):
                row = {
                    "text": f"Question {i}: What is {i} + 1?",
                    "image": img_b64,
                    "answer": str(i + 1),
                    "metadata": {"id": i},
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return jsonl_path

    def test_batch_sampling_no_duplicates(self):
        """Test that batch sampling (multiple agents) does not repeat samples within a batch.

        When num_players equals dataset size, one reset should sample each data point exactly once.
        """
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            num_samples = 10
            dataset_path = self._write_large_dataset(root, num_samples)

            env = gym_v.make(
                "Offline/SingleTurn-v0",
                datasource_type="jsonl",
                datasource_kwargs={"data_path": str(dataset_path)},
                shuffle=True,
                grader="exact_match",
                num_players=num_samples,
            )

            obs_dict, info_dict = env.reset(seed=42)

            # Verify we got num_samples agents
            self.assertEqual(len(obs_dict), num_samples)
            self.assertEqual(len(info_dict), num_samples)

            # Collect all sampled indices
            sampled_indices = set()
            for i in range(num_samples):
                agent_id = f"agent_{i}"
                idx = info_dict[agent_id]["index"]
                sampled_indices.add(idx)

            # Verify no duplicates: set size should equal num_samples
            self.assertEqual(
                len(sampled_indices),
                num_samples,
                f"Expected {num_samples} unique samples, but got {len(sampled_indices)}. "
                f"Sampled indices: {sorted(sampled_indices)}",
            )

            # Verify all indices 0 to num_samples-1 were sampled exactly once
            self.assertEqual(
                sampled_indices,
                set(range(num_samples)),
                "Expected indices [0, 1, ..., 9] but got different set",
            )

            env.close()

    def test_observation_and_grading(self):
        """Test observation completeness and grading mechanism."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dataset_path = self._write_small_dataset(root)

            env = gym_v.make(
                "Offline/SingleTurn-v0",
                datasource_type="jsonl",
                datasource_kwargs={"data_path": str(dataset_path)},
                shuffle=True,
                grader="exact_match",
                num_players=1,
            )

            agent_id = "agent_0"

            # Test observation completeness
            obs_dict, info_dict = env.reset(seed=123)
            self.assertIsNotNone(obs_dict[agent_id].image)
            self.assertIsInstance(obs_dict[agent_id].text, str)
            self.assertIsInstance(info_dict[agent_id]["oracle_answer"], str)

            # Test correct answer gets reward 1.0 (exact_match ignores whitespace)
            oracle = info_dict[agent_id]["oracle_answer"]
            _, reward_dict, term_dict, trunc_dict, info_dict = env.step(
                {agent_id: f"  {oracle}  "}
            )
            self.assertEqual(reward_dict[agent_id], 1.0)
            self.assertTrue(info_dict[agent_id]["correct"])
            self.assertTrue(term_dict["__all__"])
            self.assertTrue(trunc_dict["__all__"])

            # Test wrong answer gets reward 0.0
            env.reset(seed=123)
            _, reward_dict, _, _, info_dict = env.step({agent_id: "__wrong__"})
            self.assertEqual(reward_dict[agent_id], 0.0)
            self.assertFalse(info_dict[agent_id]["correct"])

            env.close()


if __name__ == "__main__":
    unittest.main()
