from __future__ import annotations

import os
import socket
from typing import Any
import unittest
from urllib.parse import urlparse

try:
    import torch
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from diffusers import DiffusionPipeline
except Exception as exc:  # pragma: no cover - optional dependency
    DiffusionPipeline = None
    _DIFFUSERS_IMPORT_ERROR = exc
else:
    _DIFFUSERS_IMPORT_ERROR = None

try:
    from gym_v.envs.eval.t2ieval.geneval_env import GenevalEnv
except Exception as exc:  # pragma: no cover - optional dependency
    GenevalEnv = None
    _GENEVAL_IMPORT_ERROR = exc
else:
    _GENEVAL_IMPORT_ERROR = None


def _server_reachable(url: str, timeout_s: float = 1.0) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _resolve_device(device: str | None) -> str:
    if torch is None:
        return "cpu"
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str) -> Any:
    if torch is None:
        return None
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


class TestGenevalEnvReal(unittest.TestCase):
    def setUp(self):
        if os.environ.get("RUN_GENEVAL_REAL_TEST") != "1":
            self.skipTest("Set RUN_GENEVAL_REAL_TEST=1 to enable this test.")
        if torch is None:
            self.skipTest(f"torch import failed: {_TORCH_IMPORT_ERROR}")
        if DiffusionPipeline is None:
            self.skipTest(f"diffusers import failed: {_DIFFUSERS_IMPORT_ERROR}")
        if GenevalEnv is None:
            self.skipTest(f"GenevalEnv import failed: {_GENEVAL_IMPORT_ERROR}")

        self.dataset_path = os.environ.get("GENEVAL_DATASET_PATH")
        self.dataset_root = os.environ.get("GENEVAL_DATASET_ROOT")
        if not self.dataset_path and not self.dataset_root:
            self.skipTest("Set GENEVAL_DATASET_PATH or GENEVAL_DATASET_ROOT.")

        self.server_url = os.environ.get("GENEVAL_SERVER_URL", "http://127.0.0.1:18085")
        if not _server_reachable(self.server_url):
            self.skipTest(f"Reward server not reachable at {self.server_url}.")

        self.model_path = os.environ.get("DIFFUSION_MODEL_PATH")
        if not self.model_path:
            self.skipTest("Set DIFFUSION_MODEL_PATH to a local diffusion model path.")

        self.device = _resolve_device(os.environ.get("DIFFUSION_DEVICE"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.skipTest("DIFFUSION_DEVICE=cuda requested but CUDA is unavailable.")

        self.dtype = _resolve_dtype(self.device)
        self.max_samples = int(os.environ.get("GENEVAL_REAL_MAX_SAMPLES", "1"))
        self.num_steps = int(os.environ.get("DIFFUSION_NUM_STEPS", "4"))
        self.guidance_scale = float(os.environ.get("DIFFUSION_GUIDANCE", "3.0"))
        self.height = int(os.environ.get("DIFFUSION_HEIGHT", "256"))
        self.width = int(os.environ.get("DIFFUSION_WIDTH", "256"))
        self.seed = int(os.environ.get("DIFFUSION_SEED", "1234"))

    def _build_pipeline(self) -> DiffusionPipeline:
        kwargs = {}
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype
        try:
            pipe = DiffusionPipeline.from_pretrained(
                self.model_path, safety_checker=None, **kwargs
            )
        except TypeError:
            pipe = DiffusionPipeline.from_pretrained(self.model_path, **kwargs)
        pipe = pipe.to(self.device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        return pipe

    def test_geneval_real_pipeline(self):
        env = GenevalEnv(
            dataset_path=self.dataset_path,
            dataset_root=self.dataset_root,
            server_url=self.server_url,
        )
        obs_dict, _ = env.reset(options={"max_samples": self.max_samples})
        agent_ids = sorted(obs_dict.keys(), key=lambda aid: int(aid.split("_")[1]))
        prompts = [obs_dict[aid].text for aid in agent_ids]

        pipe = self._build_pipeline()
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        with torch.inference_mode():
            output = pipe(
                prompts,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                height=self.height,
                width=self.width,
                generator=generator,
            )

        images = list(getattr(output, "images", output))
        self.assertEqual(len(images), len(agent_ids))

        actions = {
            agent_id: {"image": image}
            for agent_id, image in zip(agent_ids, images, strict=False)
        }
        _, rewards, terminated, truncated, info = env.step(actions)

        self.assertEqual(set(rewards.keys()), set(agent_ids))
        self.assertTrue(terminated["__all__"])
        self.assertFalse(truncated["__all__"])
        self.assertIn("group_rewards", info["__all__"])
        self.assertIn("group_strict_rewards", info["__all__"])
        for value in rewards.values():
            self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
