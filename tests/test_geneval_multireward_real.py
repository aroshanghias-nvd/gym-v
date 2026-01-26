from __future__ import annotations

import os
from typing import Any
import unittest

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
    from gym_v.envs.eval.t2ieval.geneval_env import GenevalPromptDataset
except Exception as exc:  # pragma: no cover - optional dependency
    GenevalPromptDataset = None
    _GENEVAL_IMPORT_ERROR = exc
else:
    _GENEVAL_IMPORT_ERROR = None

try:
    from services.rewards import multi_score
except Exception as exc:  # pragma: no cover - optional dependency
    multi_score = None
    _MULTISCORE_IMPORT_ERROR = exc
else:
    _MULTISCORE_IMPORT_ERROR = None


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


_GENEVAL_CKPT_FILENAME = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"

_METADATA_TEMPLATES = [
    {
        "tag": "counting",
        "include": [{"class": "wine glass", "count": 2}],
        "exclude": [{"class": "wine glass", "count": 3}],
        "prompt": "a photo of two wine glasses",
    },
    {
        "tag": "color_attr",
        "include": [
            {"class": "handbag", "count": 1, "color": "pink"},
            {"class": "scissors", "count": 1, "color": "black"},
        ],
        "prompt": "a photo of a pink handbag and a black scissors",
    },
]


class TestGenevalMultiRewardReal(unittest.TestCase):
    def setUp(self):
        if os.environ.get("RUN_GENEVAL_MULTI_REWARD_TEST") != "1":
            self.skipTest("Set RUN_GENEVAL_MULTI_REWARD_TEST=1 to enable this test.")
        if torch is None:
            self.skipTest(f"torch import failed: {_TORCH_IMPORT_ERROR}")
        if DiffusionPipeline is None:
            self.skipTest(f"diffusers import failed: {_DIFFUSERS_IMPORT_ERROR}")
        if GenevalPromptDataset is None:
            self.skipTest(f"Geneval import failed: {_GENEVAL_IMPORT_ERROR}")
        if multi_score is None:
            self.skipTest(f"multi_score import failed: {_MULTISCORE_IMPORT_ERROR}")

        self.dataset_path = os.environ.get("GENEVAL_DATASET_PATH")
        self.dataset_root = os.environ.get("GENEVAL_DATASET_ROOT")

        self.model_path = os.environ.get("DIFFUSION_MODEL_PATH")
        if not self.model_path:
            self.skipTest("Set DIFFUSION_MODEL_PATH to a local diffusion model path.")

        self.device = _resolve_device(os.environ.get("DIFFUSION_DEVICE"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.skipTest("DIFFUSION_DEVICE=cuda requested but CUDA is unavailable.")

        self.dtype = _resolve_dtype(self.device)
        self.max_samples = int(os.environ.get("GENEVAL_MULTI_REWARD_MAX_SAMPLES", "1"))
        self.num_steps = int(os.environ.get("DIFFUSION_NUM_STEPS", "4"))
        self.guidance_scale = float(os.environ.get("DIFFUSION_GUIDANCE", "3.0"))
        self.height = int(os.environ.get("DIFFUSION_HEIGHT", "256"))
        self.width = int(os.environ.get("DIFFUSION_WIDTH", "256"))
        self.seed = int(os.environ.get("DIFFUSION_SEED", "1234"))
        self.only_strict = (
            os.environ.get("GENEVAL_MULTI_REWARD_ONLY_STRICT", "1") != "0"
        )

        self.reward_device = _resolve_device(os.environ.get("GENEVAL_DEVICE"))
        if self.reward_device.startswith("cuda") and not torch.cuda.is_available():
            self.skipTest("GENEVAL_DEVICE=cuda requested but CUDA is unavailable.")

        self.config_path = os.environ.get("GENEVAL_CONFIG_PATH")
        self.ckpt_root = os.environ.get("GENEVAL_CKPT_ROOT")
        if not self.config_path or not self.ckpt_root:
            self.skipTest("Set GENEVAL_CONFIG_PATH and GENEVAL_CKPT_ROOT.")
        if not os.path.isfile(self.config_path):
            self.skipTest(f"GENEVAL_CONFIG_PATH not found: {self.config_path}")
        ckpt_path = os.path.join(self.ckpt_root, _GENEVAL_CKPT_FILENAME)
        if not os.path.isfile(ckpt_path):
            self.skipTest(f"GenEval checkpoint not found: {ckpt_path}")

        self.object_names_path = os.environ.get("GENEVAL_OBJECT_NAMES_PATH")
        if self.object_names_path and not os.path.isfile(self.object_names_path):
            self.skipTest(
                f"GENEVAL_OBJECT_NAMES_PATH not found: {self.object_names_path}"
            )

    def _build_prompts_and_metadata(self) -> tuple[list[str], list[dict[str, Any]]]:
        if (
            self.dataset_path or self.dataset_root
        ) and GenevalPromptDataset is not None:
            dataset = GenevalPromptDataset(
                dataset_root=self.dataset_root, file_path=self.dataset_path
            )
            metadatas = dataset.metadatas[: self.max_samples]
            prompts = [meta.get("prompt", "") for meta in metadatas]
            return prompts, metadatas

        templates = _METADATA_TEMPLATES
        metadatas = [templates[idx % len(templates)] for idx in range(self.max_samples)]
        prompts = [meta.get("prompt", "") for meta in metadatas]
        return prompts, metadatas

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

    def test_geneval_multireward_local(self):
        prompts, metadatas = self._build_prompts_and_metadata()
        if not prompts:
            self.skipTest("No prompts available for multi-reward test.")

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
        self.assertEqual(len(images), len(prompts))

        init_kwargs = {
            "config_path": self.config_path,
            "ckpt_root": self.ckpt_root,
        }
        if self.object_names_path:
            init_kwargs["object_names_path"] = self.object_names_path

        score_dict = {
            "geneval": {
                "device": self.reward_device,
                "init": init_kwargs,
            }
        }
        scorer = multi_score(self.reward_device, score_dict)
        score_details, _ = scorer(
            images,
            prompts=prompts,
            metadata={"meta_datas": metadatas, "only_strict": self.only_strict},
        )

        self.assertIn("geneval", score_details)
        result = score_details["geneval"]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(images))
        for item in result:
            self.assertIsInstance(item, dict)
            self.assertIn("score", item)
            self.assertIn("correct", item)
            self.assertIn("strict_correct", item)


if __name__ == "__main__":
    unittest.main()
