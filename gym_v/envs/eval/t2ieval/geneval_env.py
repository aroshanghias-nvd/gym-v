from __future__ import annotations

from collections import defaultdict
import json
import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from gym_v.core import Env, Observation
from gym_v.envs.eval.t2ieval.client import (
    RewardClient,
    decode_data_url,
    ensure_list,
    image_to_data_url,
    parse_openai_payload,
    run_coroutine,
)
from gym_v.logger import get_logger
from gym_v.utils.image import to_pil_list

logger = get_logger()


class GenevalPromptDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | None = None,
        split: str = "test",
        file_path: str | None = None,
    ):
        if file_path is None and dataset_root and dataset_root.endswith(".jsonl"):
            file_path = dataset_root
        if file_path is None:
            if not dataset_root:
                raise ValueError(
                    "dataset_root is required unless file_path is provided."
                )
            file_path = os.path.join(dataset_root, f"{split}_metadata.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Geneval metadata not found: {file_path}")

        self.file_path = file_path
        with open(self.file_path, encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
        self.prompts = [item.get("prompt", "") for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


def _postprocess_geneval_results(results: list[dict[str, Any]]):
    required_keys = [
        "single_object",
        "two_object",
        "counting",
        "colors",
        "position",
        "color_attr",
    ]
    scores = []
    strict_rewards = []
    grouped_strict_rewards = defaultdict(list)
    rewards = []
    grouped_rewards = defaultdict(list)
    for result in results:
        strict_rewards.append(1.0 if result.get("strict_correct") else 0.0)
        scores.append(result.get("score"))
        rewards.append(1.0 if result.get("correct") else 0.0)
        tag = result.get("tag")
        for key in required_keys:
            if key != tag:
                grouped_strict_rewards[key].append(-10.0)
                grouped_rewards[key].append(-10.0)
            else:
                grouped_strict_rewards[tag].append(
                    1.0 if result.get("strict_correct") else 0.0
                )
                grouped_rewards[tag].append(1.0 if result.get("correct") else 0.0)
    return (
        scores,
        rewards,
        strict_rewards,
        dict(grouped_rewards),
        dict(grouped_strict_rewards),
    )


def _preprocess_geneval_payload(
    image: Image.Image,
    prompt: str | None,
    metadata: Any,
    *,
    only_strict: bool,
    model: str,
) -> dict[str, Any]:
    if metadata is None:
        raise ValueError("metadata is required for geneval requests.")

    prompt_text = "" if prompt is None else str(prompt)
    data_url = image_to_data_url(image)
    meta_datas = metadata if isinstance(metadata, list) else [metadata]
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "metadata": {
            "meta_datas": meta_datas,
            "only_strict": bool(only_strict),
        },
    }


def collect_geneval_results(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for response in responses:
        item_results = parse_openai_payload(response)["geneval"]
        if len(item_results) != 1:
            raise ValueError(
                f"Expected 1 geneval result per request, got {len(item_results)}."
            )
        results.append(item_results[0])
    return results


def geneval_score_async(
    *,
    server_url: str = "http://127.0.0.1:18085",
    only_strict: bool = True,
    timeout_s: float = 120.0,
    max_retries: int = 1000,
    backoff_factor: float = 1.0,
    model: str = "geneval",
):
    client = RewardClient(
        server_url=server_url,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
    )

    async def _fn(images, prompts=None, metadatas=None, only_strict_override=None):
        images_list = list(images or [])
        prompts_list = ensure_list(prompts, len(images_list), "")
        metadatas_list = ensure_list(metadatas, len(images_list), None)
        if len(prompts_list) != len(images_list) or len(metadatas_list) != len(
            images_list
        ):
            raise ValueError("prompts/metadatas length must match images length.")
        strict = only_strict if only_strict_override is None else only_strict_override

        payloads = [
            _preprocess_geneval_payload(
                image,
                prompt,
                metadata,
                only_strict=strict,
                model=model,
            )
            for image, prompt, metadata in zip(
                images_list, prompts_list, metadatas_list, strict=False
            )
        ]
        responses = await client.request_many(payloads)
        results = collect_geneval_results(responses)
        return _postprocess_geneval_results(results)

    return _fn


class GenevalEnv(Env):
    """
    T2I eval environment for GenEval.

    reset(): returns prompts + metadata for all items (or a subset via options).
    step(): expects images as actions, queries the GenEval reward server, returns rewards.
    """

    def __init__(
        self,
        dataset_root: str | None = None,
        dataset_path: str | None = None,
        split: str = "test",
        server_url: str | None = None,
        only_strict: bool = True,
        timeout_s: float = 120.0,
        max_retries: int = 1000,
        backoff_factor: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(max_episode_steps=1)

        if dataset_root is None and dataset_path is None:
            dataset_root = os.environ.get("GENEVAL_DATASET_ROOT")

        self.dataset = GenevalPromptDataset(
            dataset_root=dataset_root, split=split, file_path=dataset_path
        )
        self.prompts = self.dataset.prompts
        self.metadatas = self.dataset.metadatas
        self.indices = list(range(len(self.prompts)))

        self._agent_ids = {f"agent_{i}" for i in self.indices}
        self._agent_id_to_index = {f"agent_{i}": i for i in self.indices}
        self._active_indices = self.indices

        self.only_strict = only_strict
        self.timeout_s = timeout_s

        if server_url is None:
            server_url = (
                os.environ.get("GENEVAL_REWARD_URL")
                or os.environ.get("GENEVAL_SERVER_URL")
                or "http://127.0.0.1:18085"
            )
        self.server_url = server_url

        self._score_async_fn = geneval_score_async(
            server_url=self.server_url,
            only_strict=self.only_strict,
            timeout_s=self.timeout_s,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    @property
    def description(self) -> str:
        return f"GenEval T2I eval env with {len(self.prompts)} prompts."

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        indices = self.indices
        if options:
            if "indices" in options:
                indices = list(options["indices"])
            elif "max_samples" in options:
                max_samples = int(options["max_samples"])
                indices = indices[:max_samples]
            if options.get("shuffle"):
                indices = list(indices)
                self.np_random.shuffle(indices)

        self._active_indices = indices
        self._agent_ids = {f"agent_{i}" for i in indices}
        self._agent_id_to_index = {f"agent_{i}": i for i in indices}

        obs_dict = {}
        info_dict = {}
        for idx in indices:
            agent_id = f"agent_{idx}"
            prompt = self.prompts[idx]
            metadata = self.metadatas[idx]
            obs_dict[agent_id] = Observation(image=None, text=prompt, metadata=metadata)
            info_dict[agent_id] = {
                "index": idx,
                "prompt": prompt,
                "metadata": metadata,
            }

        return obs_dict, info_dict

    def step(self, action_dict):
        if not action_dict:
            raise ValueError("action_dict is empty; expected generated images.")

        unknown = set(action_dict) - set(self._agent_id_to_index)
        if unknown:
            raise ValueError(f"Unknown agent ids in action_dict: {sorted(unknown)}")

        missing = set(self._agent_id_to_index) - set(action_dict)
        if missing:
            logger.warning("Missing actions for %d prompts.", len(missing))

        ordered_agents = sorted(
            action_dict.keys(), key=lambda aid: self._agent_id_to_index[aid]
        )

        images = []
        prompts = []
        metadatas = []

        for agent_id in ordered_agents:
            action = action_dict[agent_id]
            pil_image = self._action_to_pil(action)
            images.append(pil_image)
            idx = self._agent_id_to_index[agent_id]
            prompts.append(self.prompts[idx])
            metadatas.append(self.metadatas[idx])

        scores, rewards, strict_rewards, group_rewards, group_strict_rewards = (
            run_coroutine(
                self._score_async_fn(images, prompts=prompts, metadatas=metadatas)
            )
        )

        reward_dict = {}
        info_dict = {}
        for idx, agent_id in enumerate(ordered_agents):
            score = scores[idx] if idx < len(scores) else None
            reward = rewards[idx] if idx < len(rewards) else None
            strict_reward = strict_rewards[idx] if idx < len(strict_rewards) else None
            reward_value = strict_reward if strict_reward is not None else reward
            if reward_value is None:
                reward_value = 0.0
            reward_dict[agent_id] = float(reward_value)
            info_dict[agent_id] = {
                "index": self._agent_id_to_index[agent_id],
                "score": score,
                "reward": reward,
                "strict_reward": strict_reward,
            }

        terminated = {agent_id: True for agent_id in ordered_agents}
        truncated = {agent_id: False for agent_id in ordered_agents}
        terminated["__all__"] = True
        truncated["__all__"] = False
        info_dict["__all__"] = {
            "group_rewards": group_rewards,
            "group_strict_rewards": group_strict_rewards,
        }

        return {}, reward_dict, terminated, truncated, info_dict

    def _action_to_pil(self, action: Any) -> Image.Image:
        if isinstance(action, dict):
            if "image" in action:
                action = action["image"]
            elif "images" in action:
                action = action["images"]

        if isinstance(action, list | tuple) and len(action) == 1:
            action = action[0]

        if isinstance(action, str) and action.startswith("data:"):
            action = decode_data_url(action)

        images = to_pil_list(action)
        if len(images) != 1:
            raise ValueError(f"Expected single image per prompt, got {len(images)}.")
        return images[0].convert("RGB")
