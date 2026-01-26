"""
Reward API.

- Individual reward implementations live in `gym_v/rewards`.
- `multi_score(...)` builds a callable that computes per-reward results.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from importlib import import_module
import pkgutil
from typing import Any

import torch

from services.rewards.base import BaseReward
from services.rewards.geneval.geneval_reward import GenevalReward
from services.rewards.registry import REWARD_REGISTRY, register_reward
from services.rewards.utils import _parse_device, _parse_dtype

_AUTO_REGISTERED = False


def _auto_register_rewards() -> None:
    global _AUTO_REGISTERED
    if _AUTO_REGISTERED:
        return
    import services.rewards as pkg

    for module in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if module.name.endswith("_reward"):
            import_module(module.name)
    _AUTO_REGISTERED = True


def build_local_reward(name: str, **kwargs: Any) -> BaseReward:
    _auto_register_rewards()
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward '{name}'. Available: {sorted(REWARD_REGISTRY.keys())}"
        )
    factory = REWARD_REGISTRY[name]
    return factory(**kwargs)


_DEFAULT_REWARD_DEVICES: dict[str, torch.device] = {
    "geneval": torch.device("cpu"),
}


def _build_reward(
    name: str,
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
    init_kwargs: MutableMapping[str, Any],
) -> BaseReward:
    kwargs = dict(init_kwargs)

    if name == "hpsv3":
        if "cfg" in kwargs and "config_path" not in kwargs:
            kwargs["config_path"] = kwargs.pop("cfg")
        if "config" in kwargs and "config_path" not in kwargs:
            kwargs["config_path"] = kwargs.pop("config")
        if "ckpt" in kwargs and "checkpoint_path" not in kwargs:
            kwargs["checkpoint_path"] = kwargs.pop("ckpt")
        if "checkpoint" in kwargs and "checkpoint_path" not in kwargs:
            kwargs["checkpoint_path"] = kwargs.pop("checkpoint")

    if name in {"deqa", "qwenvl", "clipscore", "pickscore", "imagereward", "geneval2"}:
        if "model" in kwargs and "model_name_or_path" not in kwargs:
            kwargs["model_name_or_path"] = kwargs.pop("model")
        if "processor" in kwargs and "processor_name_or_path" not in kwargs:
            kwargs["processor_name_or_path"] = kwargs.pop("processor")
    if device is not None:
        kwargs.setdefault("device", device)
    if dtype is not None:
        kwargs.setdefault("torch_dtype" if name == "deqa" else "dtype", dtype)

    if name in REWARD_REGISTRY:
        if name == "editscore" and "config" not in kwargs:
            raise ValueError("editscore reward requires `config` in reward spec.")
        if name == "deqa":
            if "model_name" in kwargs and "model_name_or_path" not in kwargs:
                kwargs["model_name_or_path"] = kwargs.pop("model_name")
        return build_local_reward(name, **kwargs)

    raise ValueError(f"Unknown reward '{name}'.")


@dataclass(frozen=True)
class _RewardSpec:
    device: torch.device | None
    dtype: torch.dtype | None
    init: dict[str, Any]


def _parse_spec(value: Any) -> _RewardSpec:
    if isinstance(value, int | float):
        raise TypeError(
            "Numeric reward specs are no longer supported. Use a mapping instead."
        )
    if isinstance(value, Mapping):
        raw: dict[str, Any] = dict(value)
        if "weight" in raw or "w" in raw:
            raise ValueError("Reward specs no longer support weight.")
        device = _parse_device(raw.pop("device", None))
        dtype = _parse_dtype(raw.pop("dtype", None))

        init = raw.pop("init", None)
        if init is None:
            init_kwargs = raw
        else:
            if not isinstance(init, Mapping):
                raise TypeError("Reward spec 'init' must be a mapping.")
            init_kwargs = dict(init)
            if raw:
                # Allow legacy flat keys, but let `init` override them.
                merged = dict(raw)
                merged.update(init_kwargs)
                init_kwargs = merged

        # Support putting device/dtype inside init as well (top-level wins).
        init_device = _parse_device(init_kwargs.pop("device", None))
        init_dtype = _parse_dtype(init_kwargs.pop("dtype", None))
        if device is None:
            device = init_device
        if dtype is None:
            dtype = init_dtype

        return _RewardSpec(device=device, dtype=dtype, init=init_kwargs)
    raise TypeError(f"Unsupported reward spec type: {type(value)}")


@dataclass(frozen=True)
class _RewardComponent:
    name: str
    reward: BaseReward


class MultiScore:
    """Callable that computes multiple rewards and returns per-reward results.

    `score_dict` supports:
    - `{"geneval": {}}` (defaults)
    - `{"geneval": {"device": "cuda:0", "init": {...}}}` (per-reward init)
    """

    def __init__(
        self, device: torch.device | str, score_dict: Mapping[str, Any]
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.components: list[_RewardComponent] = []
        for name, spec in score_dict.items():
            parsed = _parse_spec(spec)
            reward_device = (
                parsed.device or _DEFAULT_REWARD_DEVICES.get(name) or self.device
            )
            reward = _build_reward(
                name, device=reward_device, dtype=parsed.dtype, init_kwargs=parsed.init
            )
            self.components.append(_RewardComponent(name=name, reward=reward))

    def __call__(self, images, prompts=None, metadata=None):
        # Keep inputs in their original format (tensor / numpy / PIL / list) and let each reward
        # implementation adapt via `rewards/utils.py`. This matches the legacy behavior more closely
        # (e.g. avoids an extra tensor->PIL->tensor quantization hop for CLIP-based rewards).
        images_input = images
        prompt_list = list(prompts or [])
        score_details: dict[str, Any] = {}

        for component in self.components:
            name = component.name
            result = component.reward(
                images_input, prompts=prompt_list, metadata=metadata
            )
            score_details[name] = result
        return score_details, {}


def multi_score(
    device: torch.device | str, score_dict: Mapping[str, Any]
) -> MultiScore:
    """Build a `MultiScore` callable from a reward spec mapping."""
    return MultiScore(device=device, score_dict=score_dict)


__all__ = [
    "GenevalReward",
    "MultiScore",
    "multi_score",
    "build_local_reward",
    "register_reward",
    "REWARD_REGISTRY",
]
