from __future__ import annotations

from typing import Any

import torch

from services.rewards.base import BaseReward
from services.rewards.geneval.gen_eval import load_geneval
from services.rewards.registry import register_reward


@register_reward("geneval")
class GenevalReward(BaseReward):
    """
    Local Geneval reward wrapper. Uses vendored gen_eval under rewards/local/geneval.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        config_path: str | None = None,
        ckpt_root: str | None = None,
        object_names_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, **kwargs)
        self.inference_fn = load_geneval(
            device=str(self.device),
            config_path=config_path,
            ckpt_root=ckpt_root,
            object_names_path=object_names_path,
        )

    def __call__(self, images, prompts=None, metadata=None):
        meta_datas = []
        only_strict = True
        if metadata:
            if isinstance(metadata, dict):
                meta_datas = metadata.get("meta_datas", meta_datas)
                only_strict = metadata.get("only_strict", only_strict)
            else:
                meta_datas = metadata

        return self.inference_fn(images, meta_datas, only_strict)
