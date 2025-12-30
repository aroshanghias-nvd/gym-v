"""A generic single-turn environment backed by offline examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.offline.graders import GRADERS
from gym_v.envs.offline.sources import DatasetSource, JsonlDatasetSource, OfflineSample

logger = get_logger()


class OfflineSingleTurnEnv(Env):
    """Single-turn env reading (obs, answer) from offline data.

    This env is intended for evaluation / supervised-style tasks:
    - reset(): samples an example and returns Observation(image, text, metadata)
    - step(action): grades the action against oracle answer and terminates

    Supported sources:
    - JSONL file via `dataset_path=...`

    Sampling:
    - sampling="sequential": iterate from 0..N-1
    - sampling="shuffle": iterate a random permutation (no repeats within an epoch)

    Grading:
    - grader="exact_match" (default)
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        sampling: str = "shuffle",
        grader: str = "exact_match",
        description: str | None = None,
        cache_images: bool = False,
        validate_files: bool = True,
        **kwargs: Any,
    ):
        # `max_episode_steps` is normally injected by `gym_v.make` from EnvSpec.
        super().__init__(**kwargs)
        if dataset_path is None:
            raise ValueError("`dataset_path` is required for OfflineSingleTurnEnv")

        self._dataset_path = str(dataset_path)
        self._source: DatasetSource = self._build_source(
            self._dataset_path, validate_files=validate_files
        )
        self._sampling = sampling
        self._grader_name = grader
        self._grader = GRADERS.get(grader)
        if self._grader is None:
            raise ValueError(
                f"Unknown grader={grader}. Available: {sorted(GRADERS.keys())}"
            )

        self._description = description or (
            "Offline single-turn environment.\n\n"
            "Action is a string answer. Reward is determined by the configured grader.\n"
            "Default grader: exact_match (whitespace/case normalized).\n"
        )

        self._cache_images = cache_images
        self._image_cache: dict[str, Image.Image] = {}

        self._cursor = 0
        self._shuffle_order: list[int] | None = None
        self._shuffle_pos = 0
        self._shuffle_epoch = 0
        self._current: OfflineSample | None = None
        self._current_index: int | None = None

        self._index_samplers = {
            "sequential": self._sample_index_sequential,
            "shuffle": self._sample_index_shuffle,
        }

    @staticmethod
    def _build_source(dataset_path: str, *, validate_files: bool) -> DatasetSource:
        p = Path(dataset_path)
        if not p.exists():
            raise FileNotFoundError(dataset_path)
        if p.is_file() and p.suffix.lower() == ".jsonl":
            return JsonlDatasetSource(p, validate_files=validate_files)
        raise ValueError(
            f"Unsupported dataset_path={dataset_path}. Currently only .jsonl is supported."
        )

    @property
    def description(self) -> str:
        return self._description

    def _load_image(self, image_path: str | None) -> Image.Image:
        if image_path is None:
            return Image.new("RGB", (256, 256), (245, 248, 250))
        if self._cache_images and image_path in self._image_cache:
            return self._image_cache[image_path]
        img = Image.open(image_path).convert("RGB")
        if self._cache_images:
            self._image_cache[image_path] = img
        return img

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        # If an explicit seed is provided, reset sampling state so that sampling becomes
        # deterministic starting from the beginning of the sequence for that seed.
        if seed is not None:
            self._cursor = 0
            self._shuffle_order = None
            self._shuffle_pos = 0
            self._shuffle_epoch = 0

        sampler = self._index_samplers.get(self._sampling)
        if sampler is None:
            raise ValueError(
                f"sampling must be one of: {', '.join(sorted(self._index_samplers))}"
            )
        idx = sampler()

        ex = self._source.get(idx)
        self._current = ex
        self._current_index = idx

        obs = Observation(
            image=self._load_image(ex.image_path),
            text=ex.text,
            metadata=ex.metadata or {},
        )
        info = {
            "dataset_path": self._dataset_path,
            "index": idx,
            "oracle_answer": ex.answer,
            "grader": self._grader_name,
        }
        return obs, info

    def _sample_index_sequential(self) -> int:
        idx = self._cursor % len(self._source)
        self._cursor += 1
        return idx

    def _sample_index_shuffle(self) -> int:
        # (Re-)initialize / reshuffle when starting or when the current epoch is exhausted.
        if self._shuffle_order is None or self._shuffle_pos >= len(self._source):
            # Keep deterministic behavior under reset(seed=...) by relying on self.np_random.
            # For subsequent epochs without an explicit seed reset, advance epoch counter and reshuffle.
            if self._shuffle_order is not None:
                self._shuffle_epoch += 1
            self._shuffle_order = list(range(len(self._source)))
            self.np_random.shuffle(self._shuffle_order)
            self._shuffle_pos = 0
        idx = int(self._shuffle_order[self._shuffle_pos])
        self._shuffle_pos += 1
        return idx

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._current is None or self._current_index is None:
            raise RuntimeError("Call reset() before step().")

        ex = self._current
        if ex.answer is None:
            reward = 0.0
            extra = {"correct": False, "reason": "missing_oracle_answer"}
        else:
            reward, extra = self._grader(action, ex.answer)

        obs = Observation(
            image=self._load_image(ex.image_path),
            text=None,
            metadata=ex.metadata or {},
        )
        info = {
            "dataset_path": self._dataset_path,
            "index": self._current_index,
            "oracle_answer": ex.answer,
            "grader": self._grader_name,
            **extra,
        }

        # Single-turn: always terminates; truncation handled by base Env.step()
        return obs, float(reward), True, False, info

    def render(self) -> Image.Image:
        """Return the current example image (or a placeholder)."""
        if self._current is None:
            return Image.new("RGB", (256, 256), (245, 248, 250))
        return self._load_image(self._current.image_path)
