"""Offline dataset sources (JSONL, in-memory, etc.)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class OfflineSample:
    """A single-turn offline example.

    Expected semantics:
    - text: textual prompt/context (optional but typical)
    - image_path: path to an image file on disk (optional)
    - answer: oracle / ground-truth answer (optional for some tasks)
    - metadata: arbitrary extra information
    """

    text: str | None = None
    image_path: str | None = None
    answer: str | None = None
    metadata: dict[str, Any] | None = None


class DatasetSource(Protocol):
    """A random-access collection of offline samples."""

    def __len__(self) -> int: ...

    def get(self, index: int) -> OfflineSample: ...

    def iter(self) -> Iterable[OfflineSample]: ...


class JsonlDatasetSource:
    """Loads newline-delimited JSON examples into memory.

    JSONL schema (per line):
    - text: str (optional)
    - image_path: str (optional) - relative paths are resolved relative to the jsonl file
    - answer: str (optional)
    - metadata: dict (optional)
    """

    def __init__(self, path: str | Path, *, validate_files: bool = True):
        self._path = Path(path)
        self._base_dir = self._path.parent
        self._samples: list[OfflineSample] = []
        self._validate_files = validate_files

        with self._path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"JSONL line {line_no} must be an object, got {type(obj)}"
                    )

                text = obj.get("text")
                image_path = obj.get("image_path")
                answer = obj.get("answer")
                metadata = obj.get("metadata")

                if text is not None and not isinstance(text, str):
                    raise ValueError(
                        f"JSONL line {line_no} `text` must be str or null, got {type(text)}"
                    )
                if image_path is not None and not isinstance(image_path, str):
                    raise ValueError(
                        f"JSONL line {line_no} `image_path` must be str or null, got {type(image_path)}"
                    )
                if answer is not None and not isinstance(answer, str):
                    raise ValueError(
                        f"JSONL line {line_no} `answer` must be str or null, got {type(answer)}"
                    )
                if metadata is not None and not isinstance(metadata, dict):
                    raise ValueError(
                        f"JSONL line {line_no} `metadata` must be dict or null, got {type(metadata)}"
                    )

                if text is None and image_path is None:
                    raise ValueError(
                        f"JSONL line {line_no} must have at least one of `text` or `image_path`."
                    )

                if image_path is not None:
                    p = Path(image_path)
                    if not p.is_absolute():
                        image_path = str((self._base_dir / p).resolve())
                    if self._validate_files and not Path(image_path).exists():
                        raise FileNotFoundError(
                            f"JSONL line {line_no} image_path not found: {image_path}"
                        )

                self._samples.append(
                    OfflineSample(
                        text=text,
                        image_path=image_path,
                        answer=answer,
                        metadata=metadata,
                    )
                )

        if len(self._samples) == 0:
            raise ValueError(f"Empty JSONL dataset: {self._path}")

    def __len__(self) -> int:
        return len(self._samples)

    def get(self, index: int) -> OfflineSample:
        return self._samples[index]

    def iter(self) -> Iterable[OfflineSample]:
        return iter(self._samples)
