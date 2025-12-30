"""Built-in graders for offline environments."""

from __future__ import annotations

import re
from typing import Any


def _normalize_em(s: str) -> str:
    # Simple EM normalization: strip, collapse whitespace, lowercase.
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(action: str, answer: str) -> tuple[float, dict[str, Any]]:
    """Exact match reward: 1.0 if normalized strings match, else 0.0."""
    pred = _normalize_em(action)
    gt = _normalize_em(answer)
    correct = pred == gt
    return (1.0 if correct else 0.0), {"correct": correct, "pred": action, "gt": answer}


GRADERS: dict[str, Any] = {
    "exact_match": exact_match,
}
