from __future__ import annotations

import base64
from io import BytesIO
import json
import multiprocessing as mp
import os
import random
import statistics
import time
from typing import Any
import unittest
import urllib.request

from PIL import Image


def _black_png_base64(size: int = 32) -> str:
    image = Image.new("RGB", (size, size), color=(0, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_PNG_BASE64 = _black_png_base64()
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


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _post_json(
    url: str, payload: dict[str, Any], timeout_s: float
) -> tuple[int, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        _ = resp.read()
        code = resp.getcode()
    return code, time.perf_counter() - start


def _build_payload(only_strict: bool, template_index: int) -> dict[str, Any]:
    template = _METADATA_TEMPLATES[template_index % len(_METADATA_TEMPLATES)]
    messages = [{"image": _PNG_BASE64, "prompt": template["prompt"]}]
    metadatas = {
        "meta_datas": [json.loads(json.dumps(template))],
        "only_strict": only_strict,
    }
    return {"model": "geneval", "messages": messages, "metadata": metadatas}


def _worker_run(args: tuple[str, int, bool, float, int]) -> dict[str, Any]:
    url, repeats, only_strict, timeout_s, worker_id = args
    success = 0
    failures = 0
    latencies: list[float] = []
    mock_enabled = _env_flag("MOCK_REWARD_SERVER", False)
    mock_latency_s = float(os.environ.get("MOCK_LATENCY_S", "0"))
    mock_failure_rate = float(os.environ.get("MOCK_FAILURE_RATE", "0"))
    mock_seed = int(os.environ.get("MOCK_SEED", "0"))
    rng = random.Random(mock_seed + worker_id)

    for idx in range(repeats):
        payload = _build_payload(only_strict, worker_id + idx)
        if mock_enabled:
            start = time.perf_counter()
            if mock_latency_s > 0:
                time.sleep(mock_latency_s)
            latency = time.perf_counter() - start
            if mock_failure_rate > 0 and rng.random() < mock_failure_rate:
                failures += 1
            else:
                success += 1
                latencies.append(latency)
            continue
        try:
            code, latency = _post_json(url, payload, timeout_s)
            if code >= 400:
                failures += 1
            else:
                success += 1
                latencies.append(latency)
        except Exception:
            failures += 1

    return {
        "success": success,
        "failures": failures,
        "latencies": latencies,
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


class TestRewardServerConcurrency(unittest.TestCase):
    def setUp(self):
        if os.environ.get("RUN_REWARD_SERVER_CONCURRENCY") != "1":
            self.skipTest("Set RUN_REWARD_SERVER_CONCURRENCY=1 to enable this test.")

        self.url = os.environ.get(
            "REWARD_SERVER_URL", "http://127.0.0.1:18085/v1/chat/completions"
        )
        self.processes = int(os.environ.get("CONCURRENCY_PROCESSES", "4"))
        self.repeats = int(os.environ.get("REQUESTS_PER_PROCESS", "5"))
        self.timeout_s = float(os.environ.get("REQUEST_TIMEOUT_S", "20"))
        self.only_strict = os.environ.get("ONLY_STRICT", "1") != "0"
        self.max_failure_rate = float(os.environ.get("MAX_FAILURE_RATE", "0"))
        self.allow_failure = os.environ.get("ALLOW_FAILURE", "0") == "1"
        self.start_method = os.environ.get("MP_START_METHOD", "spawn")

    def test_reward_server_concurrency(self):
        ctx = mp.get_context(self.start_method)
        args = [
            (
                self.url,
                self.repeats,
                self.only_strict,
                self.timeout_s,
                worker_id,
            )
            for worker_id in range(self.processes)
        ]

        start = time.perf_counter()
        with ctx.Pool(processes=self.processes) as pool:
            results = pool.map(_worker_run, args)
        elapsed = time.perf_counter() - start

        total_success = sum(r["success"] for r in results)
        total_failures = sum(r["failures"] for r in results)
        total_requests = total_success + total_failures
        latencies = [lat for r in results for lat in r["latencies"]]

        print(
            f"[reward-server] url={self.url} processes={self.processes} "
            f"repeats={self.repeats} "
            f"success={total_success} failures={total_failures} "
            f"elapsed={elapsed:.2f}s"
        )
        if latencies:
            p50 = _percentile(latencies, 50)
            p95 = _percentile(latencies, 95)
            avg = statistics.mean(latencies)
            if p50 is not None and p95 is not None:
                print(
                    f"[reward-server] latency_s avg={avg:.3f} p50={p50:.3f} p95={p95:.3f}"
                )

        if total_requests == 0:
            self.fail("No requests were issued.")

        if not self.allow_failure:
            failure_rate = total_failures / total_requests
            self.assertLessEqual(
                failure_rate,
                self.max_failure_rate,
                f"Failure rate {failure_rate:.2%} exceeds max {self.max_failure_rate:.2%}",
            )


if __name__ == "__main__":
    unittest.main()
