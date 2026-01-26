from __future__ import annotations

import asyncio
import base64
from collections.abc import Iterable
from io import BytesIO
import json
import threading
import time
from typing import Any

from PIL import Image
import requests


def ensure_list(value: list[Any] | None, size: int, fill: Any) -> list[Any]:
    if value is None:
        return [fill for _ in range(size)]
    return list(value)


def run_coroutine(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return run_coroutine_in_thread(coro)


def run_coroutine_in_thread(coro):
    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
        except BaseException as exc:
            error.append(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result.get("value")


def decode_data_url(value: str) -> bytes:
    _, _, b64 = value.partition(",")
    return base64.b64decode(b64)


def image_to_data_url(image: Image.Image, *, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def parse_openai_payload(response: dict[str, Any]) -> dict[str, Any]:
    if "error" in response:
        raise RuntimeError(response["error"])
    if "choices" in response:
        content = response["choices"][0]["message"]["content"]
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError("Failed to decode reward payload JSON.") from exc
        if isinstance(content, dict):
            return content
    return response


class RewardClient:
    def __init__(
        self,
        server_url: str,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.endpoint = f"{self.server_url}/v1/chat/completions"
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._session = session or requests.Session()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = max(self.max_retries, 1)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._session.post(
                    self.endpoint, json=payload, timeout=self.timeout_s
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
                return response.json()
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    raise
                sleep_s = self.backoff_factor * (2**attempt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        raise last_exc or RuntimeError("Reward request failed")

    async def request_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self.request, payload)

    async def request_many(
        self,
        payloads: Iterable[dict[str, Any]],
        *,
        concurrency: int | None = None,
    ) -> list[dict[str, Any]]:
        payload_list = list(payloads)
        if not payload_list:
            return []

        if not concurrency or concurrency <= 0:
            tasks = [self.request_async(payload) for payload in payload_list]
            return await asyncio.gather(*tasks)

        semaphore = asyncio.Semaphore(concurrency)

        async def _run(payload: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self.request_async(payload)

        tasks = [_run(payload) for payload in payload_list]
        return await asyncio.gather(*tasks)

    def close(self) -> None:
        self._session.close()
