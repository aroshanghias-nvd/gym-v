from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
import os
import time
from typing import Any
import urllib.request
import uuid

from fastapi import FastAPI, HTTPException
from PIL import Image
from ray import serve

from gym_v.logger import get_logger
from services.rewards import multi_score

logger = get_logger()


@dataclass
class _ChatRequest:
    index: int
    model: str
    images: list[Any]
    prompts: list[str]
    metadata: Any

    def append_to_batch(
        self,
        flat_images: list[Any],
        flat_prompts: list[str],
        offsets: list[tuple[int, int, int, str]],
        cursor: int,
    ) -> int:
        count = len(self.images)
        flat_images.extend(self.images)
        flat_prompts.extend(self.prompts)
        offsets.append((self.index, cursor, cursor + count, self.model))
        return cursor + count


@dataclass
class _ChatBatch:
    requests: list[_ChatRequest]
    flat_images: list[Any]
    flat_prompts: list[str]
    offsets: list[tuple[int, int, int, str]]
    merged_metadata: dict[str, Any] | None

    @classmethod
    def from_requests(cls, requests: list[_ChatRequest]) -> _ChatBatch:
        flat_images: list[Any] = []
        flat_prompts: list[str] = []
        offsets: list[tuple[int, int, int, str]] = []
        cursor = 0
        for req in requests:
            cursor = req.append_to_batch(flat_images, flat_prompts, offsets, cursor)
        merged_metadata = _merge_metadata(requests)
        return cls(
            requests=requests,
            flat_images=flat_images,
            flat_prompts=flat_prompts,
            offsets=offsets,
            merged_metadata=merged_metadata,
        )

    @classmethod
    def from_payloads(
        cls, payloads: list[Any], outputs: list[dict[str, Any]]
    ) -> _ChatBatch | None:
        normalized: list[_ChatRequest] = []
        for idx, payload in enumerate(payloads):
            try:
                normalized.append(_parse_chat_request(payload, idx))
            except Exception as exc:
                outputs[idx] = {"error": str(exc)}
        if not normalized:
            return None
        return cls.from_requests(normalized)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _decode_base64(value: str) -> bytes:
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    return base64.b64decode(value)


def _looks_like_base64(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 16 or len(stripped) % 4 != 0:
        return False
    for ch in stripped:
        if ch.isalnum() or ch in "+/=":
            continue
        return False
    return True


def _is_image_header(data: bytes) -> bool:
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return True
    if data.startswith(b"RIFF") and b"WEBP" in data[8:16]:
        return True
    return False


def _maybe_decode_base64(value: str) -> bytes | None:
    if value.startswith("data:"):
        return _decode_base64(value)
    if not _looks_like_base64(value):
        return None
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception:
        return None
    return decoded if _is_image_header(decoded) else None


def _fetch_url(url: str, *, timeout_s: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read()


def _read_file(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _load_images_from_urls(image_urls: list[str]) -> list[Image.Image]:
    """Load images from data URLs, raw base64 strings, HTTP(S) URLs, or local paths."""
    if not image_urls:
        return []

    images: list[Image.Image] = []
    for idx, url in enumerate(image_urls):
        try:
            if url.startswith("data:image/"):
                log_desc = f"data URI (length: {len(url)})"
                logger.info("Loading image %d from %s", idx + 1, log_desc)
                _, _, base64_data = url.partition(",")
                img_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(img_data)).convert("RGB")
            elif url.startswith(("http://", "https://")):
                log_desc = f"URL: {url[:80]}{'...' if len(url) > 80 else ''}"
                logger.info("Downloading image %d from %s", idx + 1, log_desc)
                img_data = _fetch_url(url)
                image = Image.open(BytesIO(img_data)).convert("RGB")
            else:
                decoded = _maybe_decode_base64(url)
                if decoded is not None:
                    log_desc = f"raw base64 string (length: {len(url)})"
                    logger.info("Loading image %d from %s", idx + 1, log_desc)
                    image = Image.open(BytesIO(decoded)).convert("RGB")
                else:
                    path = os.path.expanduser(url)
                    log_desc = f"local path: {path}"
                    logger.info("Loading image %d from %s", idx + 1, log_desc)
                    image = Image.open(path).convert("RGB")
            images.append(image)
            logger.info("Image %d loaded successfully: %s", idx + 1, image.size)
        except Exception as exc:
            error_url = f"<data of length {len(url)}>" if len(url) > 100 else url
            logger.error("Failed to load image %d from %s: %s", idx + 1, error_url, exc)
            raise RuntimeError(f"Failed to load image {idx + 1}: {exc}") from exc

    return images


def _normalize_images(images: Any) -> list[Any]:
    if isinstance(images, list | tuple):
        if not images:
            return []
        if not all(isinstance(item, str) for item in images):
            raise TypeError("images must be a string or list of strings.")
        return _load_images_from_urls(list(images))
    if isinstance(images, str):
        return _load_images_from_urls([images])
    raise TypeError("images must be a string or list of strings.")


def _bytes_to_pil(data: bytes) -> Image.Image | None:
    try:
        with Image.open(BytesIO(data)) as img:
            return img.copy()
    except Exception:
        return None


def _normalize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_metadata(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_metadata(item) for item in value)
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        image = _bytes_to_pil(bytes(value))
        return image if image is not None else value
    if isinstance(value, str):
        if value.startswith("data:"):
            decoded = _decode_base64(value)
            image = _bytes_to_pil(decoded)
            return image if image is not None else value
        decoded = _maybe_decode_base64(value)
        if decoded is not None:
            image = _bytes_to_pil(decoded)
            return image if image is not None else value
        path = os.path.expanduser(value)
        if os.path.isfile(path):
            data = _read_file(path)
            image = _bytes_to_pil(data)
            return image if image is not None else value
        return value
    return value


def _parse_chat_request(payload: Any, index: int) -> _ChatRequest:
    if not isinstance(payload, dict):
        raise TypeError(f"Request payload must be dict, got {type(payload)}.")
    messages = payload.get("messages")
    images_raw, prompts = _extract_message_items(messages)
    if not images_raw:
        raise ValueError("No images found in messages.")

    images = _normalize_images(images_raw)

    metadata = payload.get("metadata")
    if metadata is not None:
        metadata = _normalize_metadata(metadata)

    model = str(payload.get("model", "reward"))
    return _ChatRequest(
        index=index,
        model=model,
        images=images,
        prompts=prompts,
        metadata=metadata,
    )


def _merge_metadata(requests: list[_ChatRequest]) -> dict[str, Any] | None:
    items = [req.metadata for req in requests if isinstance(req.metadata, dict)]
    if not items:
        return None
    if len(items) == 1:
        return items[0]

    def _clone(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _clone(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_clone(item) for item in value]
        if isinstance(value, tuple):
            return [_clone(item) for item in value]
        return value

    merged: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if key == "meta_datas":
                merged.setdefault(key, [])
                if isinstance(value, list | tuple):
                    merged[key].extend(_clone(list(value)))
                else:
                    merged[key].append(_clone(value))
                continue
            if isinstance(value, list | tuple | dict):
                merged.setdefault(key, []).append(_clone(value))
            elif key not in merged:
                merged[key] = value

    return merged


def _slice_scores(scores: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, values in scores.items():
        if isinstance(values, list):
            sliced[key] = values[start:end]
        else:
            sliced[key] = values
    return sliced


def _materialize_scores(scores: dict[str, Any]) -> dict[str, Any]:
    materialized: dict[str, Any] = {}
    for key, values in scores.items():
        materialized[key] = values
    return materialized


def _extract_openai_content(content: Any) -> tuple[list[Any], list[str]] | None:
    if not isinstance(content, list):
        return None

    images: list[Any] = []
    texts: list[str] = []

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"text", "input_text"}:
            text = item.get("text")
            if text is None:
                continue
            texts.append(str(text))
            continue
        if item_type in {"image_url", "input_image"}:
            image_value = item.get("image_url", item.get("image"))
            if isinstance(image_value, dict):
                image_value = image_value.get("url") or image_value.get("image_url")
            if image_value is None:
                raise ValueError("image_url item is missing url.")
            images.append(image_value)

    if not images:
        return None
    prompt = " ".join(texts).strip()
    prompts = [prompt for _ in images]
    return images, prompts


def _extract_message_items(messages: Any) -> tuple[list[Any], list[str]]:
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages)}")

    images: list[Any] = []
    prompts: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        if "content" in message:
            extracted = _extract_openai_content(message.get("content"))
            if extracted is not None:
                content_images, content_prompts = extracted
                images.extend(content_images)
                prompts.extend(content_prompts)
                continue
        if "image" not in message and "images" not in message:
            continue

        image_value = message.get("images", message.get("image"))
        if image_value is None:
            raise ValueError("message is missing image")

        prompt = message.get("prompt")
        if prompt is None:
            prompt = message.get("text")
        if prompt is None and isinstance(message.get("content"), str):
            prompt = message.get("content")
        prompt_text = "" if prompt is None else str(prompt)

        for image in _as_list(image_value):
            images.append(image)
            prompts.append(prompt_text)

    return images, prompts


def _openai_response(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"reward-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(payload)},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def build_reward_service(
    *,
    max_batch_size: int = 32,
    batch_wait_timeout_s: float = 0.01,
    max_ongoing_requests: int = 5120,
):
    app = FastAPI()

    @serve.deployment(max_ongoing_requests=max_ongoing_requests)
    @serve.ingress(app)
    class RewardService:
        def __init__(self, score_dict: dict[str, Any], device: str = "cuda"):
            self.scorer = multi_score(device, score_dict)

        @app.post("/v1/chat/completions")
        async def chat(self, payload: dict[str, Any]):
            result = await self.chat_batch(payload)
            if isinstance(result, dict) and "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            return result

        @serve.batch(
            max_batch_size=max_batch_size, batch_wait_timeout_s=batch_wait_timeout_s
        )
        async def chat_batch(self, requests: list[dict[str, Any]]):
            outputs: list[dict[str, Any]] = [{} for _ in requests]
            batch = _ChatBatch.from_payloads(requests, outputs)
            if batch is None:
                return outputs
            try:
                scores, _ = self.scorer(
                    batch.flat_images,
                    prompts=batch.flat_prompts,
                    metadata=batch.merged_metadata,
                )
                scores = _materialize_scores(scores)
                for index, start, end, model in batch.offsets:
                    payload = _slice_scores(scores, start, end)
                    outputs[index] = _openai_response(model, payload)
            except Exception as exc:
                for req in batch.requests:
                    outputs[req.index] = {"error": str(exc)}

            return outputs

    return RewardService
