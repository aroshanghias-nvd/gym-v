"""
Shared image utilities for gym_v (Client-side).
Should NOT depend on torch/cuda.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def _to_uint8(array):
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError("Images must be 2D or 3D arrays.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(arr.max()) if arr.size else 1.0
            min_val = float(arr.min()) if arr.size else 0.0
            if max_val <= 1.0 and min_val >= 0.0:
                arr = (arr * 255.0).round()
            else:
                arr = np.clip(arr, 0.0, 255.0).round()
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


def to_pil_list(images) -> list[Image.Image]:
    """Convert inputs to a list of PIL Images."""
    if isinstance(images, Image.Image):
        return [images if images.mode == "RGB" else images.convert("RGB")]
    if isinstance(images, str | Path):
        with Image.open(images) as img:
            return [img.convert("RGB")]
    if isinstance(images, bytes | bytearray | memoryview):
        with Image.open(io.BytesIO(images)) as img:
            return [img.convert("RGB")]
    if isinstance(images, list | tuple):
        out = []
        for img in images:
            if isinstance(img, Image.Image):
                out.append(img if img.mode == "RGB" else img.convert("RGB"))
            elif isinstance(img, str | Path):
                with Image.open(img) as pil:
                    out.append(pil.convert("RGB"))
            elif isinstance(img, bytes | bytearray | memoryview):
                with Image.open(io.BytesIO(img)) as pil:
                    out.append(pil.convert("RGB"))
            elif isinstance(img, np.ndarray):
                out.append(Image.fromarray(_to_uint8(img)).convert("RGB"))
            # Removed torch support
            else:
                try:
                    # Fallback check for torch tensors without importing torch
                    type_str = str(type(img))
                    if "torch" in type_str and "Tensor" in type_str:
                        # We can't handle it here if we want to be torch-free
                        raise TypeError(
                            "gym_v client does not support torch tensors. Please convert to numpy/PIL."
                        )
                except Exception:
                    pass
                raise TypeError(f"Unsupported element type in list: {type(img)}")
        return out
    if isinstance(images, np.ndarray):
        arr = images
        if arr.ndim == 3:
            arr = arr[None, ...]
        return [Image.fromarray(_to_uint8(frame)).convert("RGB") for frame in arr]

    # Removed torch check
    type_str = str(type(images))
    if "torch" in type_str and "Tensor" in type_str:
        raise TypeError(
            "gym_v client does not support torch tensors. Please convert to numpy/PIL."
        )

    raise TypeError(f"Unsupported image type: {type(images)}")
