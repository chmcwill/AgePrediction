# -*- coding: utf-8 -*-
"""
Lightweight preprocessing helpers shared across inference and plotting.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


def resize_to_max_dim(img: Image.Image, max_dim: int) -> Tuple[Image.Image, float]:
    """
    Resize an image so the largest dimension is <= max_dim.
    Returns the resized image and the scale that was applied.
    """
    w, h = img.size
    scale = max(w, h) / max_dim
    if scale > 1:
        new_w, new_h = int(w / scale), int(h / scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img, scale


def load_image(image_path: str, max_dim: int) -> Image.Image:
    """
    Load an image from disk, enforce RGB, fix EXIF orientation, and resize.
    """
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img, _ = resize_to_max_dim(img, max_dim)
        return img.copy()


def face_to_tensor(face_img: Image.Image) -> torch.Tensor:
    """Convert PIL face image to normalized tensor in [-1, 1]."""
    np_array = (np.asarray(face_img).astype(np.float32) - 127.5) / 128
    return torch.as_tensor(np_array.transpose(2, 0, 1), dtype=torch.float32)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert normalized tensor back to PIL image for plotting."""
    if torch.max(tensor) <= 1:
        tensor = (tensor + 1) * (255 / 2)
    array = tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(array)


def box_to_square(bounding_box):
    """
    Convert an axis-aligned bounding box to a square by expanding the smaller side.
    """
    x1, y1, x2, y2 = bounding_box
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0

    max_side = np.max((h, w))

    bounding_box[0] = np.round(x1 + w * 0.5 - max_side * 0.5)
    bounding_box[1] = np.round(y1 + h * 0.5 - max_side * 0.5)

    addition = max_side - 1.0
    if addition % 0.5 == 0:
        addition += 0.00001

    bounding_box[2] = bounding_box[0] + np.round(addition)
    bounding_box[3] = bounding_box[1] + np.round(addition)

    return np.array(bounding_box, dtype=np.float32)


def box_add_margin(face_bounds, margin, img_w, img_h, clamp=False):
    """
    Expand a square face bounding box by a percentage margin.
    """
    if margin > 1:
        margin = margin / 100

    w = face_bounds[2] - face_bounds[0]
    h = face_bounds[3] - face_bounds[1]
    assert w == h, "Box must be square before applying margin."

    face_bounds[0] = face_bounds[0] - margin * w
    face_bounds[1] = face_bounds[1] - margin * h
    face_bounds[2] = face_bounds[2] + margin * w
    face_bounds[3] = face_bounds[3] + margin * h

    if clamp:
        face_bounds[0] = np.max((0, face_bounds[0]))
        face_bounds[1] = np.max((0, face_bounds[1]))
        face_bounds[2] = np.min((img_w, face_bounds[2]))
        face_bounds[3] = np.min((img_h, face_bounds[3]))

    return face_bounds


__all__ = [
    "box_add_margin",
    "box_to_square",
    "face_to_tensor",
    "load_image",
    "resize_to_max_dim",
    "tensor_to_pil",
]
