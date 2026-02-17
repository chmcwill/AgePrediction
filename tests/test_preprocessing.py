from pathlib import Path

import numpy as np
import torch
from PIL import Image

from age_prediction.services import preprocessing as prep


def test_resize_to_max_dim_no_resize():
    img = Image.new("RGB", (100, 80), color="white")
    resized, scale = prep.resize_to_max_dim(img, max_dim=200)
    assert resized.size == (100, 80)
    assert scale == 0.5


def test_resize_to_max_dim_downsizes():
    img = Image.new("RGB", (400, 200), color="white")
    resized, scale = prep.resize_to_max_dim(img, max_dim=200)
    assert resized.size == (200, 100)
    assert scale == 2


def test_load_image_converts_to_rgb(tmp_path):
    path = Path(tmp_path) / "gray.jpg"
    Image.new("L", (20, 20), color=128).save(path)
    img = prep.load_image(str(path), max_dim=100)
    assert img.mode == "RGB"


def test_face_to_tensor_normalizes():
    img = Image.new("RGB", (10, 10), color="white")
    tensor = prep.face_to_tensor(img)
    assert tensor.shape == (3, 10, 10)
    assert torch.max(tensor) <= 1


def test_tensor_to_pil_roundtrip():
    tensor = torch.zeros((3, 8, 8))
    img = prep.tensor_to_pil(tensor)
    assert img.size == (8, 8)


def test_box_to_square_returns_square():
    box = np.array([10, 10, 30, 50], dtype=np.float32)
    square = prep.box_to_square(box)
    w = square[2] - square[0]
    h = square[3] - square[1]
    assert np.isclose(w, h)


def test_box_add_margin_expands_box():
    box = np.array([10, 10, 30, 30], dtype=np.float32)
    expanded = prep.box_add_margin(box.copy(), margin=0.1, img_w=100, img_h=100, clamp=False)
    assert expanded[0] < box[0]
    assert expanded[2] > box[2]
