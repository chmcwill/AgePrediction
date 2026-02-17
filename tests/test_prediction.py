import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from age_prediction.services import prediction as pred


class _DummyDetector:
    def __init__(self, boxes, probs, landmarks=None):
        self._boxes = boxes
        self._probs = probs
        self._landmarks = landmarks

    def detect(self, _img, landmarks=True):
        return self._boxes, self._probs, self._landmarks if landmarks else None


class _DummyEmbeddor:
    def forward_1792(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.shape[0], 1792), dtype=torch.float32)


class _DummyModel:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 61), dtype=torch.float32)


def _make_test_image(tmp_path: Path) -> str:
    img = Image.new("RGB", (400, 400), color="white")
    path = tmp_path / "test.jpg"
    img.save(path)
    return str(path)


def test_image_to_context_loads_image(tmp_path):
    image_path = _make_test_image(tmp_path)
    ctx = pred._image_to_context(image_path, pred.DEFAULT_PREDICTION_CONFIG)
    assert ctx.filename == "test.jpg"
    assert ctx.width > 0
    assert ctx.height > 0


def test_image_to_context_invalid_path_raises():
    with pytest.raises(pred.InvalidImageError):
        pred._image_to_context("does_not_exist.jpg", pred.DEFAULT_PREDICTION_CONFIG)


def test_detect_faces_raises_when_no_boxes(tmp_path):
    image_path = _make_test_image(tmp_path)
    ctx = pred._image_to_context(image_path, pred.DEFAULT_PREDICTION_CONFIG)
    detector = _DummyDetector(None, None, None)
    with pytest.raises(pred.NoFacesFoundError):
        pred.detect_faces(ctx, detector, pred.DEFAULT_PREDICTION_CONFIG)


def test_safe_detect_scales_boxes(tmp_path):
    img = Image.new("RGB", (2000, 2000), color="white")
    detector = _DummyDetector(
        boxes=np.array([[0, 0, 100, 100]], dtype=np.float32),
        probs=np.array([0.99], dtype=np.float32),
        landmarks=np.array([[[10, 10], [20, 10], [15, 15]]], dtype=np.float32),
    )
    boxes, probs, landmarks = pred.safe_detect(detector, img, max_size=1000)
    assert np.allclose(boxes, np.array([[0, 0, 200, 200]], dtype=np.float32))
    assert probs is not None
    assert landmarks is not None


def test_front_facing_checks_nose_position():
    landmarks = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
    face = pred.FaceDetection(
        box=np.zeros(4, dtype=np.float32),
        prob=0.99,
        landmarks=landmarks,
        original_box=np.zeros(4, dtype=np.float32),
    )
    assert pred._front_facing(face) == False


def test_apply_face_filters_sets_reasons():
    cfg = pred.DEFAULT_PREDICTION_CONFIG
    img_ctx = pred.ImageContext(
        image=Image.new("RGB", (200, 200), color="white"),
        tensor=torch.zeros((3, 200, 200)),
        width=200,
        height=200,
        filename="test.jpg",
    )
    low_prob_face = pred.FaceDetection(
        box=np.array([10, 10, 100, 100], dtype=np.float32),
        prob=0.5,
        landmarks=None,
        original_box=np.array([10, 10, 100, 100], dtype=np.float32),
    )
    small_face = pred.FaceDetection(
        box=np.array([10, 10, 20, 20], dtype=np.float32),
        prob=0.99,
        landmarks=None,
        original_box=np.array([10, 10, 20, 20], dtype=np.float32),
    )
    img_ctx.faces = [low_prob_face, small_face]
    pred._apply_face_filters(img_ctx, cfg)
    assert low_prob_face.reason == pred.FilterReason.INSUFFICIENT_CONFIDENCE
    assert small_face.reason == pred.FilterReason.INSUFFICIENT_RESOLUTION


def test_crop_face_with_padding_shapes_tensor():
    tensor = torch.zeros((3, 100, 100))
    box = np.array([-10, -10, 50, 50], dtype=np.float32)
    crop = pred._crop_face_with_padding(tensor, box, resize_shape=64)
    assert crop.shape == (3, 64, 64)


def test_rotate_face_tensor_preserves_shape():
    tensor = torch.zeros((3, 100, 100))
    face = pred.FaceDetection(
        box=np.array([10, 10, 50, 50], dtype=np.float32),
        prob=0.99,
        landmarks=np.array([[10, 10], [20, 12]], dtype=np.float32),
        original_box=np.array([10, 10, 50, 50], dtype=np.float32),
    )
    rotated = pred._rotate_face_tensor(tensor, face)
    assert rotated.shape == tensor.shape


def test_predict_faces_generates_figs():
    cfg = pred.DEFAULT_PREDICTION_CONFIG
    img = Image.new("RGB", (200, 200), color="white")
    img_ctx = pred.ImageContext(
        image=img,
        tensor=torch.zeros((3, 200, 200)),
        width=200,
        height=200,
        filename="test.jpg",
    )
    face = pred.FaceDetection(
        box=np.array([10, 10, 150, 150], dtype=np.float32),
        prob=0.99,
        landmarks=None,
        original_box=np.array([10, 10, 150, 150], dtype=np.float32),
    )
    face.crop = torch.zeros((3, cfg.resize_shape, cfg.resize_shape))
    img_ctx.faces = [face]

    figs, big_fig = pred.predict_faces(
        img_ctx,
        kept_faces=[face],
        embeddor=_DummyEmbeddor(),
        model=_DummyModel(),
        device=torch.device("cpu"),
        config=cfg,
    )
    assert len(figs) == 1
    assert big_fig is not None


def test_render_figures_saves_files(tmp_path):
    fig = pred.fpl.plot_image_and_pred(
        pred.fpl.FacePlotData(
            image=Image.new("RGB", (160, 160), color="white"),
            prediction=25.0,
            output_softmax=np.ones((1, 61), dtype=np.float32) / 61.0,
        )
    )
    big_fig, _ = pred.fpl.overlay_preds_on_img(
        Image.new("RGB", (200, 200), color="white"),
        overlays=[pred.fpl.OverlayItem(bounds=np.array([10, 10, 50, 50]), label=np.array([25]))],
    )
    result, generated = pred.render_figures(
        [fig],
        big_fig,
        str(tmp_path),
        "test",
        pred.DEFAULT_PREDICTION_CONFIG,
    )
    assert result.big_fig_path is not None
    assert len(result.fig_paths) == 1
    for path in generated:
        assert os.path.exists(path)


def test_run_prediction_with_dummy_models(test_image_path, tmp_path, monkeypatch):
    detector = _DummyDetector(
        boxes=np.array([[50, 50, 200, 200]], dtype=np.float32),
        probs=np.array([0.99], dtype=np.float32),
        landmarks=None,
    )

    def _fake_get_runtime_models(classes, min_face_size):
        return detector, _DummyEmbeddor(), _DummyModel(), torch.device("cpu")

    monkeypatch.setattr("age_prediction.services.prediction.get_runtime_models", _fake_get_runtime_models)
    result, generated = pred.run_prediction(test_image_path, str(tmp_path))
    assert result.big_fig_path is not None
    assert len(result.fig_paths) == 1
    for path in generated:
        assert os.path.exists(path)
