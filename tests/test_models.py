import torch

from age_prediction.services import models


def test_predict_age_returns_expected_values():
    logits = torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32)
    classes = [10, 11, 12]
    pred, softmax = models.predict_age(logits, classes, thresh=0.0)
    assert pred.shape == (1,)
    assert softmax.shape == (1, 3)
    assert pred[0] == 12.0


def test_get_runtime_models_uses_cache(monkeypatch):
    created = {"count": 0}

    class _DummyMTCNN:
        def __init__(self, *args, **kwargs):
            created["count"] += 1

    class _DummyEmbeddor:
        def __init__(self, *args, **kwargs):
            created["count"] += 1

        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            return self

    class _DummyEnsemble:
        def __init__(self, *args, **kwargs):
            created["count"] += 1

        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(models, "MTCNN", _DummyMTCNN)
    monkeypatch.setattr(models, "Facenet_Embeddor", _DummyEmbeddor)
    monkeypatch.setattr(models, "Ensemble_Model", _DummyEnsemble)

    first = models.get_runtime_models((10, 11), 20)
    second = models.get_runtime_models((10, 11), 20)

    assert first is second
    assert created["count"] == 3
    models.get_runtime_models.cache_clear()
