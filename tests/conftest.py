import importlib
import sys

import pytest


class _DummyModel:
    def eval(self):
        return self

    def to(self, *_args, **_kwargs):
        return self


@pytest.fixture(scope="function")
def app_module(monkeypatch):
    """
    Load FlaskAgePred with heavy models stubbed so smoke tests stay fast and
    avoid large weight downloads.
    """
    import FaceModels
    import facenet_pytorch

    monkeypatch.setattr(FaceModels, "Facenet_Embeddor", lambda *a, **k: _DummyModel())
    monkeypatch.setattr(FaceModels, "Ensemble_Model", lambda *a, **k: _DummyModel())
    monkeypatch.setattr(facenet_pytorch, "MTCNN", lambda *a, **k: _DummyModel())

    sys.modules.pop("age_prediction.app", None)
    module = importlib.import_module("age_prediction.app")
    return module


@pytest.fixture
def app(app_module):
    return app_module.create_app()


@pytest.fixture
def client(app):
    return app.test_client()
