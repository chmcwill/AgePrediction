import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    import age_prediction.services.models as models

    monkeypatch.setattr(models, "Facenet_Embeddor", lambda *a, **k: _DummyModel())
    monkeypatch.setattr(models, "Ensemble_Model", lambda *a, **k: _DummyModel())
    monkeypatch.setattr(models, "MTCNN", lambda *a, **k: _DummyModel())

    sys.modules.pop("age_prediction.app", None)
    module = importlib.import_module("age_prediction.app")
    return module


@pytest.fixture
def app(app_module):
    return app_module.create_app()


@pytest.fixture
def client(app):
    return app.test_client()
