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


@pytest.fixture
def fixtures_dir():
    return os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def test_image_path(fixtures_dir):
    return os.path.join(fixtures_dir, "test_image.jpg")


@pytest.fixture
def app_local_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE", "true")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    from age_prediction.app import create_app

    app = create_app()
    app.config["LOCAL_STORAGE_DIR"] = str(tmp_path)
    return app


@pytest.fixture
def client_local_storage(app_local_storage):
    return app_local_storage.test_client()


@pytest.fixture
def app_s3_storage(monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE", "false")
    monkeypatch.setenv("S3_UPLOAD_BUCKET", "test-upload-bucket")
    monkeypatch.setenv("S3_RESULTS_BUCKET", "test-results-bucket")
    monkeypatch.setenv("S3_REGION", "us-east-2")
    from age_prediction.app import create_app

    return create_app()


@pytest.fixture
def client_s3_storage(app_s3_storage):
    return app_s3_storage.test_client()
