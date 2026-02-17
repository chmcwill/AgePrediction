import os
from types import SimpleNamespace

import pytest

from age_prediction import app as app_module


def test_presign_local_storage_returns_local_url(client_local_storage):
    response = client_local_storage.post(
        "/api/presign",
        json={"filename": "my_face.jpg", "content_type": "image/jpeg"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["key"].startswith("uploads/")
    assert "/api/upload/" in payload["url"]
    assert payload["expires_in"] > 0


def test_presign_missing_filename_returns_400(client_local_storage):
    response = client_local_storage.post("/api/presign", json={"content_type": "image/jpeg"})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "filename_required"


def test_presign_s3_returns_presigned_url(client_s3_storage, monkeypatch):
    called = {}

    def _fake_build_upload_key(filename):
        called["build_upload_key"] = filename
        return "uploads/fake_key.jpg"

    def _fake_presign(bucket, key, content_type, expires_in, region=None):
        called["presign"] = {
            "bucket": bucket,
            "key": key,
            "content_type": content_type,
            "expires_in": expires_in,
            "region": region,
        }
        return "https://example.com/presigned"

    monkeypatch.setattr(app_module.s3_storage, "build_upload_key", _fake_build_upload_key)
    monkeypatch.setattr(app_module.s3_storage, "generate_presigned_put_url", _fake_presign)

    response = client_s3_storage.post(
        "/api/presign",
        json={"filename": "face.png", "content_type": "image/png"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["url"] == "https://example.com/presigned"
    assert payload["key"] == "uploads/fake_key.jpg"
    assert called["build_upload_key"] == "face.png"
    assert called["presign"]["bucket"] == "test-upload-bucket"


def test_local_upload_put_succeeds(client_local_storage, app_local_storage, tmp_path):
    key = "uploads/test_upload.jpg"
    response = client_local_storage.put(f"/api/upload/{key}", data=b"image-bytes")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True

    saved = os.path.join(app_local_storage.config["LOCAL_STORAGE_DIR"], "test_upload.jpg")
    assert os.path.exists(saved)


def test_local_upload_disabled_returns_400(client_s3_storage):
    response = client_s3_storage.put("/api/upload/uploads/test.jpg", data=b"data")
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "local_storage_disabled"


def test_local_upload_invalid_key_returns_400(client_local_storage):
    response = client_local_storage.put("/api/upload/uploads/", data=b"data")
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "invalid_key"


def test_health_deep_triggers_model_warmup(app_local_storage, monkeypatch):
    called = {}

    def _fake_get_runtime_models(classes, min_face_size):
        called["classes"] = classes
        called["min_face_size"] = min_face_size
        return None, None, None, None

    monkeypatch.setattr("age_prediction.services.models.get_runtime_models", _fake_get_runtime_models)
    client = app_local_storage.test_client()
    response = client.get("/api/health?deep=true")
    assert response.status_code == 200
    assert called["classes"] is not None


def test_predict_local_storage_success(client_local_storage, app_local_storage, monkeypatch, tmp_path):
    key = "uploads/test_upload.jpg"
    local_path = os.path.join(app_local_storage.config["LOCAL_STORAGE_DIR"], "test_upload.jpg")
    with open(local_path, "wb") as handle:
        handle.write(b"image-bytes")

    dummy_result = SimpleNamespace(
        big_fig_path=os.path.join(str(tmp_path), "big.jpg"),
        fig_paths=[os.path.join(str(tmp_path), "small.jpg")],
    )

    monkeypatch.setattr(app_module, "storage", app_module.storage)
    monkeypatch.setattr(
        "age_prediction.services.prediction.run_prediction",
        lambda *_args, **_kwargs: (dummy_result, []),
    )

    response = client_local_storage.post("/api/predict", json={"key": key})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["big_fig_url"] is not None
    assert len(payload["fig_urls"]) == 1


def test_predict_local_storage_missing_file_returns_404(client_local_storage):
    response = client_local_storage.post("/api/predict", json={"key": "uploads/missing.jpg"})
    assert response.status_code == 404
    payload = response.get_json()
    assert payload["error"] == "not_found"


def test_predict_missing_key_returns_400(client_local_storage):
    response = client_local_storage.post("/api/predict", json={})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "key_required"


def test_predict_invalid_key_returns_400(client_local_storage):
    response = client_local_storage.post("/api/predict", json={"key": "not_uploads/test.jpg"})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "invalid_key"


def test_predict_s3_success(client_s3_storage, monkeypatch, tmp_path):
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    big_path = generated_dir / "big.jpg"
    fig_path = generated_dir / "small.jpg"
    big_path.write_bytes(b"big")
    fig_path.write_bytes(b"small")

    dummy_result = SimpleNamespace(big_fig_path=str(big_path), fig_paths=[str(fig_path)])

    monkeypatch.setattr(
        "age_prediction.services.prediction.run_prediction",
        lambda *_args, **_kwargs: (dummy_result, [str(big_path), str(fig_path)]),
    )
    monkeypatch.setattr(app_module.s3_storage, "download_to_path", lambda **_kwargs: None)
    monkeypatch.setattr(app_module.s3_storage, "upload_file", lambda **_kwargs: None)
    monkeypatch.setattr(
        app_module.s3_storage,
        "generate_presigned_get_url",
        lambda **_kwargs: "https://example.com/result.jpg",
    )

    response = client_s3_storage.post("/api/predict", json={"key": "uploads/test.jpg"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["big_fig_url"] == "https://example.com/result.jpg"
    assert payload["fig_urls"] == ["https://example.com/result.jpg"]


@pytest.mark.parametrize(
    "exception_cls,expected_error",
    [
        (app_module.InferenceOOMError, "oom"),
        (app_module.InvalidImageError, "invalid_image"),
        (app_module.NoFacesFoundError, "invalid_image"),
        (RuntimeError, "server_error"),
    ],
)
def test_predict_error_mapping(client_local_storage, monkeypatch, exception_cls, expected_error):
    def _raise(*_args, **_kwargs):
        raise exception_cls("boom")

    app = client_local_storage.application
    local_path = os.path.join(app.config["LOCAL_STORAGE_DIR"], "test.jpg")
    with open(local_path, "wb") as handle:
        handle.write(b"image-bytes")

    monkeypatch.setattr("age_prediction.services.prediction.run_prediction", _raise)
    response = client_local_storage.post("/api/predict", json={"key": "uploads/test.jpg"})
    assert response.status_code in (400, 500)
    payload = response.get_json()
    assert payload["error"] == expected_error


def test_api_options_returns_204(client_local_storage):
    response = client_local_storage.open("/api/anything", method="OPTIONS")
    assert response.status_code == 204


def test_cors_headers_added_for_api_routes(client_local_storage):
    response = client_local_storage.get("/api/health")
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == "*"


def test_request_too_large_returns_413(app_local_storage):
    app_local_storage.config["MAX_CONTENT_LENGTH"] = 1
    client = app_local_storage.test_client()
    response = client.put("/api/upload/uploads/test.jpg", data=b"ab")
    assert response.status_code == 413
    payload = response.get_json()
    assert payload["error"] == "file_too_large"
