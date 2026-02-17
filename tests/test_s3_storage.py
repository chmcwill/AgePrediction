from age_prediction.services import s3_storage


class _DummyClient:
    def __init__(self):
        self.calls = []

    def generate_presigned_url(self, operation, Params=None, ExpiresIn=None):
        self.calls.append((operation, Params, ExpiresIn))
        return "https://example.com/presigned"

    def download_file(self, bucket, key, path):
        self.calls.append(("download", bucket, key, path))

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.calls.append(("upload", path, bucket, key, ExtraArgs))


def test_safe_basename_returns_default():
    assert s3_storage._safe_basename("") == "upload"


def test_build_upload_key_prefix():
    key = s3_storage.build_upload_key("face.jpg")
    assert key.startswith("uploads/")
    assert key.endswith("face.jpg")


def test_build_results_prefix():
    assert s3_storage.build_results_prefix("abc123") == "results/abc123"


def test_generate_presigned_put_url(monkeypatch):
    client = _DummyClient()
    monkeypatch.setattr(s3_storage, "_get_s3_client", lambda region=None: client)
    url = s3_storage.generate_presigned_put_url(
        bucket="bucket",
        key="uploads/test.jpg",
        content_type="image/jpeg",
        expires_in=123,
        region="us-east-2",
    )
    assert url == "https://example.com/presigned"
    assert client.calls[0][0] == "put_object"


def test_generate_presigned_get_url(monkeypatch):
    client = _DummyClient()
    monkeypatch.setattr(s3_storage, "_get_s3_client", lambda region=None: client)
    url = s3_storage.generate_presigned_get_url(
        bucket="bucket",
        key="results/test.jpg",
        expires_in=123,
        region="us-east-2",
    )
    assert url == "https://example.com/presigned"
    assert client.calls[0][0] == "get_object"


def test_download_and_upload(monkeypatch):
    client = _DummyClient()
    monkeypatch.setattr(s3_storage, "_get_s3_client", lambda region=None: client)
    s3_storage.download_to_path("bucket", "key", "path", region="us-east-2")
    s3_storage.upload_file("path", "bucket", "key", content_type="image/jpeg", region="us-east-2")
    assert client.calls[0][0] == "download"
    assert client.calls[1][0] == "upload"
