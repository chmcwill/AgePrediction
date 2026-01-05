import os
import time
from pathlib import Path

import pytest

from age_prediction.services import storage
from age_prediction.services.errors import StorageError


class _DummyUpload:
    def __init__(self, filename: str, mimetype: str, content: bytes = b"image-bytes"):
        self.filename = filename
        self.mimetype = mimetype
        self._content = content

    def save(self, destination):
        Path(destination).write_bytes(self._content)


def test_save_upload_generates_unique_secure_name(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload = _DummyUpload("My Photo.JPG", "image/jpeg")

    dest_path = storage.save_upload(upload, str(upload_dir))

    saved = Path(dest_path)
    assert saved.exists()
    assert saved.parent == upload_dir

    hex_part, rest = saved.name.split("_", 1)
    assert len(hex_part) == 32  # uuid hex
    assert rest == "My_Photo.jpg"  # secure_filename with lowered extension


@pytest.mark.parametrize(
    "filename,mimetype",
    [
        ("script.exe", "application/octet-stream"),
        ("face.png", "text/plain"),  # wrong mimetype
    ],
)
def test_save_upload_rejects_disallowed_types(tmp_path, filename, mimetype):
    upload_dir = tmp_path / "uploads"
    upload = _DummyUpload(filename, mimetype)

    with pytest.raises(StorageError):
        storage.save_upload(upload, str(upload_dir))


def test_save_upload_requires_filename(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload = _DummyUpload("", "image/jpeg")

    with pytest.raises(StorageError):
        storage.save_upload(upload, str(upload_dir))


def test_cleanup_files_is_best_effort(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    # Should not raise even if file is missing
    storage.cleanup_files([str(missing)])


def test_cleanup_stale_files_removes_old_and_keeps_recent(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    old_file = upload_dir / "old.jpg"
    new_file = upload_dir / "new.jpg"
    old_file.write_bytes(b"a" * 10)
    new_file.write_bytes(b"a" * 10)

    now = time.time()
    max_age = 60
    os.utime(old_file, (now - max_age - 10, now - max_age - 10))  # older than threshold
    os.utime(new_file, (now, now))  # fresh

    deleted_count, _ = storage.cleanup_stale_files(str(upload_dir), max_age_seconds=max_age)

    assert deleted_count == 1
    assert not old_file.exists()
    assert new_file.exists()


def test_cleanup_stale_files_respects_size_cap_and_excludes(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    keep_file = upload_dir / "keep.jpg"
    old_file = upload_dir / "old.jpg"
    mid_file = upload_dir / "mid.jpg"

    keep_file.write_bytes(b"k" * 4000)
    old_file.write_bytes(b"o" * 5000)
    mid_file.write_bytes(b"m" * 4000)

    now = time.time()
    os.utime(old_file, (now - 300, now - 300))
    os.utime(mid_file, (now - 200, now - 200))
    os.utime(keep_file, (now - 100, now - 100))

    cap_bytes = 8000  # force deletion of oldest (old_file), leave mid_file
    deleted_count, deleted_bytes = storage.cleanup_stale_files(
        str(upload_dir),
        max_age_seconds=10_000,  # age threshold won't delete anything
        max_total_bytes=cap_bytes,
        exclude=[str(keep_file)],
    )

    assert deleted_count == 1
    assert deleted_bytes == 5000
    assert not old_file.exists()
    assert mid_file.exists()
    assert keep_file.exists()
