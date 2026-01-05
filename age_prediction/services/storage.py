# -*- coding: utf-8 -*-
"""
Helpers for storing and cleaning up uploaded and generated files.
"""
import os
import time
import uuid
from typing import Iterable, Optional

from werkzeug.utils import secure_filename

from age_prediction.services.errors import StorageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
ALLOWED_MIME_PREFIXES = ("image/",)


def _validate_upload(uploaded_file) -> None:
    """Validate presence, filename, and MIME/extension allowlist."""
    if uploaded_file is None:
        raise StorageError("No file provided")

    original_name = uploaded_file.filename or ""
    if original_name.strip() == "":
        raise StorageError("No filename provided")

    _, ext = os.path.splitext(original_name)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise StorageError(f"Unsupported file type: {ext or 'unknown'}")

    mimetype = getattr(uploaded_file, "mimetype", "") or ""
    if not any(mimetype.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES):
        raise StorageError("Uploaded file is not a recognized image type")


def _build_destination(upload_dir: str, filename: str) -> str:
    """Create a safe, unique destination path for the upload."""
    safe_name = secure_filename(filename or "")
    base, ext = os.path.splitext(safe_name)

    # If secure_filename strips everything, fall back to a neutral base.
    if base == "":
        base = "upload"
    ext = ext.lower()

    unique_prefix = uuid.uuid4().hex
    new_name = f"{unique_prefix}_{base}{ext}"
    return os.path.join(upload_dir, new_name)


def save_upload(uploaded_file, upload_dir):
    """
    Save an uploaded file to the upload directory with validation and unique naming.
    """
    _validate_upload(uploaded_file)

    os.makedirs(upload_dir, exist_ok=True)
    dest_path = _build_destination(upload_dir, uploaded_file.filename)
    uploaded_file.save(dest_path)

    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        # Cleanup if save failed silently
        cleanup_files([dest_path])
        raise StorageError("Failed to save uploaded file")

    return dest_path


def cleanup_files(paths):
    """
    Best-effort removal of files.
    """
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            # Ignore failures; cleanup is best-effort.
            continue


def _enforce_size_cap(upload_dir: str, max_total_bytes: int, exclude_set: set[str]) -> tuple[int, int]:
    """
    Delete oldest files until total size <= cap. Returns (deleted_count, deleted_bytes).
    """
    deleted_count = 0
    deleted_bytes = 0

    remaining = []
    total_size = 0
    for name in os.listdir(upload_dir):
        full_path = os.path.join(upload_dir, name)
        if not os.path.isfile(full_path) or os.path.abspath(full_path) in exclude_set:
            continue
        try:
            stat = os.stat(full_path)
        except OSError:
            continue
        remaining.append((full_path, stat.st_mtime, stat.st_size))
        total_size += stat.st_size

    if total_size <= max_total_bytes:
        return 0, 0

    # Delete oldest-first until under cap
    remaining.sort(key=lambda x: x[1])  # oldest mtime first
    for path, _, size in remaining:
        if total_size <= max_total_bytes:
            break
        try:
            os.remove(path)
            deleted_count += 1
            deleted_bytes += size
            total_size -= size
        except OSError:
            continue

    return deleted_count, deleted_bytes


def cleanup_stale_files(
    upload_dir: str,
    max_age_seconds: int,
    max_total_bytes: Optional[int] = None,
    exclude: Optional[Iterable[str]] = None,
):
    """
    Delete old files in upload_dir. Optionally enforce a total size cap by deleting oldest first.
    Returns (deleted_count, deleted_bytes).
    """
    if not upload_dir or not os.path.isdir(upload_dir):
        return 0, 0

    now = time.time()
    exclude_set = {os.path.abspath(p) for p in (exclude or []) if p}

    deleted_count = 0
    deleted_bytes = 0

    # Delete files older than max_age_seconds
    for name in os.listdir(upload_dir):
        full_path = os.path.join(upload_dir, name)
        if not os.path.isfile(full_path) or os.path.abspath(full_path) in exclude_set:
            continue
        try:
            stat = os.stat(full_path)
        except OSError:
            continue
        if now - stat.st_mtime > max_age_seconds:
            try:
                os.remove(full_path)
                deleted_count += 1
                deleted_bytes += stat.st_size
            except OSError:
                continue

    # Enforce size cap if requested
    if max_total_bytes is not None:
        extra_deleted_count, extra_deleted_bytes = _enforce_size_cap(upload_dir, max_total_bytes, exclude_set)
        deleted_count += extra_deleted_count
        deleted_bytes += extra_deleted_bytes

    return deleted_count, deleted_bytes
