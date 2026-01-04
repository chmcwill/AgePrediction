# -*- coding: utf-8 -*-
"""
Helpers for storing and cleaning up uploaded and generated files.
"""
import os


def save_upload(uploaded_file, upload_dir):
    """
    Save an uploaded file to the upload directory.
    """
    if uploaded_file is None or uploaded_file.filename == '':
        raise ValueError("No file provided")

    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, uploaded_file.filename)
    uploaded_file.save(dest_path)
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
