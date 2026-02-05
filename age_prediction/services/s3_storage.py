# -*- coding: utf-8 -*-
"""
Helpers for S3-backed uploads/results in the serverless deployment.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import boto3
from botocore.config import Config
from werkzeug.utils import secure_filename


def _get_s3_client(region: Optional[str] = None):
    """Return a boto3 S3 client for the given region (or default config)."""
    s3_config = Config(signature_version="s3v4", s3={"addressing_style": "virtual"})
    if region:
        # Force regional endpoint to avoid temporary redirect responses on presigned PUTs.
        endpoint_url = f"https://s3.{region}.amazonaws.com"
        return boto3.client("s3", region_name=region, endpoint_url=endpoint_url, config=s3_config)
    return boto3.client("s3", config=s3_config)


def _safe_basename(filename: str) -> str:
    """Return a sanitized filename (input: original name, output: safe basename)."""
    safe_name = secure_filename(filename or "")
    if safe_name == "":
        return "upload"
    return safe_name


def build_upload_key(filename: str) -> str:
    """Build an S3 object key for an upload (input: filename, output: key string)."""
    # Prefix keeps uploads/results separated and avoids user-controlled paths.
    safe_name = _safe_basename(filename)
    unique_prefix = uuid.uuid4().hex
    return f"uploads/{unique_prefix}_{safe_name}"


def build_results_prefix(request_id: str) -> str:
    """Build a results prefix (input: request id, output: prefix string)."""
    # Results are grouped by a request id so cleanup and listing are scoped.
    return f"results/{request_id}"


def generate_presigned_put_url(
    bucket: str,
    key: str,
    content_type: str,
    expires_in: int,
    region: Optional[str] = None,
) -> str:
    """Return a presigned PUT URL (inputs: bucket/key/content_type/expires; output: URL)."""
    # Caller uploads directly to S3; backend never receives the file bytes.
    client = _get_s3_client(region)
    return client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=expires_in,
    )


def generate_presigned_get_url(
    bucket: str,
    key: str,
    expires_in: int,
    region: Optional[str] = None,
) -> str:
    """Return a presigned GET URL (inputs: bucket/key/expires; output: URL)."""
    # Signed read URL lets clients fetch results without making bucket public.
    client = _get_s3_client(region)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def download_to_path(bucket: str, key: str, path: str, region: Optional[str] = None) -> None:
    """Download S3 object to local path (inputs: bucket/key/path; output: None)."""
    # Lambda uses local temp storage (/tmp) during inference.
    client = _get_s3_client(region)
    client.download_file(bucket, key, path)


def upload_file(
    path: str,
    bucket: str,
    key: str,
    content_type: Optional[str] = None,
    region: Optional[str] = None,
) -> None:
    """Upload local file to S3 (inputs: path/bucket/key; output: None)."""
    # Store inference outputs in the results bucket for short-lived access.
    client = _get_s3_client(region)
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type
    if extra_args:
        client.upload_file(path, bucket, key, ExtraArgs=extra_args)
    else:
        client.upload_file(path, bucket, key)
