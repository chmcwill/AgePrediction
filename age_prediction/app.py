# -*- coding: utf-8 -*-
"""
Flask application entrypoint and factory for the age prediction demo.
"""

from flask import Flask, url_for, request, jsonify, send_from_directory
import os
import gc
import tempfile
import uuid
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from age_prediction.services import storage
from age_prediction.services import s3_storage
from age_prediction.services.errors import (
    InvalidImageError,
    NoFacesFoundError,
    InferenceOOMError,
)

# shouldnt store permanent data in session (like forever, so just like their name is ok)
MAX_CONTENT_LENGTH_MB = 1
max_mb = MAX_CONTENT_LENGTH_MB
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_FOLDER = os.path.join(ROOT_DIR, "static")
MAX_CONTENT_LENGTH_B = MAX_CONTENT_LENGTH_MB * 1024 * 1024  #converted to bytes
DEFAULT_UPLOAD_MAX_AGE_SECONDS = 6 * 60 * 60  # 6 hours
DEFAULT_UPLOAD_MAX_TOTAL_BYTES = None  # set to int bytes to enforce a cap
UPLOAD_FOLDER_MAX_TOTAL_BYTES = 50 * 1024 * 1024  # keep upload folder under 50 MB


def create_app():
    app = Flask(__name__, static_folder=STATIC_FOLDER)
    app.config['UPLOAD_FOLDER'] = 'static/images'
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_B
    app.config['UPLOAD_MAX_AGE_SECONDS'] = DEFAULT_UPLOAD_MAX_AGE_SECONDS
    app.config['UPLOAD_MAX_TOTAL_BYTES'] = UPLOAD_FOLDER_MAX_TOTAL_BYTES
    # API_BASE_URL is injected into the static frontend for the JS client.
    app.config['API_BASE_URL'] = os.environ.get("API_BASE_URL", "")
    app.config['S3_REGION'] = os.environ.get("S3_REGION", "us-east-2")
    app.config['S3_UPLOAD_BUCKET'] = os.environ.get("S3_UPLOAD_BUCKET")
    app.config['S3_RESULTS_BUCKET'] = os.environ.get("S3_RESULTS_BUCKET")
    app.config['S3_PRESIGN_EXPIRES_SECONDS'] = int(os.environ.get("S3_PRESIGN_EXPIRES_SECONDS", "600"))
    app.config['S3_RESULT_URL_EXPIRES_SECONDS'] = int(os.environ.get("S3_RESULT_URL_EXPIRES_SECONDS", "3600"))
    # Allow browser JS to call the API when served from a static bucket/CloudFront.
    app.config['CORS_ALLOW_ORIGIN'] = os.environ.get("CORS_ALLOW_ORIGIN", "*")
    app.config['LOCAL_STORAGE'] = os.environ.get("LOCAL_STORAGE", "").lower() in ("1", "true", "yes", "on")
    app.config['LOCAL_STORAGE_DIR'] = os.environ.get(
        "LOCAL_STORAGE_DIR",
        os.path.join(ROOT_DIR, "tmp"),
    )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(e):
        """Handle oversized uploads (input: error; output: JSON response)."""
        return jsonify(
            {
                "error": "file_too_large",
                "message": f"File too large. Max size is {MAX_CONTENT_LENGTH_MB} MB.",
            }
        ), 413

    def _require_s3_config():
        """Validate S3 env vars (input: app config; output: error response or None)."""
        # Fail fast if the function isn't wired to S3 buckets.
        if app.config.get("LOCAL_STORAGE"):
            return None
        missing = []
        if not app.config.get("S3_UPLOAD_BUCKET"):
            missing.append("S3_UPLOAD_BUCKET")
        if not app.config.get("S3_RESULTS_BUCKET"):
            missing.append("S3_RESULTS_BUCKET")
        if missing:
            return jsonify(
                {
                    "error": "s3_config_missing",
                    "message": "Server storage is not configured.",
                    "missing": missing,
                }
            ), 500
        return None

    @app.after_request
    def add_cors_headers(response):
        """Add CORS headers (input: response; output: response)."""
        # CORS is only needed for API routes called by the static frontend.
        if request.path.startswith("/api/"):
            response.headers["Access-Control-Allow-Origin"] = app.config["CORS_ALLOW_ORIGIN"]
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "POST,PUT,OPTIONS"
        return response

    @app.route("/api/<path:unused>", methods=["OPTIONS"])
    def api_options(unused):
        """Return preflight response (input: any path; output: empty 204)."""
        # Preflight response for browsers.
        return ("", 204)


    @app.route("/api/presign", methods=["POST"])
    def presign_upload():
        """Create presigned PUT URL (input: JSON filename/content_type; output: JSON)."""
        # Returns a presigned URL so the browser uploads directly to S3.
        payload = request.get_json(silent=True) or {}
        filename = payload.get("filename", "")
        content_type = payload.get("content_type") or "application/octet-stream"
        if not filename:
            return jsonify({"error": "filename_required", "message": "Filename is required."}), 400

        if app.config.get("LOCAL_STORAGE"):
            os.makedirs(app.config["LOCAL_STORAGE_DIR"], exist_ok=True)
            safe_name = secure_filename(filename or "")
            if safe_name == "":
                safe_name = "upload"
            key = f"uploads/{uuid.uuid4().hex}_{safe_name}"
            url = url_for("local_upload", key=key, _external=True)
            return jsonify({"url": url, "key": key, "expires_in": app.config["S3_PRESIGN_EXPIRES_SECONDS"]})

        if (err := _require_s3_config()) is not None:
            return err

        key = s3_storage.build_upload_key(filename)
        url = s3_storage.generate_presigned_put_url(
            bucket=app.config["S3_UPLOAD_BUCKET"],
            key=key,
            content_type=content_type,
            expires_in=app.config["S3_PRESIGN_EXPIRES_SECONDS"],
            region=app.config.get("S3_REGION"),
        )
        return jsonify({"url": url, "key": key, "expires_in": app.config["S3_PRESIGN_EXPIRES_SECONDS"]})

    @app.route("/api/upload/<path:key>", methods=["PUT"])
    def local_upload(key):
        """Accept local dev uploads (input: raw body; output: JSON)."""
        if not app.config.get("LOCAL_STORAGE"):
            return jsonify(
                {"error": "local_storage_disabled", "message": "Local storage is disabled."}
            ), 400
        if not key:
            return jsonify({"error": "key_required", "message": "Upload key is required."}), 400
        filename = os.path.basename(key)
        if filename == "":
            return jsonify({"error": "invalid_key", "message": "Upload key is invalid."}), 400
        os.makedirs(app.config["LOCAL_STORAGE_DIR"], exist_ok=True)
        dest_path = os.path.join(app.config["LOCAL_STORAGE_DIR"], filename)
        with open(dest_path, "wb") as handle:
            handle.write(request.get_data())
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            storage.cleanup_files([dest_path])
            return jsonify({"error": "upload_failed", "message": "Upload failed."}), 500
        return jsonify({"ok": True, "key": key})

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health ping (input: optional deep=true; output: JSON ok flag)."""
        # Optional deep warmup loads model weights so first prediction is faster.
        if request.args.get("deep") == "true":
            # Import prediction to load plotting deps; also load model weights.
            from age_prediction.services.prediction import DEFAULT_PREDICTION_CONFIG
            from age_prediction.services.models import get_runtime_models
            classes = tuple(range(10, 71))
            get_runtime_models(DEFAULT_PREDICTION_CONFIG.classes, DEFAULT_PREDICTION_CONFIG.min_face_size)
        return jsonify({"ok": True})

    @app.route("/api/predict", methods=["POST"])
    def predict_from_s3():
        """Run inference on S3 upload (input: JSON key/request_id; output: JSON URLs)."""
        # Pulls the uploaded image from S3, runs inference, pushes results back to S3.
        payload = request.get_json(silent=True) or {}
        key = payload.get("key")
        if not key:
            return jsonify({"error": "key_required", "message": "Upload key is required."}), 400
        if not str(key).startswith("uploads/"):
            return jsonify(
                {"error": "invalid_key", "message": "Upload key must start with uploads/."}
            ), 400

        # request_id scopes result object keys; caller can reuse for grouping.
        request_id = payload.get("request_id") or uuid.uuid4().hex
        suffix = os.path.splitext(key)[1] or ".jpg"

        try:
            # Lazy import keeps /api/presign fast during cold starts.
            from age_prediction.services.prediction import run_prediction
            if app.config.get("LOCAL_STORAGE"):
                os.makedirs(app.config["LOCAL_STORAGE_DIR"], exist_ok=True)
                filename = os.path.basename(key)
                local_image = os.path.join(app.config["LOCAL_STORAGE_DIR"], filename)
                if not os.path.exists(local_image):
                    return jsonify(
                        {"error": "not_found", "message": "Uploaded file not found."}
                    ), 404
                result, generated_files = run_prediction(local_image, app.config["LOCAL_STORAGE_DIR"])

                def _local_url(path):
                    return url_for("local_results", filename=os.path.basename(path), _external=True)

                big_fig_url = _local_url(result.big_fig_path) if result.big_fig_path else None
                fig_urls = [_local_url(path) for path in result.fig_paths]
                return jsonify({"request_id": request_id, "big_fig_url": big_fig_url, "fig_urls": fig_urls})

            if (err := _require_s3_config()) is not None:
                return err

            with tempfile.TemporaryDirectory(prefix="agepred_") as tmpdir:
                # Lambda provides /tmp for scratch work; all outputs are uploaded to S3.
                local_image = os.path.join(tmpdir, f"upload{suffix}")
                s3_storage.download_to_path(
                    bucket=app.config["S3_UPLOAD_BUCKET"],
                    key=key,
                    path=local_image,
                    region=app.config.get("S3_REGION"),
                )
                result, generated_files = run_prediction(local_image, tmpdir)

                prefix = s3_storage.build_results_prefix(request_id)
                uploaded_map = {}
                for path in generated_files:
                    filename = os.path.basename(path)
                    object_key = f"{prefix}/{filename}"
                    s3_storage.upload_file(
                        path=path,
                        bucket=app.config["S3_RESULTS_BUCKET"],
                        key=object_key,
                        content_type="image/jpeg",
                        region=app.config.get("S3_REGION"),
                    )
                    uploaded_map[filename] = object_key

                big_fig_url = None
                if result.big_fig_path:
                    big_key = uploaded_map.get(os.path.basename(result.big_fig_path))
                    if big_key:
                        big_fig_url = s3_storage.generate_presigned_get_url(
                            bucket=app.config["S3_RESULTS_BUCKET"],
                            key=big_key,
                            expires_in=app.config["S3_RESULT_URL_EXPIRES_SECONDS"],
                            region=app.config.get("S3_REGION"),
                        )

                fig_urls = []
                for path in result.fig_paths:
                    fig_key = uploaded_map.get(os.path.basename(path))
                    if not fig_key:
                        continue
                    fig_urls.append(
                        s3_storage.generate_presigned_get_url(
                            bucket=app.config["S3_RESULTS_BUCKET"],
                            key=fig_key,
                            expires_in=app.config["S3_RESULT_URL_EXPIRES_SECONDS"],
                            region=app.config.get("S3_REGION"),
                        )
                    )

                return jsonify(
                    {
                        "request_id": request_id,
                        "big_fig_url": big_fig_url,
                        "fig_urls": fig_urls,
                    }
                )
        except InferenceOOMError:
            gc.collect()
            return jsonify(
                {
                    "error": "oom",
                    "message": "Image too large to process. Try a smaller resolution.",
                }
            ), 400
        except (InvalidImageError, NoFacesFoundError) as exc:
            gc.collect()
            return jsonify(
                {
                    "error": "invalid_image",
                    "message": "No usable face detected. Try a clear, front-facing image.",
                    "detail": str(exc),
                }
            ), 400
        except Exception as exc:
            return jsonify(
                {
                    "error": "server_error",
                    "message": "Server error. Please try again in a moment.",
                    "detail": str(exc),
                }
            ), 500

    @app.route("/local-results/<path:filename>")
    def local_results(filename):
        """Serve local dev results from LOCAL_STORAGE_DIR."""
        if not app.config.get("LOCAL_STORAGE"):
            return jsonify(
                {"error": "local_storage_disabled", "message": "Local storage is disabled."}
            ), 400
        safe_name = os.path.basename(filename)
        return send_from_directory(app.config["LOCAL_STORAGE_DIR"], safe_name)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=False)
