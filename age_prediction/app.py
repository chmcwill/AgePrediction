# -*- coding: utf-8 -*-
"""
Flask application entrypoint and factory for the age prediction demo.
"""

from flask import Flask, redirect, url_for, render_template, request, session, flash
import os
import gc
import matplotlib
import pillow_heif
from werkzeug.exceptions import RequestEntityTooLarge

from age_prediction.services.prediction import run_prediction
from age_prediction.services import storage
from age_prediction.services.errors import (
    InvalidImageError,
    NoFacesFoundError,
    InferenceOOMError,
    StorageError,
)


pillow_heif.register_heif_opener()

matplotlib.use('Agg')  # use the non gui backend

# shouldnt store permanent data in session (like forever, so just like their name is ok)
MAX_CONTENT_LENGTH_MB = 8
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_FOLDER = os.path.join(ROOT_DIR, "templates")
STATIC_FOLDER = os.path.join(ROOT_DIR, "static")
MAX_CONTENT_LENGTH_B = MAX_CONTENT_LENGTH_MB * 1024 * 1024  #converted to bytes
DEFAULT_UPLOAD_MAX_AGE_SECONDS = 6 * 60 * 60  # 6 hours
DEFAULT_UPLOAD_MAX_TOTAL_BYTES = None  # set to int bytes to enforce a cap
UPLOAD_FOLDER_MAX_TOTAL_BYTES = 50 * 1024 * 1024  # keep upload folder under 50 MB


def create_app():
    app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
    app.secret_key = 'hello'
    app.config['UPLOAD_FOLDER'] = 'static/images'
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_B
    app.config['UPLOAD_MAX_AGE_SECONDS'] = DEFAULT_UPLOAD_MAX_AGE_SECONDS
    app.config['UPLOAD_MAX_TOTAL_BYTES'] = UPLOAD_FOLDER_MAX_TOTAL_BYTES

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(e):
        flash(f"File is too large. Maximum allowed size is {MAX_CONTENT_LENGTH_MB} MB.")
        return redirect(url_for('home'))

    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500

    def _session_files():
        """Collect current session file paths for cleanup."""
        files = []
        if uploaded := session.get('uploaded_filename'):
            files.append(uploaded)
        files.extend(session.get('proc_images', []))
        return files

    @app.route("/", methods=["GET"])
    def home():
        exclude_paths = _session_files()
        storage.cleanup_stale_files(
            app.config['UPLOAD_FOLDER'],
            app.config['UPLOAD_MAX_AGE_SECONDS'],
            app.config.get('UPLOAD_MAX_TOTAL_BYTES'),
            exclude=exclude_paths,
        )
        return render_template("inputpage.html")

    @app.route("/", methods=["POST"])
    def upload_image():
        # Clear prior session state and files before handling a new upload
        storage.cleanup_files(_session_files())
        session.pop('uploaded_filename', None)
        session.pop('proc_images', None)

        uploaded_file = request.files.get("file")
        if not uploaded_file or uploaded_file.filename == "":
            flash("Please choose an image before submitting.")
            return redirect(url_for('home'))

        try:
            saved_path = storage.save_upload(uploaded_file, app.config['UPLOAD_FOLDER'])
        except StorageError as exc:
            flash(f"Image input unsuccessful: {exc}")
            return redirect(url_for('home'))

        session['uploaded_filename'] = saved_path
        return redirect(url_for('resultspage'))

    @app.route("/resultspage", methods=["GET"], endpoint="resultspage")
    def resultspage():
        uploaded_filename = session.get('uploaded_filename')
        if not uploaded_filename:
            flash("You did not submit an image!")
            return redirect(url_for('home'))

        try:
            result, generated_files = run_prediction(uploaded_filename, app.config['UPLOAD_FOLDER'])
        except InferenceOOMError:
            flash('OOM Error: Picture was too large to process on this server. Please upload a smaller image.')
            gc.collect()
            return redirect(url_for('home'))
        except (InvalidImageError, NoFacesFoundError) as e:
            flash(f'Unable to use picture: {e}')
            gc.collect()
            return redirect(url_for('home'))

        session['proc_images'] = generated_files
        explainsmall = result.explainsmall
        plt_big = result.plt_big
        fig_paths = result.fig_paths
        big_fig_path = result.big_fig_path
        gc.collect()
        return render_template("resultspage.html", figpath=fig_paths,
                               bigfigpath=big_fig_path, pltbig=plt_big,
                               explainsmall=explainsmall)

    @app.route("/resultspage", methods=["POST"])
    def reset_results():
        storage.cleanup_files(_session_files())
        session.pop('uploaded_filename', None)
        session.pop('proc_images', None)
        # Opportunistically prune the folder if it is over the configured cap.
        storage.cleanup_stale_files(
            app.config['UPLOAD_FOLDER'],
            app.config['UPLOAD_MAX_AGE_SECONDS'],
            app.config.get('UPLOAD_MAX_TOTAL_BYTES'),
        )
        return redirect(url_for('home'))

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=False)
