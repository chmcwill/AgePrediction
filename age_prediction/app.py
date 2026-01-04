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
)


pillow_heif.register_heif_opener()

matplotlib.use('Agg')  # use the non gui backend

# shouldnt store permanent data in session (like forever, so just like their name is ok)
max_mb = 8


def create_app():
    app = Flask(__name__)
    app.secret_key = 'hello'
    app.config['UPLOAD_FOLDER'] = 'static/images'
    app.config['MAX_CONTENT_LENGTH'] = max_mb * 1024 * 1024  # converted to bytes

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(e):
        flash(f"File is too large. Maximum allowed size is {max_mb} MB.")
        return redirect(url_for('home'))

    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500

    @app.route("/", methods=["POST", "GET"])
    def home():
        if request.method == "POST":
            # remove any previous data before writing
            if 'uploaded_filename' in session:
                session.pop('uploaded_filename', None)
            uploaded_file = request.files["file"]
            if uploaded_file.filename != '':
                saved_path = storage.save_upload(uploaded_file, app.config['UPLOAD_FOLDER'])
                session['uploaded_filename'] = saved_path
                # flash("Image Submission Successful!")
                gc.collect()
                return redirect(url_for('resultspage'))
            else:
                flash("Image input unsuccesful")
                return render_template("inputpage.html")
        else:
            return render_template("inputpage.html")

    @app.route("/resultspage", methods=["POST", "GET"])
    def resultspage():
        if request.method == 'POST':
            if request.form['submit_button'] == "Try Another Image!":
                # delete the previous images
                files_to_delete = [session.get('uploaded_filename')] + session.get('proc_images', [])
                storage.cleanup_files(files_to_delete)
                # also check if directory is super full from people just closing out
                # so then we also delete all the files
                folder_size = sum([os.path.getsize('static/images/' + file_loc)
                                   for file_loc in os.listdir('static/images')])
                if folder_size > 50 * 1024 * 1024:  # keep it under 50 mbytes
                    for file_to_delete in os.listdir('static/images'):
                        storage.cleanup_files(['static/images/' + file_to_delete])
                return redirect(url_for('home'))
            else:  # if not the button, then show results again, but if not button then how is it post,
                    # so probably wont reach this tab
                fig_path = app.config['UPLOAD_FOLDER'] + '/' + 'prediction.jpg'
                return render_template("resultspage.html", figpath=fig_path)
        else:
            explainsmall = False
            if 'uploaded_filename' in session:
                uploaded_filename = session['uploaded_filename']
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
            else:
                flash("You did not submit an image!")
                return redirect(url_for('home'))

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=False)
