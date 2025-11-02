# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:04:10 2020

@author: Cameron
"""

from flask import Flask, redirect, url_for, render_template, request, session, flash, jsonify
import torch
import os
import gc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import FaceModels as fm
import FaceDatasets as fds
from facenet_pytorch import MTCNN
from werkzeug.exceptions import RequestEntityTooLarge
import pillow_heif
pillow_heif.register_heif_opener()

matplotlib.use('Agg') #use the non gui backend

device = torch.device('cpu')

detector = MTCNN(min_face_size=30, device = device)
embeddor = fm.Facenet_Embeddor(device = device).eval().to(device)
model = fm.Ensemble_Model(device = device).eval().to(device)

#TODO: Now figure out other ways to put up images, need to be able to put up 
#your image from the prediction plot. Then also need to be able to access OTHER
#peoples webcams. Also just get picture deposit first, I think its more accurate on 
#pics rather than webcams.

#shouldnt store permanent data in session (like forever, so just like their name is ok)
app = Flask(__name__)
app.secret_key = 'hello'
app.config['UPLOAD_FOLDER'] = 'static/images'
max_mb = 8
app.config['MAX_CONTENT_LENGTH'] = max_mb * 1024 * 1024 #converted to bytes

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

@app.route("/", methods = ["POST", "GET"])
def home():
    if request.method == "POST":
        #remove any previous data before writing
        if 'uploaded_filename' in session:
            session.pop('uploaded_filename', None)
        uploaded_file = request.files["file"]
        if uploaded_file.filename != '':
            uploaded_file.save(app.config['UPLOAD_FOLDER'] + '/' + uploaded_file.filename)
            session['uploaded_filename'] = app.config['UPLOAD_FOLDER'] + '/' + uploaded_file.filename
            #flash("Image Submission Successful!")
            gc.collect()
            return redirect(url_for('resultspage'))
        else:
            flash("Image input unsuccesful")
            return render_template("inputpage.html")
    else:
        return render_template("inputpage.html")

@app.route("/resultspage", methods = ["POST", "GET"])
def resultspage():
    if request.method == 'POST':
        if request.form['submit_button'] == "Try Another Image!":
            #delete the previous images
            files_to_delete = [session['uploaded_filename']] + session['proc_images']
            for file_to_delete in files_to_delete:
                os.remove(file_to_delete)
            #also check if directory is super full from people just closing out
            #so then we also delete all the files
            folder_size = np.sum([os.path.getsize('static/images/'+file_loc) \
                                  for file_loc in os.listdir('static/images')])
            if folder_size > 50 * 1024 * 1024: #keep it under 50 mbytes
                for file_to_delete in os.listdir('static/images'):
                    os.remove('static/images/' + file_to_delete)
            return redirect(url_for('home'))
        else: #if not the button, then show results again, but if not button then how is it post, 
                #so probably wont reach this tab
            fig_path = app.config['UPLOAD_FOLDER'] + '/' + 'prediction.jpg'
            return render_template("resultspage.html", figpath=fig_path)
    else:
        explainsmall = False
        if 'uploaded_filename' in session:
            uploaded_filename = session['uploaded_filename']
            #big fancy calculations, figlist will be individual prediction plots
            #or it will be no faces found error, big_fig will be total image
            #with annotations or it will be None
            #Also added a try except here in case something goes wrong
            #like out of memory, then we can flash a message
            try:
                figlist, big_fig = fds.jpg2age(uploaded_filename, detector, embeddor, model, 
                              device=device, tight_layout = True)
            except RuntimeError as e:
                if "memory" in str(e).lower() or "oom" in str(e).lower():
                    flash('OOM Error: Picture was too large to process on this server. Please upload a smaller image.')
                    gc.collect()
                    return redirect(url_for('home'))
                else:
                    raise
            if figlist[0] == 'No faces found' or figlist[0] == 'Image not RGB':
                flash('Unable to use picture: ' + str(figlist[0]))
                gc.collect()
                return redirect(url_for('home'))
            else:
                filename = '.'.join(uploaded_filename.split('/')[-1].split('.')[:-1])
                fig_paths = []
                session['proc_images'] = []
                #we arent going to plot if only one face, its redundant
                #unless the face errored then we use annot to communicate error
                if len(figlist) > 1 or isinstance(figlist[0], str):
                    big_fig_path = app.config['UPLOAD_FOLDER'] + '/' + filename + \
                        '_big_fig_' + str(np.random.randint(0, 10000)) + '.jpg'
                    big_fig.savefig(big_fig_path)
                    session['proc_images'].append(big_fig_path)         #add these paths to session so we can delete them later
                    plt_big = True
                else:
                    big_fig_path = None
                    plt_big = False
                for fi, fig in enumerate(figlist):
                    if isinstance(fig, str) == False:
                        fig_paths.append(app.config['UPLOAD_FOLDER'] + '/' + filename + '_prediction' 
                                         + str(fi) + '_' + str(np.random.randint(0, 10000)) + '.jpg')
                        fig.savefig(fig_paths[-1])
                        session['proc_images'].append(fig_paths[-1])    #add these paths to session so we can delete them later
                        explainsmall = True
            plt.close('all')
            gc.collect()
            return render_template("resultspage.html", figpath=fig_paths, 
                                   bigfigpath=big_fig_path, pltbig=plt_big,
                                   explainsmall=explainsmall)
        else:
            flash("You did not submit an image!")
            return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=False)