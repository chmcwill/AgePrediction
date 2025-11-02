# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:15:22 2020

@author: Cameron 

This file contains plotting functions for the age prediction project.

"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import FaceDatasets as fds

def imshow_tensor(tensor, title_str = 'Image Tensor', landmarks = None):
    ''' Take tensor and plot it as img
    INPUTS:
        tensor (torch.tensor): tensor of an image
        title_str (str): the str to use to title the plot with
        landmarks (numpy.array): the landmarks to optionally annotate with
    '''
    img = fds.tensor_to_img(tensor.cpu())
    plt.figure(figsize = (6,6))
    plt.imshow(img)
    plt.title(title_str)
    if landmarks is not None:
        for i in range(landmarks.shape[0]): 
            plt.plot(landmarks[i,0], landmarks[i,1], 'o', color = 'b')

def plot_image_and_pred(image, pred, CLASSIF, image_name = 'input image', output_softmax = None,
                        img_input_size = 160, fig_ax = None, tight_layout = True, figshow = False):
    ''' Function to take in an image with a prediction and create a plot with two sublots
        one for the cropped and rotated image and the other for the prediction.
        if CLASSIF is True, then will plot the PDF of the age prediction
    INPUTS:
        image (pil image): a pillow image to plot with
        pred (float): the age prediction corresponding to the image
        CLASSIF (bool): True if using classification, False if using regression
        image_name (str): name to title the plot with, usually this is the filename
        output_softmax (np.array): the softmax output to plot the classification results
        img_input_size (int): the size of the image the model takes in, must plot realistically
        fig_ax (axes object): if there already is an axes object you want to plot on, you can 
                            re plot on it instead of creating a new figure
        tight_layout (bool): whether or not to call tight layout on the image to 
                            help prevent overlap and reduce whitespace
        figshow(bool): whether or not to call fig.show() after plotting, this 
                        needs to be False when operating a website
    OUTPUTS:
        fig (matplotlib figure): the figure that the plot was created on, then it can
                                be passed out and saved on the web for visualization
                                on the website.    
    '''
    #first ensure input is square, and resize it to the model input if needed
    h, w = image.size
    assert h == w, 'Must input square image, no warping for model, use extract face and 0 pad'
    if h != img_input_size:
        image = nn.functional.interpolate(image, size=(img_input_size, img_input_size),
                                                  mode='bilinear', align_corners=False)
    #create or use a previous figure axes to plot on
    if fig_ax is None:
        fig, axs = plt.subplots(1, 2, figsize = (10, 5))                        #create figure with subplots
    else:
        fig, axs = fig_ax
        for ax in axs:
            ax.clear()
    
    axs[0].imshow(image)                                                        #plot image on first subplot
    axs[0].set_title('Cropped and Rotated Face', fontsize = 13)
    if CLASSIF:
        fig.suptitle('Age Prediction using Classification for ' + image_name, fontsize = 16)
        axs[1].plot(range(10,71), output_softmax.squeeze())                     #plot age pred distrib
        axs[1].set_xlabel('Age', fontsize = 12)
        axs[1].set_ylabel('Probability', fontsize = 12)
        axs[1].set_title('Age Prediction Distribution (pred = ' + str(pred) + ')', fontsize = 13)
        axs[1].plot([pred]*2, [0, np.max(output_softmax)], linewidth = 2)       #plot bar for prediction
        axs[1].legend(['PDF', 'Expectation'], loc="upper right")
    else:
        fig.suptitle('Age Prediction using Regression for ' + image_name)
        axs[1].set_xlabel('Age')
        axs[1].set_title('Age Prediction (pred = ' + str(pred) + ')')
        axs[1].plot([pred]*2, [0, 1], linewidth = 2)                            #plot bar for prediction
        axs[1].set_xlim([10, 70])
    
    if tight_layout:
        plt.tight_layout()
    
    if figshow:
        plt.draw()
        plt.pause(.001)
        fig.show()
    
    return fig

def overlay_preds_on_img(image, face_bounds, preds):
    ''' Create an annotated version of the inputted image with the passed in face bounds
        and predictions. The annotations will be sized according to how many faces 
        are in the image and whether or not they are close to the edge or not.
    INPUTS:
        image (pil image): image containing the faces that we want to annotate on
        face_bounds (list of np.array): list of bounding box locations on the face
        preds (list of int or str): list of predictions or error messages corresponding
                                    to each face bound
    OUTPUTS:
        fig (figure object): the matplotlib figure that the image with annotations are 
                            plotted on
        ax (axes object): the axes object that the image with annotations are plotted on
                            there is only one axes so not a list   
    '''
    #get number of faces in image
    n_faces = len(face_bounds)
    
    #find ratio closest to fig size for plotting minimal whitespace
    img_w, img_h = image.size
    ratio = img_w/img_h
    plt_h = 10
    plt_w = max(int(plt_h*(ratio)), 8)
    fig, ax = plt.subplots(1, 1, figsize = (plt_w, plt_h))  
    
    #show the image on the axes and title with padding to avoid being annotated over
    ax.imshow(image)
    ax.set_title('Total Image with Annotated Predictions', fontsize = 20, pad=20)
    
    #sort to have errors first, so they are not covering up ages
    preds_error = [pred for pred in preds if isinstance(pred, str)]
    preds_tensr = [pred for pred in preds if isinstance(pred, np.ndarray)]
    face_bounds_error = [fb for fb, pred in zip(face_bounds, preds) if isinstance(pred, str)]
    face_bounds_tensr = [fb for fb, pred in zip(face_bounds, preds) if isinstance(pred, np.ndarray)]
    #now add the lists to have errors be plotted first
    preds_sorted = preds_error + preds_tensr
    face_bounds_sorted = face_bounds_error + face_bounds_tensr
    
    #get the sizes for each face from the sorted bounds
    face_sizes = [face_bound[3] - face_bound[1] for face_bound in face_bounds_sorted]
    
    #for each face in the list, get the bounding box and the prediction
    for i, (face_bound, pred) in enumerate(zip(face_bounds_sorted, preds_sorted)):
        #now we want to see here if prediction or flag
        if isinstance(pred, str):
            #set size and verbosity for error messages
            pred = 'Error:\n' + pred
            txt_size = 9
            h_offset = .75
        else:
            #set size and verbosity based on how many faces
            pred = 'Age: ' + str(pred[0]) if n_faces <= 3 else str(pred[0])
            txt_size = 20 if n_faces <= 2 else 18 if n_faces <= 5 else 15 if n_faces <= 8 else 12
            h_offset = .4
        
        # Create a Rectangle patch for the bounding box
        rect = patches.Rectangle((face_bound[0],face_bound[1]),face_bound[2]-face_bound[0],
                                 face_bound[3]-face_bound[1],linewidth=1,
                                 edgecolor=(1.0, 0.7, 0.7),facecolor='none')
        # Add the patch to the Axes 
        ax.add_patch(rect)
        
        #lets get the location to put the age or error annotation
        bbx_x = np.mean(face_bound[[0,2]]) #mean of the x1 and x2
        #now if the face is near the top, we need to indent the arrow a bit
        arrow_indent_pct = .2 if face_bound[1] < .05*img_h else .05 if face_bound[1] < .1*img_h else .01 
        bbx_y = np.max(face_bound[1], 0) + arrow_indent_pct*face_sizes[i]
        
        #now get the amount to offset the text from the arrow end, 
        #this is in reg cartesian coords on the whole image
        offset_y = np.min((h_offset*(face_bound[3] - face_bound[1]), .95*bbx_y))
        #we want offset x to be slightly left on the left side of the image 
        #and vice versa, this helps space out and make things less redundant
        #we want to go from -.02 to .02 of bbx_x and clamp [0, img_w] 
        #so linearly interpolate
        offset_x_pct = (.02 - (-.02)) * (bbx_x/img_w) + (-.02)
        offset_x = np.clip(offset_x_pct*img_w, -bbx_x, (img_w - bbx_x)) 
        
        #create the annotation object and annotate it to axis
        ax.annotate(pred,
                  xy=(bbx_x, bbx_y), xycoords='data',                           #location for arrow end
                  xytext=(bbx_x+offset_x, bbx_y-offset_y), textcoords='data',   #location for bubble annotation
                  size=txt_size, va = 'center', ha = 'center',                  #size for annot and we want it centered
                  bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),   #make the bubble rounded and pink/red
                  arrowprops=dict(arrowstyle="wedge,tail_width=1.",             #style the arrow itself
                                  fc=(1.0, 0.7, 0.7), ec="none",
                                  patchA=None,
                                  patchB=None,
                                  relpos=(0.5, 0.5))                            #we want the arrow to connect to middle of bubble
                  )
    #call tight layout to minimize whitespace
    plt.tight_layout()
    return fig, ax

