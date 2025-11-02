# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:25:12 2020

@author: Cameron

This file contains helper functions pertaining to data processing for the 
age prediction project.

"""

import os
import cv2
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import WeightedRandomSampler

import FacePlotting as fpl
import FaceModels as fm

def jpg2age(test_image_path, detector, embeddor, model, tight_layout = True, 
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            figshow = False):
    ''' This function is used for taking in a path to an image and identifying
        all the faces then predicting the age of each. Further, it annotates 
        the predictions on the original image itself if there are multiple faces.
        This is mainly used for the website.
    INPUTS:
        test_image_path (str): path to the image that will be tested
        detector (object): the face bounding box detector
        embeddor (object): the pretrained InceptionV1 pytorch model to create 
                            embeddings on the image.
        model (object): the model to take the embeddings and output an age
        tight_layout (bool): whether or not to call tight layout on plots to 
                            prevent overlap of plots and reduce whitespace
        device (str): whether we are calculating on gpu or cpu
        figshow(bool): whether or not to call fig.show() after plotting, this 
                        needs to be False when operating a website
    OUTPUTS:
        figs (list): list of figures of each face and its prediction
        big_fig (figure): the initial image itself with all its annotations
    '''
    #load the image using pillow
    test_image = load_and_resize_image(test_image_path, max_dim=2048)

    #extract multiple faces from the image
    test_image_tensors, face_bounds = extract_multiple_faces(test_image, detector, 
                                      margin = 20, resize_shape = 160, return_box = True, 
                                      face_prob_thresh = .95, min_face_size = 30, device = device)
    #initialize lists for the figures and predictions
    figs, preds = [], []
    #if there was no face or it was not rgb, then we will return the error rather than plot
    if test_image_tensors[0] == 'No faces found' or test_image_tensors[0] == 'Image not RGB':
        assert len(test_image_tensors) == 1  #if no faces found, should have this error only
        figs.append(test_image_tensors[0])
        big_fig = None
    #if we had atleast one face located, then lets either show error or predict face for each
    else:
        for test_image_tensor in test_image_tensors:
            if torch.is_tensor(test_image_tensor):
                #convert the tensor to an image for plotting the face later
                test_image_face = tensor_to_img(test_image_tensor)
                test_image_tensor = test_image_tensor.to(device)

                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    #pass the face tensor to the embeddor to get the 1792 dim embedding
                    embedding = embeddor.forward_1792(test_image_tensor) 
                    
                    #now predict the age from the embedding using our trained network
                    output = model(embedding).cpu().float()
                    
                #take the expectation of the probability on each age for prediction
                pred, output_softmax = fm.predict_age(output, classes = range(10,71))
                
                #now plot the image and its prediction side by side
                filename = '.'.join(test_image_path.split('/')[-1].split('.')[:-1])
                fig = fpl.plot_image_and_pred(test_image_face, pred, CLASSIF = True, 
                                    output_softmax = output_softmax, image_name = filename,
                                    tight_layout = tight_layout, figshow = figshow)
                test_image_face.close() #close the image to prevent memory leak
                #keep track of predictions and figures to pass out later
                preds.append(pred)
                figs.append(fig)
            else:
                #if the face was not sufficient quality to predict on, we keep 
                #track of the error messages
                preds.append(test_image_tensor) 
                figs.append(test_image_tensor)
                
        #make a big plot with mini annots on all the faces, as long as there were not no faces found
        big_fig, ax = fpl.overlay_preds_on_img(test_image, face_bounds, preds)
        test_image.close() #close the image to prevent memory leak

    return figs, big_fig

def load_and_resize_image(path, max_dim=2048):
    """
    Load an image from disk, apply EXIF orientation, 
    and resize so the largest dimension is <= max_dim.
    """
    with Image.open(path) as img:   # ensures file handle closes after reading
        # Fix orientation (handles sideways images)
        img = ImageOps.exif_transpose(img)

        # Resize if needed
        w, h = img.size
        scale = max(w, h) / max_dim
        if scale > 1:
            new_w, new_h = int(w / scale), int(h / scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return img.copy() # return a safe, detached image

def random_sample(root_path, desired_samples, paste_new_dir = False, name_new_dir = None):
    ''' This function is useful for sampling a vector of specified ages. This helps 
        create unbiased samples for creating tests sets or investigating cleanliness 
        of data.
    INPUTS:
        root_path (str): path to the folder containing the age subfolders
        desired_samples (array or int): if array, then will sample one image from each of
                            the ages in the array (ages must be [10,70]), if int then 
                            will randomly sample from n randomly generated ages
        paste_new_dir (bool): if True, then it will take the sampled pictures and copy
                            them to the directory specified   
        name_new_dir (str): will name the sample dir for easier referencing later if you want
    OUTPUTS:
        samples (list): list of PIL images 
    '''
    
    if isinstance(desired_samples, int):
        desired_samples = np.random.randint(10,71,desired_samples)
    
    #We want to create a new directory to put these random samples in
    #this is helpful for making the survey, and for checking errors
    if paste_new_dir:
        num_samples_already = len(os.listdir('data/samples'))
        samples_dump_path = 'data/samples/sample' + str(num_samples_already) + \
            '_size' + str(len(desired_samples))
        samples_dump_path += '' if name_new_dir is None else '_' + name_new_dir
        os.mkdir(samples_dump_path)
        
    #now sampleee
    samples_img = []
    samples_tensor = [] 
    sample_names = []
    for age in desired_samples:
        #open that ages subfolder and choose a random img to copy
        img_list = os.listdir(root_path + '/' + str(int(age)))
        sample_name = random.choice(img_list)
        #prevent repeated sampling
        while sample_name in sample_names:         
            sample_name = random.choice(img_list)
        sample_names.append(sample_name)
        sample_path = root_path + '/' + str(int(age)) + '/' + sample_name
        img = Image.open(sample_path)
        samples_img.append(img)
        samples_tensor.append(img_to_tensor(img))
        
        if paste_new_dir:
            shutil.copy(sample_path, samples_dump_path + '/' + sample_name)
    
    return samples_img, samples_tensor

def make_subfolders(path_above, folder_names):
    #loop through all the folder names and add them to the end of the path_above
    for folder in folder_names:
        os.mkdir(os.path.join(path_above, str(folder)))
            
def img_to_tensor(img):
    #convert image to tensor and scale to be [-1,1]
    np_array = (np.asarray(img).astype(np.float32) - 127.5) / 128
    return torch.as_tensor(np_array.transpose(2,0,1), dtype = torch.float32)
    
def tensor_to_img(tensor):
    #convert tensor back to PIL image [0, 255]
    if torch.max(tensor) <= 1:
        tensor = (tensor + 1) * (255/2)
    return Image.fromarray(tensor.squeeze().permute(1,2,0).numpy().astype(np.uint8))

def np_to_tensor(np_array):
    #permutes the axis to chan first, and converts to np
    return torch.as_tensor(np.moveaxis(np_array.squeeze(), 2, 0), dtype = torch.float32)

def tensor_to_np(tensor):
    #permutes axis to chan last, converts to np
    return np.moveaxis(tensor.cpu().numpy().squeeze(), 0, 2).astype(np.float32)
    
def box_to_square(bounding_box):
    #bounding_box is [x1, y1, x2, y2] locations of bounding box indices in image
    #this function is to make the bounding box square by extending the smaller
    #of the width or the height
    
    x1, y1, x2, y2 = bounding_box
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    
    max_side = np.max((h,w))

    #if max side is w then w*.5 - max_side*.5 = 0, so doesn't affect
    #so we can vectorize if needed
    bounding_box[0] = np.round(x1 + w*.5 - max_side*.5)
    bounding_box[1] = np.round(y1 + h*.5 - max_side*.5)

    #now that x1 and y1 is adjusted, can use them to adjust to the x2, y2
    #just use the max side to index a square beyond them
    addition = max_side - 1.0
    if addition % .5 == 0: #prevent stupid numpy rounding down
        addition += .00001
        
    bounding_box[2] = bounding_box[0] + np.round(addition)
    bounding_box[3] = bounding_box[1] + np.round(addition)
    
    bounding_box = np.array(bounding_box, dtype=np.float32)
    
    return bounding_box

def box_add_margin(face_bounds, margin, img_w, img_h, clamp = False):
    #function to take the face bounds and add margin to them, margin is assumed to be pct if > 1
    if margin > 1:
        margin = margin/100
    #calc w and h of the sqaure input box
    w, h = (face_bounds[2] - face_bounds[0], face_bounds[3] - face_bounds[1])
    assert w == h, 'img must be square by now after box_to_square'
    #now add the margin to the face bounds
    face_bounds[0] = face_bounds[0] - margin*w
    face_bounds[1] = face_bounds[1] - margin*h
    face_bounds[2] = face_bounds[2] + margin*w
    face_bounds[3] = face_bounds[3] + margin*h
    
    #clamp indices to be within the bonds of the photo
    if clamp:
        face_bounds[0] = np.max((0, face_bounds[0]))
        face_bounds[1] = np.max((0, face_bounds[1]))
        face_bounds[2] = np.min((img_w, face_bounds[2]))
        face_bounds[3] = np.min((img_h, face_bounds[3]))
        
    return face_bounds
    
def align_img(img_tensor, face_bounds, landmarks, img_w, img_h):
    #will rotate image about center of face bounds to 
    #make the eyes be level so the face is not at an angle
    dx = (landmarks[1,0] - landmarks[0,0])
    dy = (landmarks[1,1] - landmarks[0,1])
    angle = np.degrees(np.arctan2(dy,dx))
    midpoint = tuple(np.mean(face_bounds.reshape(2,2), axis = 0))
    M = cv2.getRotationMatrix2D(midpoint, angle, 1)
    img_np = tensor_to_np(img_tensor)
    img_np_rot = cv2.warpAffine(img_np, M, (img_w, img_h))
    img_tensor_rot = np_to_tensor(img_np_rot)
    return img_tensor_rot

def crop_face(img_tensor, face_bounds, img_h, img_w, max_dim, resize_shape):
    #take in an image and pad it with zeros if index is outside the image edges
    if face_bounds[0] < 0 or face_bounds[1] < 0 or face_bounds[2] > img_w or face_bounds[3] > img_h:
        pad_dim = int(max_dim)
        image_padded = torch.zeros((3, int(img_h+2*pad_dim), int(img_w+2*pad_dim)))
        #now assign image to middle of the padded
        image_padded[:, pad_dim:pad_dim+img_h, pad_dim:pad_dim+img_w] = img_tensor.detach().clone() #break relationship with x
        #now crop out from the padded tensor, and resize to 24x24
        face_crop = image_padded[:, pad_dim+face_bounds[1]:pad_dim+face_bounds[3], \
                                pad_dim+face_bounds[0]:pad_dim+face_bounds[2]+1].unsqueeze(0)
    else:
        #if index is safely in image, just simple index needed
        face_crop = img_tensor[:, face_bounds[1]:face_bounds[3], face_bounds[0]:face_bounds[2]].unsqueeze(0)
    #last, reshape to correct size to feed ubti the nn
    face_crop = nn.functional.interpolate(face_crop, size=(resize_shape, resize_shape),
                                          mode='bilinear', align_corners=False)
    return face_crop

def img_to_img_and_tensor(img):
    #function takes in several possible datatypes for image and outputs a pil image and tensor
    #this is a helper function for extracting faces
    
    if torch.is_tensor(img):
        img_tensor = img
        img = tensor_to_img(img)
    elif isinstance(img, np.ndarray):
        if img.shape[0] != 3:
            img = np.moveaxis(img, -1, 0)
        assert img.shape[0] == 3, 'Must put color in first or last channel'  #make sure channels are proper
        img_tensor = torch.from_numpy(img) 
        img = tensor_to_img(img_tensor)
    elif isinstance(img, str):
        img = Image.open(img)
        img_tensor = img_to_tensor(img)
    else: #pil image
        img_tensor = img_to_tensor(img)
    return img, img_tensor

def check_multiple_faces(face_probs, face_bounds, verbose):
    #function to check if there are multiple faces in an image. if one faces is
    #60% larger than the other face we will still go ahead and use it.
    
    #check if face is by far the most defined in the photo (to be sure age is correct)
    strong_single = face_probs[0] > .05 + face_probs[1]
    #check if other faces are tiny and in the background, then it is permissable
    #find 20pct of size of face in main pic, if other face is less than 60pct
    #of the main face then we let it slide
    pct_of_face = .6*np.mean(face_bounds[0][2:4] - face_bounds[0][0:2])
    background = np.mean(face_bounds[1][2:4] - face_bounds[1][0:2]) < pct_of_face
    #if strong single or other face is in background, then it is ok to use image, if neither then not ok to use
    unique_face = strong_single or background
    if not(unique_face):
        if verbose >= 2:
            print('Num boxes when failed unique: ', face_bounds.shape[0], 'boxs', face_bounds, 'conf', face_probs)
    return unique_face

def extract_face_helper(img_tensor, face_prob, face_bounds, landmarks, margin, img_w, img_h, 
                     face_prob_thresh, min_face_size, resize_shape, verbose, image_path):
    ''' Take the img_tensor, face_prob, face_bounds, and landmarks and further carry out
        the repeatable part of the extracting the face. This is broken out into a 
        helper function because it is duplicated for extract one face and
        extract multiple faces.
    INPUTS:
        img_tensor (torch.tensor): the image tensor to extract the face from
        face_prob (float): single probability that the face is a face
        face_bounds (np.array): x1, y1, x2, y2 for the face bounding box
        landmarks (np.array): landmarks for the eyes, nose, mouth (left first then right)
        margin (int): the percent amount of margin to add to the face, 20 will 
                    increase the face bounds by 20% of their original size
        img_w, img_h (int): the height and width of the original input image
        face_prob_thresh (float): the minimum probability to say yes to potential 
                                face in an image found by the detector
        min_face_size (int): minimum number of pixels on the minimum dimension of the
                            bounding box to be acceptable to the model as enough information
        resize_shape (int): the number of pixels to resize the square cropped image to
                            this should be the size that the model expects
        verbose (int): control level of output
        image_path (str): image path for traceable error messages
    OUTPUTS:
        face_tensor (str or torch.tensor): the extracted face as a tensor or an 
                            error message as to why the face was not used or not
                            even found
    '''
    #get nose x loc for testing face rotation
    nose_x = landmarks[2,0]
    eye_pad = .1*(landmarks[1,0] - landmarks[0,0])
    #make sure nose between eyes in x dim
    if not(nose_x > landmarks[0,0] - eye_pad and nose_x < landmarks[1,0] + eye_pad):
        if verbose >= 1:
            Warning('Side profile face, nose outside of eyes in image', image_path)
        return 'not_front_facing'
    #check if face is confident enough to use well
    if face_prob < face_prob_thresh:
        if verbose >= 1:
            Warning('Face not confident enough in image', image_path)
        return 'insufficient_confidence'
    #check if face is large enough to use well
    w, h = (face_bounds[2] - face_bounds[0] + 1, face_bounds[3] - face_bounds[1] + 1)
    min_dim = np.min((w, h))
    max_dim = np.max((w,h))
    if min_dim < min_face_size:
        if verbose >= 1:
            Warning('Face not large enough in image', image_path)
        return 'insufficient_resolution'
    #made it through first checks, now preproc/standardize the face
    face_bounds = box_to_square(face_bounds)
    face_bounds = box_add_margin(face_bounds, margin, img_w, img_h)
    #check if going to index more than half off the image, then just too far to be valuable data
    if face_bounds[3] < 1.4*img_h and face_bounds[2] < 1.4*img_w and \
        face_bounds[1] > -.4*img_h and face_bounds[0] > -.4*img_w:
        
        #so now we actually care about the image to the point we will spend time rotating and such
        face_bounds = np.round(face_bounds).astype(np.int32)
        #before crop, rotate img tensor about midpoint of face bounds
        #this level the eyes and help balance all the faces
        img_tensor_rot = align_img(img_tensor, face_bounds, landmarks, img_w, img_h)
        #now that img is rotated about center, finally crop it out
        face_tensor = crop_face(img_tensor_rot, face_bounds, img_h, img_w, max_dim, resize_shape)
        if verbose >= 2:
            print('Face found, conf:', face_prob, 'size:', max_dim)    
            plt.figure()
            plt.imshow(tensor_to_img(face_tensor))
            plt.title('Face extraction for ' + image_path)
            
        return face_tensor
    else:
        if verbose >= 1:
            Warning('Not enough margin in image', image_path)
        return 'not_enough_margin'

def safe_detect(detector, img, max_size=1024):
    #function to safely detect faces in large images by resizing them first
    #if the image is larger than max_size in either dim, then resize it first
    #before passing to the detector, then scale back up the boxes and landmarks
    with torch.no_grad():
        w, h = img.size
        scale = max(w, h) / max_size
        if scale > 1:
            new_w, new_h = int(w / scale), int(h / scale)
            img_resized = img.resize((new_w, new_h))
            boxes, probs, landmarks = detector.detect(img_resized, landmarks=True)
            if boxes is not None:
                boxes *= scale   # scale back up to original coords
                if landmarks is not None:
                    landmarks *= scale
            return boxes, probs, landmarks
        else:
            return detector.detect(img, landmarks=True)

def extract_one_face(img, detector, margin, resize_shape = 160, min_face_size = 40,
                 verbose = 0, image_path = '', face_prob_thresh = .9, 
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ''' Function to take in an image and output an extracted face or a flag as to 
        why there wasn't a face to extract. For use in the data cleaning.
    INPUTS:
        img (str, pil image, torch.tensor, or numpy.array): the image to extract the face from
        detector (object): the face bounding box detector
        margin (int): the percent amount of margin to add to the face, 20 will 
                    increase the face bounds by 20% of their original size
        resize_shape (int): the number of pixels to resize the square cropped image to
                            this should be the size that the model expects
        min_face_size (int): minimum number of pixels on the minimum dimension of the
                            bounding box to be acceptable to the model as enough information
        image_path (str): image path for traceable error messages
        face_prob_thresh (float): the minimum probability to say yes to potential 
                                face in an image found by the detector
        device (str): whether we are calculating on gpu or cpu
    OUTPUTS:
        face_tensor (str or torch.tensor): the extracted face as a tensor or an 
                            error message as to why the face was not used or not
                            even found
    '''
    #accepts pil img or torch tensor or path to image
    img, img_tensor = img_to_img_and_tensor(img)
    img_tensor = img_tensor.to(device)
    img_h, img_w = img_tensor.shape[-2:]
    #detect faces and landmarks in image
    face_bounds, face_probs, landmarks = safe_detect(detector, img, max_size=1600)
    nface = face_bounds.shape[0] if face_bounds is not None else 0
    #if we have atleast 1 face, then lets extract it
    if nface > 0:
        #check if the face is the only prominent face in the image
        unique_face = check_multiple_faces(face_probs, face_bounds, verbose)
        if unique_face:
            #filter out faces that are angled too much or are side profile
            #this is if nose left of left eye or right of right eye
            face_bounds, face_prob, landmarks = face_bounds[0], face_probs[0], landmarks[0]
            face_tensor = extract_face_helper(img_tensor, face_prob, face_bounds, 
                                    landmarks, margin, img_w, img_h, face_prob_thresh, 
                                    min_face_size, resize_shape, verbose, image_path)
            return face_tensor
        else:
            if verbose >= 1:
                Warning('Face not the only face in image', image_path)
            return 'not_unique'
    else:
        if verbose >= 1:
            Warning('No faces found in image', image_path)
        return 'not_found'

def extract_multiple_faces(img, detector, margin, resize_shape = 224, min_face_size = 40,
                 verbose = 0, image_path = '', face_prob_thresh = .95, return_box = False,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ''' Function to take in an image and output extracted faces or flags as to 
        why the face wasnt acceptable. For use in the website that classifies 
        multiple faces in one image.
    INPUTS:
        img (str, pil image, torch.tensor, or numpy.array): the image to extract the face from
        detector (object): the face bounding box detector 
        margin (int): the percent amount of margin to add to the face, 20 will 
                    increase the face bounds by 20% of their original size
        resize_shape (int): the number of pixels to resize the square cropped image to
                            this should be the size that the model expects
        min_face_size (int): minimum number of pixels on the minimum dimension of the
                            bounding box to be acceptable to the model as enough information
        image_path (str): image path for traceable error messages
        face_prob_thresh (float): the minimum probability to say yes to potential 
                                face in an image found by the detector
        return_box (bool): whether or not to return the bounding boxes as well
                            this is used for annotating images
        device (str): whether we are calculating on gpu or cpu
    OUTPUTS:
        face_tensor (str or torch.tensor): the extracted face as a tensor or an 
                            error message as to why the face was not used or not
                            even found
    '''
    #gets multiple faces and returns list of faces or list of errors
    face_tensors = []
    bounding_boxes = []
    if img.mode == 'RGB':
        #accepts pil img or torch tensor or path to image
        img, img_tensor = img_to_img_and_tensor(img)
        img_tensor = img_tensor.to(device)
        img_h, img_w = img_tensor.shape[-2:]
        #detect faces in image
        face_bounds, face_probs, landmarks = detector.detect(img, landmarks=True)
        nface = face_bounds.shape[0] if face_bounds is not None else 0
        if nface > 0:
            assert face_bounds.ndim == 2, 'should be dim 1 of num faces and dim 2 of the 4 box coords'
            for i in range(nface):
                #for each face in all possible faces see if they qualify
                face_bounds_, face_prob_, landmarks_ = face_bounds[i], face_probs[i], landmarks[i]
                face_tensor = extract_face_helper(img_tensor, face_prob_, face_bounds_, 
                                    landmarks_, margin, img_w, img_h, face_prob_thresh, 
                                    min_face_size, resize_shape, verbose, image_path)
                #append the face bounds and tensors to lists for later plotting
                bounding_boxes.append(face_bounds_)
                face_tensors.append(face_tensor)
        else:
            if verbose >= 1:
                Warning('No faces found', image_path)
            face_tensors.append('No faces found')
            bounding_boxes.append(None)
    else:
        if verbose >= 1:
            Warning('Image not RGB')
        face_tensors.append('Image not RGB')
        bounding_boxes.append(None)
        
    if return_box:
        return face_tensors, bounding_boxes
    else: 
        return face_tensors
    
class Data:
    def __init__(self, train_folder, val_folder, test_folder, weight_thresh = 10, 
                 num_workers = 0, batch_size = 64, img_size = 160):
        '''
        Class that keeps all data stuff in one place.

        Parameters
        ----------
        train_folder : str
            The path to the training data
        val_folder : str
            The path to the validation data
        test_folder : str
            The path to the test data
        weight_thresh : int, optional
            The number to divide the max of the distribution by when evening out classes. 
            We clamp less than 1, so by driving most below 1 a higher number 
            will result in more even sampling at the risk of oversampling sparse ages
            and overfitting to examples in them. A low number will cause a lot
            of weights to be clamped and thus have not as even sampling. The default is 10.
            10 will get the least freq classes to be about 50% of the most frequent.
            (raw is 400 compared to 8,000)
        num_workers : int, optional
            num cores for loading data. The default is 0.
        batch_size : int, optional
            num examples per training step. The default is 64.
        img_size : int, optional
            size to resize pictures to before sending into model. The default is 160.

        Returns
        -------
        Creates properties that can be referenced later. Keeps all data stuff in one place.
        
        '''
        self.num_workers = num_workers              # number of subprocesses to use for data loading
        self.batch_size = batch_size                # how many samples per batch to load
        self.img_size = img_size                    #img_size to make square images
        self.weight_thresh = weight_thresh          #the number to divide the max of the distribution by when evening out classes
        self.train_folder_dir = train_folder
        self.val_folder_dir = val_folder
        self.test_folder_dir = test_folder
        
    def create_loaders(self, image = True):
        if image:
            train_transform = transforms.Compose([
                                transforms.RandomRotation(5),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop(self.img_size, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
                                transforms.ToTensor(),
                                ])
            
            test_transform = transforms.Compose([
                                transforms.Resize([self.img_size,self.img_size]),
                                transforms.ToTensor(),
                                ])
            
            train_data = datasets.ImageFolder(self.train_folder_dir, train_transform)
            val_data   = datasets.ImageFolder(self.val_folder_dir, train_transform)
            test_data  = datasets.ImageFolder(self.test_folder_dir, test_transform)
        else:
            #then is the tensor embeddings
            def loader_fcn(path):
                return torch.load(path).squeeze()
            loader_fcn_input = loader_fcn
            train_data = datasets.DatasetFolder(self.train_folder_dir, loader_fcn_input, '.pt')
            val_data   = datasets.DatasetFolder(self.val_folder_dir, loader_fcn_input, '.pt')
            test_data  = datasets.DatasetFolder(self.test_folder_dir, loader_fcn_input, '.pt')
        
        #store data on classes for later use
        self.num_classes = len(train_data.classes)
        self.classes = train_data.classes
        
        #calculate number for each class to then calculate weights for sampler
        class_count = np.bincount(train_data.targets).tolist()
        freq_thresh = np.max(class_count)/self.weight_thresh                    #take a tenth of the mode to normalize off of (that way dont insanely oversample the edge ages)
        train_weights = freq_thresh/torch.tensor(class_count, dtype=torch.float)
        train_weights = torch.clamp(train_weights, 0, 1)                        #now middle classes will be less but not tooo much less
        
        #calculate train, val, and test weighted sampler using train weights
        train_sample_weights = train_weights[train_data.targets]
        val_sample_weights = train_weights[val_data.targets]
        test_sample_weights = train_weights[test_data.targets]
        
        #make the samplers using the calculated weight vectors
        train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_data), replacement=True)
        val_sampler   = WeightedRandomSampler(weights=val_sample_weights, num_samples=len(val_data), replacement=True)
        test_sampler  = WeightedRandomSampler(weights=test_sample_weights, num_samples=len(test_data), replacement=True)
        
        #creating data loaders
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size = self.batch_size,
                                    sampler = train_sampler, num_workers = self.num_workers)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size = self.batch_size,
                                    sampler = val_sampler, num_workers = self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size = self.batch_size,
                                    sampler = test_sampler, num_workers = self.num_workers)
        
