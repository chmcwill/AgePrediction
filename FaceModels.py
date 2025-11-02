# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:19:49 2020

@author: Cameron
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms, utils
import torch.optim as optim
from facenet_pytorch import MTCNN, InceptionResnetV1

def predict_age(model_out, classes = range(10,71), thresh = .015):
    #model out is the outputs nodes for each data point in the batch 
    #so will be of shape batch_size by n_classes
    #use expectation over all classes to predict age
    softmax = torch.nn.Softmax(dim=1)
    softmax_out = softmax(model_out).detach().cpu().numpy()
    #zero out values with small prob to avoid skewing
    #first index values greater than and get sum to scale back to area under curve 1
    greater_than_mask = np.zeros_like(softmax_out)
    greater_than_mask[softmax_out >= thresh] = 1
    sum_greater_than = np.sum(np.multiply(greater_than_mask, softmax_out), axis=1) #sum along columns to get sum for each element in batch
    #now zero out values with small probability to avoid skewing
    softmax_out[softmax_out < thresh] = 0
    #now dot product to find expectation
    ages_all = np.array(classes).astype(np.float32)
    pred = np.round(np.dot(softmax_out, ages_all)/sum_greater_than,1)           #scale back by dividing by sum (np will broadcast elementwise)
    return pred, softmax_out

def calc_mean_abs_error(predictions, targets):
    #predictions and targets are both tensors of same length (usually length batchsize)
    diff = predictions - targets
    mae = np.mean(np.abs(diff))
    return mae

class Facenet_Embeddor(torch.nn.Module):
    #to take the facenet and embed
    def __init__(self, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Facenet_Embeddor, self).__init__()
        self.name = 'Facenet_Embeddor'
        #loading our own saved weights to ensure embeddings stay the same over time
        self.resnet = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631).eval().to(device)
        self.resnet.load_state_dict(torch.load(
            'best_models/inception_resnet_v1_vggface2.pth', 
            map_location=device))

        features_list = list(self.resnet.children())
        
        self.embed_1792 = nn.Sequential(*features_list[:-4]).requires_grad_(False)
        
    def forward_1792(self, x):
        x = self.embed_1792(x)
        return x.view(x.shape[0],-1)
    
    def forward_512(self, x):
        return self.resnet(x)

class Facenet_Model(nn.Module):
    #class to use the inception net specifically trained on vggface2 so that
    #the features are important for faces
    #we then use our own nn to calculate the prediction from the embeddings
    
    def __init__(self, num_outputs, pretrained_data = 'vggface2',
                 drop_pct = .25, freeze_resnet = True):
        super(Facenet_Model, self).__init__()
        
        if pretrained_data == 'vggface2': #keep these stable across package updates
            self.inception_resnet = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631)
            self.resnet.load_state_dict(torch.load('best_models/inception_resnet_v1_vggface2.pth', map_location='cpu'))
        else:
            self.inception_resnet = InceptionResnetV1(pretrained=pretrained_data)


        #freeze resnet, we only will train the prediction layer
        if freeze_resnet:
            self.inception_resnet = self.inception_resnet.eval()
            for param in self.inception_resnet.parameters():
                param.requires_grad = False
        self.name = 'Inception_CustomEnd'
        self.fc = nn.Sequential(nn.BatchNorm1d(512),
                                nn.Linear(512, 256),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 128),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct),
                                nn.BatchNorm1d(128),
                                nn.Linear(128, 64),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct),
                                nn.BatchNorm1d(64),
                                nn.Linear(64, num_outputs))

    def forward(self, x):
        #first compute the 512 embedding from inception resnet
        x = self.inception_resnet(x)
        
        #then pass the embedding through some fc layers to learn age
        out = self.fc(x)
        
        return out
    
class Ensemble_Model(nn.Module):
    #class to aggregate several aready trained NNs
    def __init__(self, model_list = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Ensemble_Model, self).__init__()
        self.name = 'Ensemble_Model'
        self.device = device
        if model_list is None:
            model_list = []
            #model_list.append(torch.load('best_models/FC_model_1792_classif_model_agepred_' + \
            #                             'bestval_personal_mae_5.614_bias_balanced_Nov-03-2020.pt'))
            #model_list.append(torch.load('best_models/FC_model_1792_classif_model_agepred_' + \
            #                             'bestval_personal_mae_5.498_bias_older_Nov-05-2020.pt'))
            model_list.append(torch.load('best_models/FC_model_1792_classif_model_agepred_' + \
                                         'bestval_personal_mae_5.406_bias_younger_Nov-04-2020.pt',
                                         weights_only=False, map_location=device))
            model_list.append(torch.load('best_models/FC_model_1792_classif_model_agepred' + \
                                         '_bestval_personal_mae_5.266_bias_balanced_Nov-10-2020.pt',
                                         weights_only=False, map_location=device))
            model_list.append(torch.load('best_models/FC_model_1792_classif_model_agepred' + \
                                         '_bestval_personal_mae_5.164_bias_balanced_Nov-10-2020.pt',
                                         weights_only=False, map_location=device))
            #model_list.append(torch.load('best_models/FC_model_1792_classif_model_agepred' + \
            #                             '_bestval_personal_mae_5.161_bias_balanced_Nov-09-2020.pt'))
            #model_list.append(torch.load('best_models/FC_model_1792_classif_model_agepred' + \
            #                             '_bestval_personal_mae_5.182_bias_balanced_Nov-09-2020.pt'))
            
            self.model_list = [model.to(device).eval() for model in model_list]
        else:
            self.model_list = [model.to(device).eval() for model in model_list] #make sure on device and eval
        
    def forward(self, x):
        #average the logits of all models
        logits = []
        for model in self.model_list:
            logits.append(model(x))
        #stack committee logits
        logits = torch.stack(logits)
        #average across the committee
        logits = logits.mean(0)
        return logits
        
class FC_model_1792(nn.Module):
    #simple fc layer nn
    def __init__(self, num_outputs, drop_pct = .25):
        super(FC_model_1792, self).__init__()
        self.name = 'FC_model_1792'
        self.fc = nn.Sequential(nn.Dropout(drop_pct+.1),
                                nn.BatchNorm1d(1792),
                                nn.Linear(1792, 512),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct+.05),
                                nn.BatchNorm1d(512),
                                nn.Linear(512, 256),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 128),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct),
                                nn.BatchNorm1d(128),
                                nn.Linear(128, 64),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct-.05),
                                nn.BatchNorm1d(64),
                                nn.Linear(64, num_outputs))
    def forward(self, x):
        return self.fc(x)

class FC_model_512(nn.Module):
    #simple fc layer nn    
    def __init__(self, num_outputs, drop_pct = .25):
        super(FC_model_512, self).__init__()
        self.name = 'FC_model_512'
        self.fc = nn.Sequential(nn.BatchNorm1d(512),
                                nn.Linear(512, 256),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 128),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct),
                                nn.BatchNorm1d(128),
                                nn.Linear(128, 64),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_pct),
                                nn.BatchNorm1d(64),
                                nn.Linear(64, num_outputs))
    def forward(self, x):
        return self.fc(x)
    
def define_model(model_use, classif, lr, num_classes, verbose = 0):
    if not classif:
        assert num_classes == 1
    if classif:
        criterion = nn.CrossEntropyLoss(reduction = 'mean')
        if model_use == 'inceptionV1frozen':
            model = Facenet_Model(num_classes)
            optimizer = optim.Adam(model.fc.parameters(), lr) #only pass the fc layer params
        elif model_use == 'inceptionV1':
            model = Facenet_Model(num_classes, freeze_resnet = False)
            optimizer = optim.Adam(model.parameters(), lr) #only pass the fc layer params
        elif model_use == 'FC_model_1792':
            model = FC_model_1792(num_classes)
            optimizer = optim.Adam(model.parameters(), lr)
        elif model_use == 'vgg16':
            model = models.vgg16(pretrained=True)
            if verbose >= 1:
                print('Original vgg16 architecture:')
                print(model)
            # Freeze training for all "features" layers
            for param in model.features.parameters():
                param.requires_grad = False
            #customize the end of the model
            model.classifier = nn.Sequential(
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(512 * 7 * 7),
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(4096),
                    nn.Linear(4096, 1024),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(1024),
                    nn.Linear(1024, 512),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, num_classes),
                )    
            optimizer = optim.Adam(model.classifier.parameters(), lr)
        elif model_use == 'resnet18':
            model = models.resnet18(pretrained=True)
            print('Original resnet18 architecture:')
            print(model)
            # Freeze training for all "features" layers
            for param in model.parameters():
                param.requires_grad = False
            # trying for the resnet (may want to unfreeze some conv layers)
            # Now add these fc layers, they will add with req grad True
            #input to avg pool is 512x7x7
            model.avgpool = nn.AdaptiveAvgPool2d(output_size = (3,3))
            model.fc = nn.Sequential(
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(4608), #512x3x3
                    nn.Linear(4608, 1024),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(1024),
                    nn.Linear(1024, 512),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(512),    
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, num_classes),
                )
            optimizer = optim.Adam(model.fc.parameters(), lr)
        else:
            raise Exception('Invalid model_use')
    else:
        criterion = nn.L1Loss(reduction='mean')
        if model_use == 'inceptionV1frozen':
            model = Facenet_Model(num_classes)
            optimizer = optim.Adam(model.fc.parameters(), lr) #only pass the fc layer params
        elif model_use == 'inceptionV1':
            model = Facenet_Model(num_classes, freeze_resnet = False)
            optimizer = optim.Adam(model.parameters(), lr) #only pass the fc layer params
        elif model_use == 'FC_model_1792':
            model = FC_model_1792(num_classes)
            optimizer = optim.Adam(model.parameters(), lr)
        elif model_use == 'vgg16':
            model = models.vgg16(pretrained=True)
            print('Original vgg16 architecture:')
            print(model)
            # Freeze training for all "features" layers
            for param in model.features.parameters():
                param.requires_grad = False
            #customize the end of the model
            model.classifier = nn.Sequential(
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(512 * 7 * 7),
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(4096),
                    nn.Linear(4096, 1024),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(1024),
                    nn.Linear(1024, 512),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, 1),
                )    
            optimizer = optim.Adam(model.classifier.parameters(), lr)
        elif model_use == 'resnet18':
            model = models.resnet18(pretrained=True)
            print('Original resnet18 architecture:')
            print(model)
            # Freeze training for all "features" layers
            for param in model.parameters():
                param.requires_grad = False
            # trying for the resnet (may want to unfreeze some conv layers)
            # Now add these fc layers, they will add with req grad True
            #input to avg pool is 512x7x7
            model.avgpool = nn.AdaptiveAvgPool2d(output_size = (3,3))
            model.fc = nn.Sequential(
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(4608), #512x3x3
                    nn.Linear(4608, 1024),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(1024),
                    nn.Linear(1024, 512),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(512),    
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    nn.Dropout(p=.3),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 1),
                )
            optimizer = optim.Adam(model.fc.parameters(), lr)
        else:
            raise Exception('Invalid model_use')
    model.name = model_use + '_classif' if classif else model_use + '_regression' 
    return model, criterion, optimizer