# -*- coding: utf-8 -*-
"""
Runtime model initialization helpers.
"""
from functools import lru_cache

import torch
from facenet_pytorch import MTCNN

import FaceModels as fm


@lru_cache(maxsize=1)
def get_runtime_models():
    """
    Lazily initialize and cache detector/embeddor/model for inference.
    """
    device = torch.device('cpu')
    detector = MTCNN(min_face_size=30, device=device)
    embeddor = fm.Facenet_Embeddor(device=device).eval().to(device)
    model = fm.Ensemble_Model(device=device).eval().to(device)
    return detector, embeddor, model, device
