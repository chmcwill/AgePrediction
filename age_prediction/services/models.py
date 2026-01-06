# -*- coding: utf-8 -*-
"""
Runtime model initialization helpers and lightweight inference-only model definitions.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1, MTCNN

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BEST_MODELS_DIR = ROOT_DIR / "best_models"


def _load_state_dict(filename: str, device: torch.device) -> dict:
    """Load a state dict checkpoint from best_models."""
    path = BEST_MODELS_DIR / filename
    return torch.load(path, map_location=device)


def predict_age(model_out: torch.Tensor, classes: Iterable[int], thresh: float = 0.015) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expectation over class probabilities to estimate age.
    """
    softmax_out = torch.nn.functional.softmax(model_out, dim=1).detach().cpu().numpy()
    greater_than_mask = np.zeros_like(softmax_out)
    greater_than_mask[softmax_out >= thresh] = 1
    sum_greater_than = np.sum(np.multiply(greater_than_mask, softmax_out), axis=1)
    softmax_out[softmax_out < thresh] = 0
    ages_all = np.array(list(classes)).astype(np.float32)
    pred = np.round(np.dot(softmax_out, ages_all) / sum_greater_than, 1)
    return pred, softmax_out


class Facenet_Embeddor(torch.nn.Module):
    """
    Thin wrapper around InceptionResnetV1 to emit 1792-d embeddings with fixed weights.
    """

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.name = "Facenet_Embeddor"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.resnet = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631).eval().to(device)
        self.resnet.load_state_dict(_load_state_dict("inception_resnet_v1_vggface2.pth", device=device))

        features_list = list(self.resnet.children())
        self.embed_1792 = nn.Sequential(*features_list[:-4]).requires_grad_(False)

    def forward_1792(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.embed_1792(x)
        return x.view(x.shape[0], -1)

    def forward_512(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class FCModel1792(nn.Module):
    """
    Lightweight fully-connected head for 1792-d embeddings (classification).
    """

    def __init__(self, num_outputs: int, drop_pct: float = 0.25):
        super().__init__()
        self.name = "FC_model_1792"
        self.fc = nn.Sequential(
            nn.Dropout(drop_pct + 0.1),
            nn.BatchNorm1d(1792),
            nn.Linear(1792, 512),
            nn.LeakyReLU(),
            nn.Dropout(drop_pct + 0.05),
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
            nn.Dropout(drop_pct - 0.05),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Ensemble_Model(nn.Module):
    """
    Simple ensemble that averages logits from a list of pre-trained FC heads.
    """

    def __init__(self, classes: Iterable[int], model_list=None, device: torch.device | None = None):
        super().__init__()
        self.name = "Ensemble_Model"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        classes_tuple = tuple(classes)

        if model_list is None:
            # Prefer loading weights into a known architecture to avoid pickled modules.
            num_outputs = len(classes_tuple)
            weight_files = [
                "FC_model_1792_classif_weights_agepred_bestval_personal_mae_5.406_bias_younger_Nov-04-2020.pt",
                "FC_model_1792_classif_weights_agepred_bestval_personal_mae_5.266_bias_balanced_Nov-10-2020.pt",
                "FC_model_1792_classif_weights_agepred_bestval_personal_mae_5.164_bias_balanced_Nov-10-2020.pt",
            ]
            model_list = []
            for wf in weight_files:
                state_dict = _load_state_dict(wf, device=device)
                model = FCModel1792(num_outputs=num_outputs)
                model.load_state_dict(state_dict)
                model_list.append(model)

        self.model_list = [model.to(device).eval() for model in model_list]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [model(x) for model in self.model_list]
        logits = torch.stack(logits)
        logits = logits.mean(0)
        return logits


@lru_cache(maxsize=1)
def get_runtime_models(classes: Tuple[int, ...], min_face_size: int):
    """
    Lazily initialize and cache detector/embeddor/model for inference keyed by age classes
    and detector min face size.
    """
    device = torch.device("cpu")
    detector = MTCNN(min_face_size=min_face_size, device=device)
    embeddor = Facenet_Embeddor(device=device).eval().to(device)
    model = Ensemble_Model(classes=classes, device=device).eval().to(device)
    return detector, embeddor, model, device
