# -*- coding: utf-8 -*-
"""
Prediction service that wraps model execution without Flask dependencies.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

import FaceDatasets as fds
from age_prediction.services.models import get_runtime_models


@dataclass
class PredictionResult:
    fig_paths: List[str]
    big_fig_path: Optional[str]
    plt_big: bool
    explainsmall: bool
    error: Optional[str] = None
    error_code: Optional[str] = None


def run_prediction(image_path, upload_dir):
    """
    Run age prediction on an image and return paths to generated figures.
    """
    detector, embeddor, model, device = get_runtime_models()

    try:
        figlist, big_fig = fds.jpg2age(image_path, detector, embeddor, model,
                                       device=device, tight_layout=True)
    except RuntimeError as e:
        if "memory" in str(e).lower() or "oom" in str(e).lower():
            return PredictionResult([], None, False, False,
                                    error="OOM Error: Picture was too large to process on this server. Please upload a smaller image.",
                                    error_code="oom")
        raise

    if figlist[0] == 'No faces found' or figlist[0] == 'Image not RGB':
        return PredictionResult([], None, False, False,
                                error='Unable to use picture: ' + str(figlist[0]),
                                error_code="invalid_image")

    filename = '.'.join(image_path.split('/')[-1].split('.')[:-1])
    fig_paths: List[str] = []
    generated_files: List[str] = []

    # we arent going to plot if only one face, its redundant
    # unless the face errored then we use annot to communicate error
    if len(figlist) > 1 or isinstance(figlist[0], str):
        big_fig_path = os.path.join(upload_dir, filename + '_big_fig_' + str(np.random.randint(0, 10000)) + '.jpg')
        big_fig.savefig(big_fig_path)
        generated_files.append(big_fig_path)
        plt_big = True
    else:
        big_fig_path = None
        plt_big = False

    explainsmall = False
    for fi, fig in enumerate(figlist):
        if isinstance(fig, str) is False:
            fig_paths.append(os.path.join(upload_dir, filename + '_prediction'
                                          + str(fi) + '_' + str(np.random.randint(0, 10000)) + '.jpg'))
            fig.savefig(fig_paths[-1])
            generated_files.append(fig_paths[-1])
            explainsmall = True

    plt.close('all')
    return PredictionResult(fig_paths=fig_paths,
                            big_fig_path=big_fig_path,
                            plt_big=plt_big,
                            explainsmall=explainsmall,
                            error=None,
                            error_code=None)
