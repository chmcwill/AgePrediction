# -*- coding: utf-8 -*-
"""
Plotting utilities for inference-time visualizations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from age_prediction.services.preprocessing import tensor_to_pil


@dataclass
class FacePlotData:
    image: Image.Image
    prediction: float
    output_softmax: Optional[np.ndarray] = None
    image_name: str = "input image"
    img_input_size: int = 160
    classification: bool = True
    age_min: int = 10
    age_max: int = 70


@dataclass
class OverlayItem:
    bounds: np.ndarray
    label: Union[np.ndarray, str]


def imshow_tensor(tensor: Tensor, title_str: str = "Image Tensor", landmarks: Optional[np.ndarray] = None):
    """Plot a tensor as an image for quick inspection."""
    img = tensor_to_pil(tensor.cpu())
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(title_str)
    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            plt.plot(landmarks[i, 0], landmarks[i, 1], "o", color="b")


def _prepare_face_image(image: Image.Image, img_input_size: int) -> Tensor:
    """
    Ensure face image is square and resized to model input size.
    Returns a numpy array ready for matplotlib.
    """
    h, w = image.size
    if h != w:
        raise ValueError("Face image must be square before plotting.")

    if h != img_input_size:
        image = image.resize((img_input_size, img_input_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def plot_image_and_pred(
    data: FacePlotData,
    fig_ax=None,
    tight_layout: bool = True,
    figshow: bool = False,
):
    """
    Plot a cropped face alongside its predicted age or probability distribution.
    """
    image_arr = _prepare_face_image(data.image, data.img_input_size)
    pred_display = f"{data.prediction:.1f}"

    if fig_ax is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, axs = fig_ax
        for ax in axs:
            ax.clear()

    axs[0].imshow(image_arr)
    axs[0].set_title("Cropped and Rotated Face", fontsize=13)
    if data.classification:
        if data.output_softmax is None:
            raise ValueError("output_softmax is required when classification=True")
        fig.suptitle(f"Age Prediction using Classification", fontsize=16)
        age_axis = range(data.age_min, data.age_max + 1)
        axs[1].plot(age_axis, data.output_softmax.squeeze())
        axs[1].set_xlabel("Age", fontsize=12)
        axs[1].set_ylabel("Probability", fontsize=12)
        axs[1].set_title(f"Age Prediction Distribution (pred = {pred_display})", fontsize=13)
        axs[1].plot([data.prediction] * 2, [0, np.max(data.output_softmax)], linewidth=2)
        axs[1].legend(["PDF", "Expectation"], loc="upper right")
    else:
        fig.suptitle(f"Age Prediction using Regression")
        axs[1].set_xlabel("Age")
        axs[1].set_title(f"Age Prediction (pred = {pred_display})")
        axs[1].plot([data.prediction] * 2, [0, 1], linewidth=2)
        axs[1].set_xlim([data.age_min, data.age_max])

    if tight_layout:
        plt.tight_layout()

    if figshow:
        plt.draw()
        plt.pause(0.001)
        fig.show()

    return fig


def overlay_preds_on_img(image: Image.Image, overlays: Sequence[OverlayItem]):
    """
    Create an annotated version of the input image with face bounds and predictions/errors.
    """
    if image is None:
        raise ValueError("Image is required for overlay plotting.")

    img_w, img_h = image.size
    ratio = img_w / img_h
    plt_h = 10
    plt_w = max(int(plt_h * ratio), 8)
    fig, ax = plt.subplots(1, 1, figsize=(plt_w, plt_h))

    ax.imshow(image)
    ax.set_title("Full Image with Annotated Predictions", fontsize=20, pad=20)

    preds_error = [item.label for item in overlays if isinstance(item.label, str)]
    preds_tensor = [item.label for item in overlays if not isinstance(item.label, str)]
    bounds_error = [item.bounds for item in overlays if isinstance(item.label, str)]
    bounds_tensor = [item.bounds for item in overlays if not isinstance(item.label, str)]

    preds_sorted = preds_error + preds_tensor
    face_bounds_sorted = bounds_error + bounds_tensor

    face_sizes = []
    for face_bound in face_bounds_sorted:
        if face_bound is None:
            face_sizes.append(0)
        else:
            face_sizes.append(face_bound[3] - face_bound[1])

    n_faces = len(face_bounds_sorted)
    for i, (face_bound, pred) in enumerate(zip(face_bounds_sorted, preds_sorted)):
        if face_bound is None:
            continue

        if isinstance(pred, str):
            pred = "Error:\n" + pred
            txt_size = 9
            h_offset = 0.75
        else:
            pred_val = float(np.ravel(pred)[0])
            pred_text = f"{pred_val:.1f}"
            pred = f"Age: {pred_text}" if n_faces <= 3 else pred_text
            txt_size = 20 if n_faces <= 2 else 18 if n_faces <= 5 else 15 if n_faces <= 8 else 12
            h_offset = 0.4

        rect = patches.Rectangle(
            (face_bound[0], face_bound[1]),
            face_bound[2] - face_bound[0],
            face_bound[3] - face_bound[1],
            linewidth=1,
            edgecolor=(1.0, 0.7, 0.7),
            facecolor="none",
        )
        ax.add_patch(rect)

        bbx_x = np.mean(face_bound[[0, 2]])
        arrow_indent_pct = 0.2 if face_bound[1] < 0.05 * img_h else 0.05 if face_bound[1] < 0.1 * img_h else 0.01
        bbx_y = np.max([face_bound[1], 0]) + arrow_indent_pct * face_sizes[i]

        offset_y = np.min((h_offset * (face_bound[3] - face_bound[1]), 0.95 * bbx_y))
        offset_x_pct = (0.02 - (-0.02)) * (bbx_x / img_w) + (-0.02)
        offset_x = np.clip(offset_x_pct * img_w, -bbx_x, (img_w - bbx_x))

        ax.annotate(
            pred,
            xy=(bbx_x, bbx_y),
            xycoords="data",
            xytext=(bbx_x + offset_x, bbx_y - offset_y),
            textcoords="data",
            size=txt_size,
            va="center",
            ha="center",
            bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
            arrowprops=dict(
                arrowstyle="wedge,tail_width=1.",
                fc=(1.0, 0.7, 0.7),
                ec="none",
                patchA=None,
                patchB=None,
                relpos=(0.5, 0.5),
            ),
        )

    plt.tight_layout()
    return fig, ax


__all__ = [
    "FacePlotData",
    "OverlayItem",
    "imshow_tensor",
    "overlay_preds_on_img",
    "plot_image_and_pred",
]
