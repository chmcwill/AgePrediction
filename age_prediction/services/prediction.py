# -*- coding: utf-8 -*-
"""
Prediction service that wraps model execution without Flask dependencies.
Structured for inference-only code paths.
"""
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TVF

from age_prediction.services.errors import (
    InferenceOOMError,
    InvalidImageError,
    NoFacesFoundError,
)
from age_prediction.services import plotting as fpl
from age_prediction.services.models import get_runtime_models, predict_age
from age_prediction.services.preprocessing import (
    box_add_margin,
    box_to_square,
    face_to_tensor,
    load_image,
    resize_to_max_dim,
    tensor_to_pil,
)


@dataclass
class PredictionConfig:
    age_min: int = 10  # minimum age represented by the classifier
    age_max: int = 70  # maximum age represented by the classifier
    max_dim: int = 2048  # max dimension for the loaded image used for inference/cropping
    margin: int = 20  # percent margin to add around a squared face box
    resize_shape: int = 160  # model input size for cropped faces
    face_prob_thresh: float = 0.95  # detector probability threshold
    min_face_size: int = 30  # minimum face box dimension (pixels) before filtering
    tight_layout: bool = True  # whether to call tight_layout on per-face plots
    fig_dpi: int = 100  # DPI for saved figures (lower = faster, smaller)
    detect_max_size: int = 1600  # max dimension for the detector (safe_detect)
    def __post_init__(self):
        if self.age_min > self.age_max:
            raise ValueError("age_min must be <= age_max")

    @property
    def classes(self) -> Tuple[int, ...]:
        return tuple(range(self.age_min, self.age_max + 1))


@dataclass
class PredictionResult:
    """Metadata about saved prediction figures (not per-face predictions)."""
    fig_paths: List[str]  # paths to per-face plots
    big_fig_path: Optional[str]  # path to overlay plot
    plt_big: bool  # whether to display the overlay plot
    explainsmall: bool  # whether to display small plots


class FilterReason(str, Enum):
    NOT_FRONT_FACING = "not_front_facing"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    INSUFFICIENT_RESOLUTION = "insufficient_resolution"
    NOT_ENOUGH_MARGIN = "not_enough_margin"
    IMAGE_NOT_RGB = "image_not_rgb"
    NO_FACES_FOUND = "no_faces_found"


@dataclass
class FaceDetection:
    box: np.ndarray  # working box (after square/margin adjustments)
    prob: float  # detector confidence
    landmarks: Optional[np.ndarray]  # detector landmarks for rotation/front-facing
    original_box: np.ndarray  # detector raw box before adjustments
    status: str = "kept"  # "kept" or "filtered"
    reason: Optional[FilterReason] = None  # filter reason when filtered
    crop: Optional[torch.Tensor] = None  # cropped/rotated/padded face tensor
    prediction: Optional[np.ndarray] = None  # per-face prediction output


@dataclass
class ImageContext:
    image: Image.Image  # PIL image used for plotting/overlay
    tensor: torch.Tensor  # normalized image tensor used for cropping/inference
    width: int  # image width
    height: int  # image height
    filename: str  # basename of the input image
    faces: List[FaceDetection] = field(default_factory=list)  # detected faces and metadata


DEFAULT_PREDICTION_CONFIG = PredictionConfig()


def _image_to_context(image_path: str, config: PredictionConfig) -> ImageContext:
    """Load image and create a reusable context with tensor + metadata."""
    try:
        image = load_image(image_path, config.max_dim)
    except Exception as exc:
        raise InvalidImageError(f"Unable to load image: {exc}") from exc
    tensor = face_to_tensor(image)
    return ImageContext(
        image=image,
        tensor=tensor,
        width=image.width,
        height=image.height,
        filename=os.path.basename(image_path),
    )


def _wrap_faces(boxes, probs, landmarks) -> List[FaceDetection]:
    faces: List[FaceDetection] = []
    if landmarks is None:
        landmarks = [None] * len(boxes)
    for box, prob, lm in zip(boxes, probs, landmarks):
        face_box = np.asarray(box, dtype=np.float32)
        face_lm = None if lm is None else np.asarray(lm, dtype=np.float32)
        faces.append(
            FaceDetection(
                box=face_box,
                prob=float(prob),
                landmarks=face_lm,
                original_box=face_box.copy(),
            )
        )
    return faces


def detect_faces(img_ctx: ImageContext, detector, config: PredictionConfig) -> List[FaceDetection]:
    """Detect faces and wrap into FaceDetection objects."""
    boxes, probs, landmarks = safe_detect(detector, img_ctx.image, max_size=config.detect_max_size)
    if boxes is None or len(boxes) == 0:
        raise NoFacesFoundError("No faces found in image")
    return _wrap_faces(boxes, probs, landmarks)


def safe_detect(detector, img: Image.Image, max_size: int = 1600):
    """Run detector on a resized copy if the image is very large, then scale boxes back."""
    with torch.no_grad():
        img_resized, scale = resize_to_max_dim(img, max_size)
        if scale > 1:
            boxes, probs, landmarks = detector.detect(img_resized, landmarks=True)
            if boxes is not None:
                boxes = boxes * scale
            if landmarks is not None:
                landmarks = landmarks * scale
            return boxes, probs, landmarks
        return detector.detect(img, landmarks=True)


def _mark_filtered(face: FaceDetection, reason: FilterReason) -> None:
    face.status = "filtered"
    face.reason = reason


def _front_facing(face: FaceDetection) -> bool:
    """Check nose between eyes heuristic to ensure front-facing."""
    if face.landmarks is None or face.landmarks.shape[0] < 3:
        return True
    nose_x = face.landmarks[2, 0]
    eye_pad = 0.1 * (face.landmarks[1, 0] - face.landmarks[0, 0])
    return (nose_x > face.landmarks[0, 0] - eye_pad) and (nose_x < face.landmarks[1, 0] + eye_pad)


def _has_enough_margin(box: np.ndarray, img_w: int, img_h: int) -> bool:
    return (
        box[3] < 1.4 * img_h
        and box[2] < 1.4 * img_w
        and box[1] > -0.4 * img_h
        and box[0] > -0.4 * img_w
    )


def _apply_face_filters(img_ctx: ImageContext, config: PredictionConfig) -> None:
    """Apply face-level filters and annotate reasons instead of dropping faces."""
    for face in img_ctx.faces:
        if not _front_facing(face):
            _mark_filtered(face, FilterReason.NOT_FRONT_FACING)
            continue

        if face.prob is None or not np.isfinite(face.prob) or face.prob < config.face_prob_thresh:
            _mark_filtered(face, FilterReason.INSUFFICIENT_CONFIDENCE)
            continue

        w = face.original_box[2] - face.original_box[0]
        h = face.original_box[3] - face.original_box[1]
        if min(w, h) < config.min_face_size:
            _mark_filtered(face, FilterReason.INSUFFICIENT_RESOLUTION)
            continue

        square_box = box_to_square(face.original_box.copy())
        margin_box = box_add_margin(square_box, config.margin, img_ctx.width, img_ctx.height, clamp=False)
        face.box = margin_box

        if not _has_enough_margin(margin_box, img_ctx.width, img_ctx.height):
            _mark_filtered(face, FilterReason.NOT_ENOUGH_MARGIN)
            continue


def _crop_face_with_padding(
    img_tensor: torch.Tensor, box: np.ndarray, resize_shape: int
) -> torch.Tensor:
    """Crop a square box, padding beyond image edges to keep it square."""
    x1, y1, x2, y2 = [int(round(coord)) for coord in box]
    _, img_h, img_w = img_tensor.shape

    # inclusive coordinates to match legacy helper (+1 on width/height)
    pad_left = max(-x1, 0)
    pad_top = max(-y1, 0)
    pad_right = max(x2 - (img_w - 1), 0)
    pad_bottom = max(y2 - (img_h - 1), 0)

    x1_clamped = max(x1, 0)
    y1_clamped = max(y1, 0)
    x2_clamped = min(x2, img_w - 1)
    y2_clamped = min(y2, img_h - 1)

    crop = img_tensor[:, y1_clamped : y2_clamped + 1, x1_clamped : x2_clamped + 1]
    if any(v > 0 for v in (pad_left, pad_right, pad_top, pad_bottom)):
        crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), value=0)

    crop = F.interpolate(
        crop.unsqueeze(0),
        size=(resize_shape, resize_shape),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return crop


def _prepare_face_crops(img_ctx: ImageContext, config: PredictionConfig) -> None:
    """Generate cropped tensors (with padding) for kept faces."""
    for face in img_ctx.faces:
        if face.status != "kept":
            continue
        # Rotate image to align eyes before cropping if landmarks available.
        rotated_tensor = _rotate_face_tensor(img_ctx.tensor, face) if face.landmarks is not None else img_ctx.tensor

        face.crop = _crop_face_with_padding(rotated_tensor, face.box, config.resize_shape)


def _rotate_face_tensor(img_tensor: torch.Tensor, face: FaceDetection) -> torch.Tensor:
    """Rotate image around face center to align eyes horizontally."""
    if face.landmarks is None or face.landmarks.shape[0] < 2:
        return img_tensor
    dx = face.landmarks[1, 0] - face.landmarks[0, 0]
    dy = face.landmarks[1, 1] - face.landmarks[0, 1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    midpoint = (
        float(0.5 * (face.box[0] + face.box[2])),
        float(0.5 * (face.box[1] + face.box[3])),
    )
    return TVF.rotate(img_tensor, angle=angle, center=midpoint, fill=0)


def _clamp_box(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Clamp a box to image bounds for safer plotting."""
    if box is None:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [
            np.clip(box[0], 0, img_w),
            np.clip(box[1], 0, img_h),
            np.clip(box[2], 0, img_w),
            np.clip(box[3], 0, img_h),
        ],
        dtype=np.float32,
    )


def _build_overlay_payload(img_ctx: ImageContext) -> List[fpl.OverlayItem]:
    """Build overlay annotations with bounds and prediction/error labels."""
    overlays: List[fpl.OverlayItem] = []
    for face in img_ctx.faces:
        if face.status == "kept" and face.prediction is not None:
            label = face.prediction
        else:
            label = face.reason.value if face.reason else FilterReason.NO_FACES_FOUND.value
        bounds = _clamp_box(face.box if face.box is not None else face.original_box, img_ctx.width, img_ctx.height)
        overlays.append(fpl.OverlayItem(bounds=bounds, label=label))
    return overlays


def predict_faces(
    img_ctx: ImageContext,
    kept_faces: List[FaceDetection],
    embeddor,
    model,
    device,
    config: PredictionConfig,
) -> Tuple[List, object]:
    """Run embeddings, age prediction, and figure creation."""
    figs = []
    base_name = img_ctx.filename
    face_images: List[Image.Image] = []
    face_tensors: List[torch.Tensor] = []
    face_indices: List[int] = []

    for idx, face in enumerate(kept_faces):
        if face.crop is None:
            continue
        face_images.append(tensor_to_pil(face.crop))
        face_tensors.append(face.crop)
        face_indices.append(idx)

    if face_tensors:
        batch = torch.stack(face_tensors).to(device)
        try:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                embeddings = embeddor.forward_1792(batch)
                outputs = model(embeddings).cpu().float()
        except RuntimeError as e:
            for img in face_images:
                img.close()
            if "memory" in str(e).lower() or "oom" in str(e).lower():
                raise InferenceOOMError("OOM during inference") from e
            raise

        preds, softmaxes = predict_age(outputs, classes=config.classes)
        for local_idx, face_idx in enumerate(face_indices):
            face = kept_faces[face_idx]
            pred_array = np.atleast_1d(np.asarray(preds[local_idx]))
            face.prediction = pred_array

            fig = fpl.plot_image_and_pred(
                fpl.FacePlotData(
                    image=face_images[local_idx],
                    prediction=float(pred_array[0]),
                    output_softmax=softmaxes[local_idx : local_idx + 1],
                    image_name=base_name,
                    img_input_size=config.resize_shape,
                    classification=True,
                    age_min=config.age_min,
                    age_max=config.age_max,
                ),
                tight_layout=config.tight_layout,
                figshow=False,
            )
            face_images[local_idx].close()
            figs.append(fig)

    overlays = _build_overlay_payload(img_ctx)
    big_fig, _ = fpl.overlay_preds_on_img(img_ctx.image, overlays)
    return figs, big_fig


def render_figures(figlist, big_fig, upload_dir, base_filename, config: PredictionConfig):
    """Save figures to disk and return paths plus display flags."""
    fig_paths: List[str] = []
    generated_files: List[str] = []

    big_fig_path = None
    plt_big = False
    if big_fig is not None:
        big_fig_path = os.path.join(
            upload_dir, f"{base_filename}_big_fig_{uuid.uuid4().hex}.jpg"
        )
        big_fig.savefig(big_fig_path, dpi=config.fig_dpi)
        generated_files.append(big_fig_path)
        plt_big = True

    explainsmall = False
    for fi, fig in enumerate(figlist):
        fig_path = os.path.join(
            upload_dir,
            f"{base_filename}_prediction{fi}_{uuid.uuid4().hex}.jpg",
        )
        fig.savefig(fig_path, dpi=config.fig_dpi)
        fig_paths.append(fig_path)
        generated_files.append(fig_path)
        explainsmall = True

    for fig in figlist:
        if hasattr(fig, "close"):
            plt.close(fig)
    if hasattr(big_fig, "close"):
        plt.close(big_fig)

    return PredictionResult(
        fig_paths=fig_paths,
        big_fig_path=big_fig_path,
        plt_big=plt_big,
        explainsmall=explainsmall,
    ), generated_files


def run_prediction(image_path, upload_dir, config: PredictionConfig = DEFAULT_PREDICTION_CONFIG):
    """Orchestrate the prediction workflow and save outputs."""
    detector, embeddor, model, device = get_runtime_models(tuple(config.classes), config.min_face_size)

    img_ctx = _image_to_context(image_path, config)
    try:
        img_ctx.faces = detect_faces(img_ctx, detector, config)
        _apply_face_filters(img_ctx, config)
        _prepare_face_crops(img_ctx, config)

        kept_faces = [face for face in img_ctx.faces if face.status == "kept"]
        figlist, big_fig = predict_faces(img_ctx, kept_faces, embeddor, model, device, config)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        result, generated_files = render_figures(figlist, big_fig, upload_dir, base_filename, config)
        return result, generated_files
    finally:
        try:
            img_ctx.image.close()
        except Exception:
            pass
