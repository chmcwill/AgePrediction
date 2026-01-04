# -*- coding: utf-8 -*-
"""
Prediction service that wraps model execution without Flask dependencies.
Structured for inference-only code paths.
"""
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms import functional as TVF

import FaceModels as fm
import FacePlotting as fpl
from FaceDatasets import box_add_margin, box_to_square
from age_prediction.services.errors import (
    InferenceOOMError,
    InvalidImageError,
    NoFacesFoundError,
)
from age_prediction.services.models import get_runtime_models


@dataclass
class PredictionConfig:
    max_dim: int = 2048  # max dimension for the loaded image used for inference/cropping
    margin: int = 20  # percent margin to add around a squared face box
    resize_shape: int = 160  # model input size for cropped faces
    face_prob_thresh: float = 0.95  # detector probability threshold
    min_face_size: int = 30  # minimum face box dimension (pixels) before filtering
    tight_layout: bool = True  # whether to call tight_layout on per-face plots
    detect_max_size: int = 1600  # max dimension for the detector (safe_detect)


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


def load_image(image_path: str, config: PredictionConfig) -> Image.Image:
    """
    Load and resize image for inference. Returns an RGB PIL image.
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")

            img, _ = _resize_to_max_dim(img, config.max_dim)
            return img.copy()
    except Exception as exc:
        raise InvalidImageError(f"Unable to load image: {exc}") from exc


def face_to_tensor(face_img: Image.Image) -> torch.Tensor:
    """Convert PIL face image to normalized tensor in [-1, 1]."""
    np_array = (np.asarray(face_img).astype(np.float32) - 127.5) / 128
    return torch.as_tensor(np_array.transpose(2, 0, 1), dtype=torch.float32)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert normalized tensor back to PIL image for plotting."""
    if torch.max(tensor) <= 1:
        tensor = (tensor + 1) * (255 / 2)
    array = tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(array)


def _image_to_context(image_path: str, config: PredictionConfig) -> ImageContext:
    """Load image and create a reusable context with tensor + metadata."""
    image = load_image(image_path, config)
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
        img_resized, scale = _resize_to_max_dim(img, max_size)
        if scale > 1:
            boxes, probs, landmarks = detector.detect(img_resized, landmarks=True)
            if boxes is not None:
                boxes = boxes * scale
            if landmarks is not None:
                landmarks = landmarks * scale
            return boxes, probs, landmarks
        return detector.detect(img, landmarks=True)


def _resize_to_max_dim(img: Image.Image, max_dim: int) -> Tuple[Image.Image, float]:
    """Resize image so the largest dimension is <= max_dim; returns (image, scale_used)."""
    w, h = img.size
    scale = max(w, h) / max_dim
    if scale > 1:
        new_w, new_h = int(w / scale), int(h / scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img, scale


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


def _build_overlay_payload(img_ctx: ImageContext) -> Tuple[List[np.ndarray], List[object]]:
    """Build aligned bounds + preds/reasons for overlay plotting."""
    bounds: List[np.ndarray] = []
    preds: List[object] = []
    for face in img_ctx.faces:
        bounds.append(_clamp_box(face.box if face.box is not None else face.original_box, img_ctx.width, img_ctx.height))
        if face.status == "kept" and face.prediction is not None:
            preds.append(face.prediction)
        else:
            reason = face.reason.value if face.reason else FilterReason.NO_FACES_FOUND.value
            preds.append(reason)
    return bounds, preds


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

    for face in kept_faces:
        if face.crop is None:
            continue

        face_image = tensor_to_pil(face.crop)
        face_tensor = face.crop.to(device)

        try:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                embedding = embeddor.forward_1792(face_tensor)
                output = model(embedding).cpu().float()
        except RuntimeError as e:
            face_image.close()
            if "memory" in str(e).lower() or "oom" in str(e).lower():
                raise InferenceOOMError("OOM during inference") from e
            raise

        pred, output_softmax = fm.predict_age(output, classes=range(10, 71))
        pred_array = np.atleast_1d(np.asarray(pred))
        face.prediction = pred_array

        fig = fpl.plot_image_and_pred(
            face_image,
            float(pred_array[0]),
            CLASSIF=True,
            output_softmax=output_softmax,
            image_name=base_name,
            tight_layout=config.tight_layout,
            figshow=False,
        )
        face_image.close()
        figs.append(fig)

    face_bounds, preds_overlay = _build_overlay_payload(img_ctx)
    big_fig, _ = fpl.overlay_preds_on_img(img_ctx.image, face_bounds, preds_overlay)
    return figs, big_fig


def render_figures(figlist, big_fig, upload_dir, base_filename):
    """Save figures to disk and return paths plus display flags."""
    fig_paths: List[str] = []
    generated_files: List[str] = []

    big_fig_path = None
    plt_big = False
    if big_fig is not None:
        big_fig_path = os.path.join(
            upload_dir, f"{base_filename}_big_fig_{np.random.randint(0, 10000)}.jpg"
        )
        big_fig.savefig(big_fig_path)
        generated_files.append(big_fig_path)
        plt_big = True

    explainsmall = False
    for fi, fig in enumerate(figlist):
        fig_path = os.path.join(
            upload_dir,
            f"{base_filename}_prediction{fi}_{np.random.randint(0, 10000)}.jpg",
        )
        fig.savefig(fig_path)
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
    detector, embeddor, model, device = get_runtime_models()

    img_ctx = _image_to_context(image_path, config)
    try:
        img_ctx.faces = detect_faces(img_ctx, detector, config)
        _apply_face_filters(img_ctx, config)
        _prepare_face_crops(img_ctx, config)

        kept_faces = [face for face in img_ctx.faces if face.status == "kept"]
        figlist, big_fig = predict_faces(img_ctx, kept_faces, embeddor, model, device, config)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        result, generated_files = render_figures(figlist, big_fig, upload_dir, base_filename)
        return result, generated_files
    finally:
        try:
            img_ctx.image.close()
        except Exception:
            pass
