# -*- coding: utf-8 -*-
"""
Typed errors for prediction workflow.
"""


class PredictionError(Exception):
    """Base class for prediction-related errors."""


class InvalidImageError(PredictionError):
    """Raised when the uploaded image is not usable (e.g., not RGB)."""


class NoFacesFoundError(PredictionError):
    """Raised when no faces are detected in the image."""


class InferenceOOMError(PredictionError):
    """Raised when inference fails due to an out-of-memory condition."""


class StorageError(PredictionError):
    """Raised when file upload/save fails validation or cannot be stored."""
