"""
Image and Document Preprocessing Module for Future Model Training
Standardizes input dimensions, normalizes color spaces, and computes forensic feature representations.
"""

import cv2
import numpy as np
from typing import Tuple


def preprocess_document_image(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = True
) -> np.ndarray:
    """
    Standardizes document image for neural network feature extractors:
    1. Bilinear resize to fixed input resolution
    2. Optional float32 normalization to [0.0, 1.0] or [-1.0, 1.0]
    """
    if img_bgr is None:
        raise ValueError("Cannot preprocess null image array")

    resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    if normalize:
        processed = resized.astype(np.float32) / 255.0
    else:
        processed = resized

    return processed


def compute_forensic_residual_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts high-pass spatial noise residuals (SRM-like filter) for forensic tampering detection.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    residual = cv2.filter2D(gray, -1, kernel)
    return residual
