"""
Data Augmentation for Document Forgery Detection
Simulates realistic transmission, capture, and physical noise without altering tampering semantics.
"""

import cv2
import numpy as np
import random


class DocumentAugmentor:
    """
    Applies controlled realistic perturbations to document images.
    """

    @staticmethod
    def add_gaussian_blur(img: np.ndarray, max_ksize: int = 5) -> np.ndarray:
        """Simulates camera defocus / lens blur."""
        k = random.choice([3, 5]) if max_ksize >= 3 else 3
        return cv2.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def simulate_jpeg_compression(img: np.ndarray, quality_range: tuple = (40, 85)) -> np.ndarray:
        """Simulates messaging app / web recompression."""
        q = random.randint(quality_range[0], quality_range[1])
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    @staticmethod
    def adjust_brightness_contrast(img: np.ndarray, alpha_range=(0.85, 1.15), beta_range=(-15, 15)) -> np.ndarray:
        """Simulates varying document capture lighting conditions."""
        alpha = random.uniform(alpha_range[0], alpha_range[1])
        beta = random.randint(beta_range[0], beta_range[1])
        return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
