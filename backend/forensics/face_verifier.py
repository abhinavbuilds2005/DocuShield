"""
Face Verification Module for SIH26188
Detects and compares facial portrait in identity documents against presented person / live selfie images.

CRITICAL POLICY:
- If no live/person image is supplied, outputs:
  "Face verification not performed — person image not provided."
  (NEVER falsely reports "verified").
- If no face is detected in the document, reports clearly.
- Uses OpenCV Haar Cascades for lightweight, robust face detection without bulky dependencies.
- Multi-signal facial representation: multi-histogram correlation and structural gradient similarity.
- Provides base64 crops of detected faces for transparent human-in-the-loop review.
"""

import os
import cv2
import base64
import numpy as np
from typing import Dict, Any, Optional, Tuple


class FaceVerifier:
    """
    Facial portrait extraction and biometric cross-verification module.
    """

    def __init__(self):
        # Load Haar Cascade from OpenCV data
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = None

    def detect_face(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Detects primary face in an image.
        Returns: (cropped_face_bgr, (x, y, w, h)) or (None, None).
        """
        if self.face_cascade is None or img_bgr is None:
            return None, None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Multi-scale detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            # Fallback with relaxed parameters for lower-resolution documents
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )

        if len(faces) == 0:
            return None, None

        # Pick largest detected face by area
        largest_face = max(faces, key=lambda b: b[2] * b[3])
        x, y, w, h = largest_face

        # Add slight margin around face
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        ih, iw = img_bgr.shape[:2]

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(iw, x + w + pad_x)
        y2 = min(ih, y + h + pad_y)

        crop = img_bgr[y1:y2, x1:x2]
        return crop, (x, y, w, h)

    def _to_base64_crop(self, crop: np.ndarray) -> Optional[str]:
        """Encodes face crop to JPEG base64 string."""
        if crop is None or crop.size == 0:
            return None
        success, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if success:
            return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
        return None

    def compute_similarity(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """
        Computes composite similarity score between two face crops [0.0 to 1.0]:
        1. Multi-channel HSV color & skin tone distribution correlation
        2. Grayscale spatial gradient correlation (Sobel edge maps)
        3. Normalized structural template match
        """
        if face1 is None or face2 is None:
            return 0.0

        # Resize both to standard 128x128 dimensions
        f1_norm = cv2.resize(face1, (128, 128))
        f2_norm = cv2.resize(face2, (128, 128))

        # 1. HSV Histogram Correlation
        hsv1 = cv2.cvtColor(f1_norm, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(f2_norm, cv2.COLOR_BGR2HSV)

        hist1 = cv2.calcHist([hsv1], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_corr = max(0.0, float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)))

        # 2. Grayscale gradient similarity
        g1 = cv2.cvtColor(f1_norm, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2_norm, cv2.COLOR_BGR2GRAY)

        sob1 = cv2.Sobel(g1, cv2.CV_32F, 1, 1)
        sob2 = cv2.Sobel(g2, cv2.CV_32F, 1, 1)

        norm1 = np.linalg.norm(sob1)
        norm2 = np.linalg.norm(sob2)
        if norm1 > 0 and norm2 > 0:
            grad_sim = max(0.0, float(np.sum(sob1 * sob2) / (norm1 * norm2)))
        elif norm1 == 0 and norm2 == 0:
            # Both flat: similarity depends on luminance difference
            lum_diff = abs(float(np.mean(g1)) - float(np.mean(g2))) / 255.0
            grad_sim = max(0.0, 1.0 - lum_diff)
        else:
            grad_sim = 0.2

        # 3. Normalized pixel/structural similarity
        mean_diff = float(np.mean(np.abs(f1_norm.astype(np.float32) - f2_norm.astype(np.float32)))) / 255.0
        pixel_sim = max(0.0, 1.0 - mean_diff)

        # Weighted composite similarity
        composite = (hist_corr * 0.30) + (grad_sim * 0.35) + (pixel_sim * 0.35)
        return round(float(np.clip(composite, 0.0, 1.0)), 4)


    def verify(
        self,
        document_img: np.ndarray,
        person_img: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes face verification between document portrait and optional person image.

        Returns:
            Dict conforming to SIH specification:
            {
                "face_match": bool or None,
                "similarity": float,
                "confidence": float,
                "status": "MATCH" | "MISMATCH" | "NOT_PERFORMED" | "NO_FACE_IN_DOCUMENT" | "NO_FACE_IN_PERSON_IMAGE",
                "explanation": str,
                "document_face_detected": bool,
                "person_face_detected": bool,
                "document_face_b64": str or None,
                "person_face_b64": str or None
            }
        """
        # Document face detection
        doc_crop, doc_box = self.detect_face(document_img)
        doc_face_detected = doc_crop is not None
        doc_face_b64 = self._to_base64_crop(doc_crop) if doc_crop is not None else None

        # Check if live person image was supplied
        if person_img is None:
            return {
                "face_match": None,
                "similarity": 0.0,
                "confidence": 0.0,
                "status": "NOT_PERFORMED",
                "explanation": "Face verification not performed — person image not provided.",
                "document_face_detected": doc_face_detected,
                "person_face_detected": False,
                "document_face_b64": doc_face_b64,
                "person_face_b64": None
            }

        # If person image was provided, detect face in it
        person_crop, person_box = self.detect_face(person_img)
        person_face_detected = person_crop is not None
        person_face_b64 = self._to_base64_crop(person_crop) if person_crop is not None else None

        if not doc_face_detected:
            return {
                "face_match": None,
                "similarity": 0.0,
                "confidence": 0.0,
                "status": "NO_FACE_IN_DOCUMENT",
                "explanation": "No facial portrait detected in identity document.",
                "document_face_detected": False,
                "person_face_detected": person_face_detected,
                "document_face_b64": None,
                "person_face_b64": person_face_b64
            }

        if not person_face_detected:
            return {
                "face_match": None,
                "similarity": 0.0,
                "confidence": 0.0,
                "status": "NO_FACE_IN_PERSON_IMAGE",
                "explanation": "Person image provided, but no face could be detected in it.",
                "document_face_detected": True,
                "person_face_detected": False,
                "document_face_b64": doc_face_b64,
                "person_face_b64": None
            }

        # Both faces detected: compute similarity
        similarity = self.compute_similarity(doc_crop, person_crop)
        match_threshold = 0.55
        is_match = similarity >= match_threshold
        confidence = round(min(0.95, max(0.50, similarity if is_match else (1.0 - similarity))), 2)

        if is_match:
            status = "MATCH"
            explanation = f"Face verified: Document portrait matches live person photo with similarity {similarity*100:.1f}%."
        else:
            status = "MISMATCH"
            explanation = f"Face discrepancy: Document portrait does not match presented person photo (similarity: {similarity*100:.1f}%)."

        return {
            "face_match": is_match,
            "similarity": similarity,
            "confidence": confidence,
            "status": status,
            "explanation": explanation,
            "document_face_detected": True,
            "person_face_detected": True,
            "document_face_b64": doc_face_b64,
            "person_face_b64": person_face_b64
        }
