"""
Aadhaar Document Field Detection Module
Extracts localized bounding boxes for key document regions:
- aadhaar_number
- name
- dob
- gender
- photo
- qr_code
- emblem_header

Design Principles:
1. Singleton Model Loading: Loaded once on first request, reused across calls.
2. Graceful Degradation: Never crashes screening pipeline on model failure or missing weights.
3. Environment Configuration: Model path, confidence threshold, and device are configurable.
4. Privacy Enforcement: Never exposes full personal identity numbers in debug logs.
5. Coordinate Normalization & Box Clipping: Guaranteed valid pixel and normalized boxes.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

# Singleton instance cache
_FIELD_DETECTOR_INSTANCE = None


class AadhaarFieldDetector:
    """
    Lightweight YOLO-based document region and field localization engine.
    Used solely for field detection/cropping to assist OCR and forensic inspection.
    NOTE: Field detection does NOT prove or evaluate document authenticity.
    """

    def __init__(self):
        self.enabled = os.environ.get("FIELD_DETECTOR_ENABLED", "true").strip().lower() in ("true", "1", "yes")
        default_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "aadhaar_field_detector.pt"
        )
        self.model_path = os.environ.get("FIELD_DETECTOR_MODEL_PATH", default_model_path)
        self.conf_threshold = float(os.environ.get("FIELD_DETECTOR_CONFIDENCE", "0.35"))
        self.device_pref = os.environ.get("FIELD_DETECTOR_DEVICE", "auto").strip().lower()

        self._model = None
        self._init_attempted = False
        self._is_available = False
        self._class_names = {}

    def _lazy_init(self):
        """Initializes model on first invocation to save memory until needed."""
        if self._init_attempted:
            return
        self._init_attempted = True

        if not self.enabled:
            print("[VISION] Field detector is disabled via FIELD_DETECTOR_ENABLED=false")
            self._is_available = False
            return

        if not os.path.exists(self.model_path):
            print(f"[VISION WARNING] Field detector weights not found at: {self.model_path}. Running without field detector.")
            self._is_available = False
            return

        try:
            import torch
            from ultralytics import YOLO

            if self.device_pref == "cpu":
                device = "cpu"
            elif (self.device_pref == "cuda" or self.device_pref == "auto") and torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = "0"
            else:
                device = "cpu"

            print(f"[VISION] Loading field detector from: {self.model_path} (device: {device})...")
            self._model = YOLO(self.model_path)
            self._device = device
            self._class_names = self._model.names if hasattr(self._model, "names") else {}
            self._is_available = True
            print(f"[VISION] Field detector successfully loaded with {len(self._class_names)} classes.")
        except Exception as e:
            print(f"[VISION ERROR] Could not initialize YOLO field detector: {e}")
            self._model = None
            self._is_available = False

    @property
    def is_available(self) -> bool:
        self._lazy_init()
        return self._is_available

    @property
    def classes(self) -> Dict[int, str]:
        self._lazy_init()
        return self._class_names

    @property
    def device(self) -> str:
        self._lazy_init()
        return getattr(self, "_device", "cpu")

    def detect_fields(self, image_path_or_array, iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Detects document field bounding boxes in an Aadhaar card image.

        Args:
            image_path_or_array: File path (str) or BGR numpy array.
            iou_threshold: Overlap threshold for duplicate box suppression.

        Returns:
            Dict containing:
                - status: 'SUCCESS', 'DISABLED', or 'UNAVAILABLE'
                - fields: List of detected field dicts:
                    {
                        "class_id": int,
                        "class_name": str,
                        "confidence": float,
                        "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
                        "bbox_normalized": {"x": float, "y": float, "width": float, "height": float}
                    }
                - field_crops: Dict mapping field names to cropped BGR numpy arrays
                - image_dims: [width, height]
        """
        self._lazy_init()

        empty_result = {
            "status": "DISABLED" if not self.enabled else "UNAVAILABLE",
            "fields": [],
            "field_crops": {},
            "image_dims": [0, 0]
        }

        if not self._is_available or self._model is None:
            return empty_result

        # Load image
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                return empty_result
            img = cv2.imread(image_path_or_array)
            if img is None:
                return empty_result
        else:
            img = image_path_or_array

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return empty_result

        try:
            # Run inference with autograd disabled
            import torch
            with torch.no_grad():
                results = self._model(
                    img,
                    conf=self.conf_threshold,
                    iou=iou_threshold,
                    device=self._device,
                    verbose=False
                )

            if not results or len(results) == 0:
                return {
                    "status": "SUCCESS",
                    "fields": [],
                    "field_crops": {},
                    "image_dims": [w, h]
                }

            boxes = results[0].boxes
            detected_fields = []
            field_crops = {}

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = round(float(box.conf[0].item()), 3)
                    cls_name = self._class_names.get(cls_id, f"class_{cls_id}")

                    coords = box.xyxy[0].tolist()
                    x1 = max(0, min(w - 1, int(coords[0])))
                    y1 = max(0, min(h - 1, int(coords[1])))
                    x2 = max(x1 + 1, min(w, int(coords[2])))
                    y2 = max(y1 + 1, min(h, int(coords[3])))
                    bw = x2 - x1
                    bh = y2 - y1

                    # Normalized box
                    norm_box = {
                        "x": round(x1 / w, 4),
                        "y": round(y1 / h, 4),
                        "width": round(bw / w, 4),
                        "height": round(bh / h, 4)
                    }

                    field_item = {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "bbox_normalized": norm_box,
                        "source": "yolov8n_trained"
                    }
                    detected_fields.append(field_item)

                    # Extract padded crop for OCR or photo inspection
                    from backend.vision.crop_utils import extract_padded_crop
                    crop, crop_meta = extract_padded_crop(img, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, pad_ratio=0.08)
                    if crop is not None and crop.size > 0:
                        # If duplicate class exists, store highest confidence crop
                        if cls_name not in field_crops or conf > field_crops[cls_name]["confidence"]:
                            field_crops[cls_name] = {
                                "crop": crop,
                                "confidence": conf,
                                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "padded_bbox": crop_meta.get("padded_bbox"),
                                "source": "yolov8n_trained"
                            }

            # Sort detected fields by class and confidence
            detected_fields.sort(key=lambda x: x["confidence"], reverse=True)

            return {
                "status": "SUCCESS",
                "model_name": "Trained YOLOv8n Field Detector (Aadhaar Only)",
                "model_type": "yolov8n_trained",
                "fields": detected_fields,
                "field_crops": field_crops,
                "image_dims": [w, h]
            }

        except Exception as e:
            print(f"[VISION ERROR] Exception during field detection: {e}")
            return {
                "status": "ERROR",
                "fields": [],
                "field_crops": {},
                "image_dims": [w, h]
            }


def get_field_detector() -> AadhaarFieldDetector:
    """Returns the singleton field detector instance."""
    global _FIELD_DETECTOR_INSTANCE
    if _FIELD_DETECTOR_INSTANCE is None:
        _FIELD_DETECTOR_INSTANCE = AadhaarFieldDetector()
    return _FIELD_DETECTOR_INSTANCE
