"""
PAN Card Field Detection Module (YOLOv8n + Layout Heuristic Fallback)
DocuShield AI — Phase 2 Modular Expansion

Extracts localized bounding boxes for key PAN card regions:
- pan_number
- name
- father_name
- date_of_birth

Design Principles:
1. Modular Architecture: Independent from Aadhaar detector.
2. Graceful Fallback: If YOLO model fails to load or yields zero detections,
   falls back to canonical layout heuristics. Never crashes the screening pipeline.
3. Provenance Transparency: Output explicitly states whether detections originate
   from 'trained_yolov8n' or 'layout_heuristic'.
4. Forensic Safeguards: Field localization only; NEVER evaluates document authenticity.
5. Privacy & PII: Never logs or exposes raw PAN numbers or citizen names.
"""

import os
import re
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.vision.crop_utils import extract_padded_crop

_PAN_DETECTOR_INSTANCE = None


class PANFieldDetector:
    """
    Modular YOLO-based field detection and crop generator for Indian PAN cards.
    Includes canonical layout heuristic fallback for low-contrast/synthetic edge cases.
    """

    CLASS_NAMES = {
        0: "pan_number",
        1: "name",
        2: "father_name",
        3: "date_of_birth"
    }

    def __init__(self):
        self.enabled = os.environ.get("PAN_FIELD_DETECTOR_ENABLED", "true").strip().lower() in ("true", "1", "yes")
        default_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "pan_field_detector.pt"
        )
        self.model_path = os.environ.get("PAN_FIELD_DETECTOR_MODEL_PATH", default_model_path)
        self.conf_threshold = float(os.environ.get("PAN_FIELD_DETECTOR_CONFIDENCE", "0.30"))
        self.device_pref = os.environ.get("PAN_FIELD_DETECTOR_DEVICE", "auto").strip().lower()

        self._model = None
        self._init_attempted = False
        self._is_available = False
        self._device = "cpu"

    def _lazy_init(self):
        """Lazy loads YOLOv8n PAN detector model on first request."""
        if self._init_attempted:
            return
        self._init_attempted = True

        if not self.enabled:
            print("[VISION] PAN field detector disabled via PAN_FIELD_DETECTOR_ENABLED=false")
            self._is_available = False
            return

        if not os.path.exists(self.model_path):
            print(f"[VISION WARNING] PAN field detector weights not found at: {self.model_path}. Using layout fallback.")
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

            print(f"[VISION] Loading PAN field detector from: {self.model_path} (device: {device})...")
            self._model = YOLO(self.model_path)
            self._device = device
            self._is_available = True
            print("[VISION] PAN field detector successfully initialized.")
        except Exception as e:
            print(f"[VISION ERROR] Failed to initialize YOLO PAN field detector: {e}. Falling back to layout heuristic.")
            self._model = None
            self._is_available = False

    @property
    def is_available(self) -> bool:
        self._lazy_init()
        return self._is_available

    @property
    def classes(self) -> Dict[int, str]:
        return dict(self.CLASS_NAMES)

    @property
    def device(self) -> str:
        self._lazy_init()
        return self._device

    def _generate_layout_heuristic_fields(self, img: np.ndarray, w: int, h: int) -> Dict[str, Any]:
        """
        Estimates canonical Indian PAN card field positions based on standardized
        physical dimensions and layout geometry (NSDL/UTIITSL spec).
        Used when YOLO model is unavailable or encounters flat digital drawings.
        """
        # Canonical proportions for standard landscape PAN card:
        # Width: 85.6mm, Height: 53.98mm (aspect ~ 1.58)
        heuristic_specs = [
            # class_id, class_name, x_rel, y_rel, w_rel, h_rel, conf
            (1, "name", 0.08, 0.28, 0.52, 0.10, 0.65),
            (2, "father_name", 0.08, 0.41, 0.52, 0.10, 0.65),
            (3, "date_of_birth", 0.08, 0.54, 0.35, 0.09, 0.65),
            (0, "pan_number", 0.08, 0.67, 0.48, 0.11, 0.70)
        ]

        fields = []
        field_crops = {}

        for cls_id, cls_name, xr, yr, wr, hr, conf in heuristic_specs:
            x1 = max(0, min(w - 1, int(xr * w)))
            y1 = max(0, min(h - 1, int(yr * h)))
            x2 = max(x1 + 1, min(w, int((xr + wr) * w)))
            y2 = max(y1 + 1, min(h, int((yr + hr) * h)))

            norm_box = {
                "x": round(x1 / w, 4),
                "y": round(y1 / h, 4),
                "width": round((x2 - x1) / w, 4),
                "height": round((y2 - y1) / h, 4)
            }

            field_item = {
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "bbox_normalized": norm_box,
                "source": "layout_heuristic"
            }
            fields.append(field_item)

            crop, crop_meta = extract_padded_crop(img, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, pad_ratio=0.08)
            if crop is not None and crop.size > 0:
                field_crops[cls_name] = {
                    "crop": crop,
                    "confidence": conf,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "padded_bbox": crop_meta.get("padded_bbox"),
                    "source": "layout_heuristic"
                }

        return {
            "status": "SUCCESS",
            "detector_source": "layout_heuristic",
            "model_name": "pan_field_detector",
            "model_type": "layout_heuristic",
            "fallback": True,
            "fields": fields,
            "field_crops": field_crops,
            "image_dims": [w, h],
            "note": "Field bounding boxes generated via standardized PAN layout geometry."
        }

    def detect_fields(self, image_path_or_array, iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Detects PAN card fields (pan_number, name, father_name, date_of_birth).

        Args:
            image_path_or_array: File path (str) or BGR numpy array.
            iou_threshold: Box overlap suppression threshold.

        Returns:
            Dict containing fields list, crops dict, dims, and provenance metadata.
        """
        self._lazy_init()

        empty_result = {
            "status": "DISABLED" if not self.enabled else "UNAVAILABLE",
            "detector_source": "none",
            "model_name": "pan_field_detector",
            "model_type": "none",
            "fallback": False,
            "fields": [],
            "field_crops": {},
            "image_dims": [0, 0]
        }

        # Load image safely
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                return empty_result
            img = cv2.imread(image_path_or_array)
            if img is None:
                return empty_result
        elif isinstance(image_path_or_array, np.ndarray):
            img = image_path_or_array
        else:
            return empty_result

        if img is None or img.size == 0 or len(img.shape) < 2:
            return empty_result

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return empty_result

        if not self.enabled:
            return empty_result

        # Try trained YOLOv8n detector first if available
        if self._is_available and self._model is not None:
            try:
                import torch
                with torch.no_grad():
                    results = self._model(
                        img,
                        conf=self.conf_threshold,
                        iou=iou_threshold,
                        device=self._device,
                        verbose=False
                    )

                if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    detected_fields = []
                    field_crops = {}

                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = round(float(box.conf[0].item()), 3)
                        cls_name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                        coords = box.xyxy[0].tolist()
                        x1 = max(0, min(w - 1, int(coords[0])))
                        y1 = max(0, min(h - 1, int(coords[1])))
                        x2 = max(x1 + 1, min(w, int(coords[2])))
                        y2 = max(y1 + 1, min(h, int(coords[3])))
                        bw = x2 - x1
                        bh = y2 - y1

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
                            "source": "trained_yolov8n"
                        }
                        detected_fields.append(field_item)

                        crop, crop_meta = extract_padded_crop(img, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, pad_ratio=0.08)
                        if crop is not None and crop.size > 0:
                            if cls_name not in field_crops or conf > field_crops[cls_name]["confidence"]:
                                field_crops[cls_name] = {
                                    "crop": crop,
                                    "confidence": conf,
                                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                    "padded_bbox": crop_meta.get("padded_bbox"),
                                    "source": "trained_yolov8n"
                                }

                    detected_fields.sort(key=lambda x: x["confidence"], reverse=True)

                    return {
                        "status": "SUCCESS",
                        "detector_source": "trained_yolov8n",
                        "model_name": "pan_field_detector",
                        "model_type": "yolov8n_trained",
                        "fallback": False,
                        "fields": detected_fields,
                        "field_crops": field_crops,
                        "image_dims": [w, h]
                    }

            except Exception as e:
                print(f"[VISION WARNING] YOLO PAN detector failed: {e}. Executing layout fallback.")

        # Fallback to standardized layout heuristics
        return self._generate_layout_heuristic_fields(img, w, h)


def get_pan_field_detector() -> PANFieldDetector:
    """Returns singleton instance of the PAN Field Detector."""
    global _PAN_DETECTOR_INSTANCE
    if _PAN_DETECTOR_INSTANCE is None:
        _PAN_DETECTOR_INSTANCE = PANFieldDetector()
    return _PAN_DETECTOR_INSTANCE
