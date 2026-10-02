"""
Indian Driving Licence Field Detection Module (YOLOv8n + Sarathi Layout Heuristic Fallback)
DocuShield AI — Phase 2 Part 3 Modular Expansion

Target Fields:
- licence_number (Class 0)
- date_of_birth (Class 1)
- name (Class 2)

Architectural Guardrails:
1. Crop Assistance Only: Operates strictly as a crop-assistance and localization module.
   Never independently dictates authenticity, validity, or verification status.
2. Conservative Gating: Discards YOLO detections with confidence < 0.50.
3. Two-Field Fallback Rule: If fewer than 2 distinct reliable fields are localized,
   automatically and seamlessly falls back to canonical Sarathi / MoRTH layout heuristics.
4. Provenance Transparency: Returns clear metadata ('dl_yolo' vs 'dl_layout_heuristic',
   'fallback': bool, 'fallback_reason': str, 'experimental': True).
5. Safe Clamping: All bounding boxes are [x, y, width, height] clamped to [0, W] and [0, H].
6. Privacy Protection: Never logs raw licence numbers, names, or birth dates.
"""

import os
import re
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.vision.crop_utils import extract_padded_crop

_DL_DETECTOR_INSTANCE = None


class DLFieldDetector:
    """
    Dedicated modular YOLOv8n detector and crop generator for Indian Driving Licences.
    Includes deterministic fallback to Sarathi / MoRTH layout heuristics.
    """

    CLASS_NAMES = {
        0: "licence_number",
        1: "date_of_birth",
        2: "name"
    }

    def __init__(self):
        self.enabled = os.environ.get("DL_FIELD_DETECTOR_ENABLED", "true").strip().lower() in ("true", "1", "yes")
        default_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "dl_field_detector.pt"
        )
        self.model_path = os.environ.get("DL_FIELD_DETECTOR_MODEL_PATH", default_model_path)
        self.conf_threshold = float(os.environ.get("DL_FIELD_DETECTOR_CONFIDENCE", "0.50"))
        self.device_pref = os.environ.get("DL_FIELD_DETECTOR_DEVICE", "auto").strip().lower()

        self._model = None
        self._init_attempted = False
        self._is_available = False
        self._device = "cpu"
        self._load_error_reason = None

    def _lazy_init(self):
        """Lazy loads YOLOv8n DL detector model on first request (cached singleton)."""
        if self._init_attempted:
            return
        self._init_attempted = True

        if not self.enabled:
            print("[VISION] DL field detector disabled via DL_FIELD_DETECTOR_ENABLED=false")
            self._load_error_reason = "detector_disabled"
            self._is_available = False
            return

        if not os.path.exists(self.model_path):
            print(f"[VISION WARNING] DL field detector weights not found at: {self.model_path}. Using layout fallback.")
            self._load_error_reason = "model_file_missing"
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

            print(f"[VISION] Loading DL field detector from: {self.model_path} (device: {device})...")
            self._model = YOLO(self.model_path)
            self._device = device
            self._is_available = True
            self._load_error_reason = None
            print("[VISION] DL field detector successfully initialized.")
        except Exception as e:
            print(f"[VISION ERROR] Failed to initialize YOLO DL field detector: {e}. Falling back to layout heuristic.")
            self._model = None
            self._is_available = False
            self._load_error_reason = f"model_loading_failed: {str(e)}"

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

    def _generate_heuristic_fields(self, img: np.ndarray, reason: str) -> Dict[str, Any]:
        """
        Generates canonical fallback bounding boxes based on Sarathi / MoRTH standardized geometry.
        Used when the YOLO model is missing, throws an exception, or produces < 2 high-confidence fields.
        """
        h, w = img.shape[:2]
        
        # Canonical proportions for standardized landscape Indian Driving Licence (Sarathi / Smart Card):
        # 85.6mm x 53.98mm (aspect ratio ~ 1.58)
        # Class 0: licence_number - Upper-right / center-top header zone
        # Class 1: date_of_birth  - Left-center below name
        # Class 2: name           - Left-center below state header
        heuristic_specs = [
            (0, "licence_number", 0.32, 0.16, 0.62, 0.12, 0.65),
            (2, "name",           0.24, 0.32, 0.54, 0.12, 0.65),
            (1, "date_of_birth",  0.24, 0.48, 0.42, 0.11, 0.65)
        ]

        fields = []
        field_crops = {}

        for cls_id, cls_name, xr, yr, wr, hr, conf in heuristic_specs:
            x1 = max(0, min(w - 1, int(xr * w)))
            y1 = max(0, min(h - 1, int(yr * h)))
            x2 = max(x1 + 1, min(w, int((xr + wr) * w)))
            y2 = max(y1 + 1, min(h, int((yr + hr) * h)))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            norm_box = {
                "x": round(x1 / w, 4),
                "y": round(y1 / h, 4),
                "width": round(bw / w, 4),
                "height": round(bh / h, 4)
            }

            field_item = {
                "field": cls_name,
                "class_name": cls_name,
                "class_id": cls_id,
                "confidence": conf,
                "box": [x1, y1, bw, bh],
                "bbox": [x1, y1, bw, bh],
                "bbox_dict": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "bbox_normalized": norm_box,
                "source": "dl_layout_heuristic"
            }
            fields.append(field_item)

            crop, crop_meta = extract_padded_crop(img, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, pad_ratio=0.08)
            if crop is not None and crop.size > 0:
                field_crops[cls_name] = {
                    "crop": crop,
                    "confidence": conf,
                    "box": [x1, y1, bw, bh],
                    "bbox": [x1, y1, bw, bh],
                    "padded_bbox": crop_meta.get("padded_bbox"),
                    "source": "dl_layout_heuristic"
                }

        return {
            "status": "SUCCESS",
            "detector_source": "dl_layout_heuristic",
            "model_name": "dl_field_detector.pt",
            "fallback": True,
            "fallback_reason": reason,
            "document_type": "DRIVING_LICENCE",
            "experimental": True,
            "fields": fields,
            "field_crops": field_crops,
            "image_dims": [w, h],
            "note": f"Field boxes generated via canonical Sarathi layout heuristic ({reason})."
        }

    def detect_fields(self, image_path_or_array, iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Detects Indian Driving Licence fields with conservative gating (conf >= 0.50)
        and mandatory fallback if fewer than 2 valid fields are localized.

        Returns:
            Dict containing detected fields, field crops, and provenance metadata.
        """
        self._lazy_init()

        # 1. Load and validate image input
        if isinstance(image_path_or_array, (str, Path)):
            path_str = str(image_path_or_array)
            if not os.path.exists(path_str):
                return {
                    "status": "ERROR",
                    "detector_source": "error",
                    "model_name": "dl_field_detector.pt",
                    "fallback": True,
                    "fallback_reason": "image_file_not_found",
                    "document_type": "DRIVING_LICENCE",
                    "experimental": True,
                    "fields": [],
                    "field_crops": {}
                }
            img = cv2.imread(path_str)
        elif isinstance(image_path_or_array, np.ndarray):
            img = image_path_or_array.copy()
        else:
            return {
                "status": "ERROR",
                "detector_source": "error",
                "model_name": "dl_field_detector.pt",
                "fallback": True,
                "fallback_reason": "invalid_image_type",
                "document_type": "DRIVING_LICENCE",
                "experimental": True,
                "fields": [],
                "field_crops": {}
            }

        if img is None or img.size == 0 or len(img.shape) < 2:
            return {
                "status": "ERROR",
                "detector_source": "error",
                "model_name": "dl_field_detector.pt",
                "fallback": True,
                "fallback_reason": "unreadable_image_array",
                "document_type": "DRIVING_LICENCE",
                "experimental": True,
                "fields": [],
                "field_crops": {}
            }

        h, w = img.shape[:2]

        # 2. Image quality / suitability check
        if min(h, w) < 50:
            return self._generate_heuristic_fields(img, reason="unsuitable_image_quality (dimensions too small)")

        # 3. Check model availability; fallback if unavailable
        if not self._is_available or self._model is None:
            reason = getattr(self, "_load_error_reason", None) or "model_unavailable"
            return self._generate_heuristic_fields(img, reason=reason)

        # 3. Perform YOLO inference safely
        try:
            results = self._model.predict(
                source=img,
                conf=0.25,          # Low threshold for raw detection, strict filtering below
                iou=iou_threshold,
                device=self._device,
                verbose=False,
                rect=True
            )
        except Exception as inf_err:
            print(f"[VISION ERROR] DL YOLO inference failed: {inf_err}. Engaging layout fallback.")
            return self._generate_heuristic_fields(img, reason=f"inference_exception: {str(inf_err)}")

        if not results or len(results) == 0:
            return self._generate_heuristic_fields(img, reason="zero_raw_detections")

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return self._generate_heuristic_fields(img, reason="zero_bounding_boxes")

        # 4. Filter and process detections
        yolo_fields = []
        field_crops = {}
        distinct_classes = set()

        for box in boxes:
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())

            # CONSERVATIVE GATING: Reject detections below threshold (0.50)
            if conf < self.conf_threshold:
                continue

            if cls_id not in self.CLASS_NAMES:
                continue

            cls_name = self.CLASS_NAMES[cls_id]

            # Parse coordinates in pixels
            xyxy = box.xyxy[0].tolist()
            x1 = max(0, min(w - 1, int(xyxy[0])))
            y1 = max(0, min(h - 1, int(xyxy[1])))
            x2 = max(x1 + 1, min(w, int(xyxy[2])))
            y2 = max(y1 + 1, min(h, int(xyxy[3])))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # Validate positive dimensions
            if bw <= 0 or bh <= 0:
                continue

            norm_box = {
                "x": round(x1 / w, 4),
                "y": round(y1 / h, 4),
                "width": round(bw / w, 4),
                "height": round(bh / h, 4)
            }

            field_item = {
                "field": cls_name,
                "class_name": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "box": [x1, y1, bw, bh],
                "bbox": [x1, y1, bw, bh],
                "bbox_dict": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "bbox_normalized": norm_box,
                "source": "dl_yolo"
            }
            yolo_fields.append(field_item)
            distinct_classes.add(cls_id)

            # Generate field crop for OCR with safe padding
            crop, crop_meta = extract_padded_crop(img, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, pad_ratio=0.08)
            if crop is not None and crop.size > 0:
                # Retain highest-confidence crop per class
                if cls_name not in field_crops or conf > field_crops[cls_name]["confidence"]:
                    field_crops[cls_name] = {
                        "crop": crop,
                        "confidence": round(conf, 4),
                        "box": [x1, y1, bw, bh],
                        "bbox": [x1, y1, bw, bh],
                        "padded_bbox": crop_meta.get("padded_bbox"),
                        "source": "dl_yolo"
                    }

        # 5. TWO-FIELD FALLBACK RULE:
        # If fewer than 2 valid semantic fields are detected with confidence >= 0.50,
        # fallback to the layout heuristic path.
        if len(distinct_classes) < 2:
            return self._generate_heuristic_fields(
                img,
                reason=f"insufficient_reliable_fields (found {len(distinct_classes)} classes >= {self.conf_threshold})"
            )

        # 6. Return successful high-confidence YOLO field localization
        return {
            "status": "SUCCESS",
            "detector_source": "dl_yolo",
            "model_name": "dl_field_detector.pt",
            "fallback": False,
            "fallback_reason": None,
            "document_type": "DRIVING_LICENCE",
            "experimental": True,
            "fields": yolo_fields,
            "field_crops": field_crops,
            "image_dims": [w, h],
            "note": f"Localized {len(yolo_fields)} fields via trained YOLOv8n detector."
        }


def get_dl_field_detector() -> DLFieldDetector:
    """Returns singleton instance of DLFieldDetector."""
    global _DL_DETECTOR_INSTANCE
    if _DL_DETECTOR_INSTANCE is None:
        _DL_DETECTOR_INSTANCE = DLFieldDetector()
    return _DL_DETECTOR_INSTANCE
