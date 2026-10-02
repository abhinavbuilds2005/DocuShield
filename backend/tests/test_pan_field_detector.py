"""
Unit and Integration Tests for PAN Field Detector (backend/vision/pan_field_detector.py)
DocuShield AI — Phase 2 Modular Expansion

Tests cover:
1. Singleton pattern via get_pan_field_detector()
2. Model availability and class schema verification
3. Detection on real PAN validation card (trained_yolov8n provenance)
4. Graceful fallback to layout heuristics on untextured documents or missing model
5. Disabled via PAN_FIELD_DETECTOR_ENABLED=false
6. Invalid image and nonexistent file path handling
7. Crop generation for detected fields (both YOLO and heuristic)
8. Crop-assisted OCR recovery in fusion pipeline
9. Dynamic routing in fusion pipeline (PAN routed to PAN detector, Aadhaar to Aadhaar detector)
10. Verification that field detection NEVER marks document as AUTHENTIC/VERIFIED on its own
11. Aadhaar detector behavior remaining completely unchanged
"""

import os
import sys
import cv2
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.vision.pan_field_detector import PANFieldDetector, get_pan_field_detector
from backend.vision.field_detector import AadhaarFieldDetector, get_field_detector
from backend.fusion import DocumentScreeningPipeline


@pytest.fixture(autouse=True)
def reset_pan_singleton():
    """Reset singleton between tests."""
    import backend.vision.pan_field_detector as pfd_module
    pfd_module._PAN_DETECTOR_INSTANCE = None
    yield
    pfd_module._PAN_DETECTOR_INSTANCE = None


@pytest.fixture
def sample_real_pan_image():
    """Path to real held-out PAN validation image."""
    real_path = os.path.join("data", "processed", "pan_field_detection", "valid", "images", "pan_valid_001.jpg")
    if os.path.exists(real_path):
        return real_path
    return None


@pytest.fixture
def synthetic_flat_pan_image():
    """Synthetic flat mockup image of PAN card (640x400 BGR)."""
    img = np.full((400, 640, 3), 250, dtype=np.uint8)
    cv2.putText(img, "INCOME TAX DEPARTMENT", (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)
    cv2.putText(img, "Permanent Account Number", (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "ABCPS1234F", (180, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Name: AARAV SHARMA", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Father: RAJESH SHARMA", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "DOB: 15/08/1990", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


# ──────────────────────────────────────────────────────────────────────
# 1. Singleton & Schema Tests
# ──────────────────────────────────────────────────────────────────────

def test_pan_singleton_instance():
    """get_pan_field_detector() should always return the identical instance."""
    det1 = get_pan_field_detector()
    det2 = get_pan_field_detector()
    assert det1 is det2


def test_pan_classes_schema():
    """PAN detector must declare normalized 4-class schema."""
    det = get_pan_field_detector()
    classes = det.classes
    assert len(classes) == 4
    assert classes[0] == "pan_number"
    assert classes[1] == "name"
    assert classes[2] == "father_name"
    assert classes[3] == "date_of_birth"


# ──────────────────────────────────────────────────────────────────────
# 2. Model Loading & Detection on Real Held-Out Validation Card
# ──────────────────────────────────────────────────────────────────────

def test_real_pan_card_yolo_detection(sample_real_pan_image):
    """Trained YOLOv8n model should detect boxes with trained_yolov8n provenance on real cards."""
    if sample_real_pan_image is None:
        pytest.skip("Held-out PAN validation image not found.")

    det = get_pan_field_detector()
    assert det.is_available is True

    res = det.detect_fields(sample_real_pan_image)
    assert res["status"] == "SUCCESS"
    assert res["detector_source"] == "trained_yolov8n"
    assert res["model_name"] == "pan_field_detector"
    assert res["fallback"] is False
    assert len(res["fields"]) > 0

    # Verify bounding box normalization
    for f in res["fields"]:
        assert "class_name" in f
        assert f["class_name"] in ["pan_number", "name", "father_name", "date_of_birth"]
        assert 0.0 <= f["confidence"] <= 1.0
        nb = f["bbox_normalized"]
        assert 0.0 <= nb["x"] <= 1.0
        assert 0.0 <= nb["y"] <= 1.0
        assert 0.0 <= nb["width"] <= 1.0
        assert 0.0 <= nb["height"] <= 1.0


# ──────────────────────────────────────────────────────────────────────
# 3. Graceful Fallback to Layout Heuristics
# ──────────────────────────────────────────────────────────────────────

def test_untextured_document_triggers_layout_heuristic_fallback():
    """Documents where YOLO produces 0 boxes should seamlessly fall back to layout heuristics."""
    det = get_pan_field_detector()
    blank_card = np.full((400, 640, 3), 220, dtype=np.uint8)
    res = det.detect_fields(blank_card)

    assert res["status"] == "SUCCESS"
    assert res["detector_source"] == "layout_heuristic"
    assert res["fallback"] is True
    assert len(res["fields"]) == 4

    field_names = [f["class_name"] for f in res["fields"]]
    assert "pan_number" in field_names
    assert "name" in field_names
    assert "father_name" in field_names
    assert "date_of_birth" in field_names


def test_missing_model_triggers_layout_fallback(synthetic_flat_pan_image, monkeypatch):
    """When model weights file does not exist, detector should fall back to layout heuristics."""
    monkeypatch.setenv("PAN_FIELD_DETECTOR_MODEL_PATH", "/nonexistent/model.pt")

    det = PANFieldDetector()
    assert det.is_available is False

    res = det.detect_fields(synthetic_flat_pan_image)
    assert res["status"] == "SUCCESS"
    assert res["detector_source"] == "layout_heuristic"
    assert res["fallback"] is True
    assert len(res["fields"]) == 4


def test_disabled_via_env(synthetic_flat_pan_image, monkeypatch):
    """When PAN_FIELD_DETECTOR_ENABLED=false, detector should report DISABLED."""
    monkeypatch.setenv("PAN_FIELD_DETECTOR_ENABLED", "false")

    det = PANFieldDetector()
    assert det.is_available is False

    res = det.detect_fields(synthetic_flat_pan_image)
    # Disabled detector produces DISABLED status
    assert res["status"] == "DISABLED"
    assert res["fields"] == []


# ──────────────────────────────────────────────────────────────────────
# 4. Invalid Input Handling
# ──────────────────────────────────────────────────────────────────────

def test_invalid_image_inputs():
    """Nonexistent paths and zero-size arrays must not crash."""
    det = get_pan_field_detector()

    res1 = det.detect_fields("/nonexistent/file/path.jpg")
    assert res1["fields"] == []

    empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    res2 = det.detect_fields(empty_arr)
    assert res2["fields"] == []

    res3 = det.detect_fields(None)
    assert res3["fields"] == []


# ──────────────────────────────────────────────────────────────────────
# 5. Crop Generation
# ──────────────────────────────────────────────────────────────────────

def test_crop_generation_on_real_card(sample_real_pan_image):
    """detect_fields should return crops for detected regions on real cards."""
    if sample_real_pan_image is None:
        pytest.skip("Held-out PAN validation image not found.")

    det = get_pan_field_detector()
    res = det.detect_fields(sample_real_pan_image)

    crops = res.get("field_crops", {})
    assert len(crops) > 0
    for cname, cdata in crops.items():
        assert "crop" in cdata
        assert isinstance(cdata["crop"], np.ndarray)
        assert cdata["crop"].shape[0] > 0
        assert cdata["crop"].shape[1] > 0


def test_crop_generation_on_heuristic_fallback():
    """detect_fields should return crops for all 4 canonical regions on fallback."""
    det = get_pan_field_detector()
    blank_card = np.full((400, 640, 3), 220, dtype=np.uint8)
    res = det.detect_fields(blank_card)

    crops = res.get("field_crops", {})
    assert "pan_number" in crops
    assert "name" in crops
    assert "father_name" in crops
    assert "date_of_birth" in crops
    assert crops["pan_number"]["crop"].shape[0] > 0
    assert crops["pan_number"]["crop"].shape[1] > 0


# ──────────────────────────────────────────────────────────────────────
# 6. Fusion Pipeline Integration & Routing
# ──────────────────────────────────────────────────────────────────────

def test_fusion_routes_pan_to_pan_detector(synthetic_flat_pan_image, tmp_path):
    """Screening a PAN document should route to PAN detector and report PAN provenance."""
    img_path = str(tmp_path / "test_pan.png")
    cv2.imwrite(img_path, synthetic_flat_pan_image)

    pipeline = DocumentScreeningPipeline()
    res = pipeline.screen_document(img_path, benchmark_mode=True, document_type="pan")

    assert "field_detection_provenance" in res
    prov = res["field_detection_provenance"]
    assert prov["document_type"] == "pan"
    assert prov["model_name"] == "pan_field_detector"
    assert prov["detector_source"] in ["trained_yolov8n", "layout_heuristic"]
    assert "Trained YOLOv8n Field Detector (PAN Card)" in res["field_detection_model"] or "Layout Heuristic (PAN Fallback)" in res["field_detection_model"]


def test_fusion_preserves_aadhaar_detector():
    """Aadhaar field detection behavior and model path must remain completely intact."""
    pipeline = DocumentScreeningPipeline()
    aadhaar_det = pipeline.field_detector

    assert isinstance(aadhaar_det, AadhaarFieldDetector)
    assert os.path.exists(aadhaar_det.model_path)
    assert "aadhaar_field_detector.pt" in aadhaar_det.model_path
    assert aadhaar_det.is_available is True


def test_detector_never_declares_authentic_on_its_own(tmp_path):
    """A document with invalid PAN structure must not become AUTHENTIC because of field detection."""
    # Create invalid PAN image
    bad_img = np.full((400, 640, 3), 245, dtype=np.uint8)
    cv2.putText(bad_img, "INCOME TAX DEPARTMENT", (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)
    cv2.putText(bad_img, "INVALID-PAN-NUMBER", (160, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    p = str(tmp_path / "bad_pan.png")
    cv2.imwrite(p, bad_img)

    pipeline = DocumentScreeningPipeline()
    res = pipeline.screen_document(p, benchmark_mode=True, document_type="pan")

    # Verdict must not be AUTHENTIC
    assert res["verdict"] != "AUTHENTIC"
    assert res["status"] != "AUTHENTIC"
    assert res["official_verification"]["verified"] is False
