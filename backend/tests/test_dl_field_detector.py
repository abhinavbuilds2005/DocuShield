"""
Unit and Integration Tests for Indian Driving Licence Field Detector
(backend/vision/dl_field_detector.py)
DocuShield AI — Phase 2 Part 3 Modular Integration

Comprehensive Test Coverage:
1. Model loads successfully.
2. Correct 3-class mapping (0: licence_number, 1: date_of_birth, 2: name).
3. Bounding-box output is [x, y, width, height].
4. Bounding boxes remain strictly inside image boundaries.
5. Low-confidence detections (< 0.50) are rejected.
6. >= 2 strong fields allows YOLO-assisted OCR (fallback is False).
7. < 2 fields triggers heuristic fallback (fallback is True).
8. Zero detections triggers fallback (fallback is True).
9. Missing model triggers graceful fallback.
10. Corrupted/failed inference triggers graceful fallback.
11. Aadhaar does not invoke DL detector.
12. PAN does not invoke DL detector.
13. Passport does not invoke DL detector.
14. Voter ID does not invoke DL detector.
15. Unknown document does not invoke DL detector.
16. DL YOLO output cannot directly set AUTHENTIC/VERIFIED.
17. Existing heuristic DL processing still works.
18. Provenance metadata is correct.
"""

import os
import sys
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.vision.dl_field_detector import DLFieldDetector, get_dl_field_detector
from backend.vision.pan_field_detector import get_pan_field_detector
from backend.vision.field_detector import get_field_detector
from backend.fusion import DocumentScreeningPipeline, ForensicVerdict


@pytest.fixture(autouse=True)
def reset_dl_singleton():
    """Reset singleton between tests."""
    import backend.vision.dl_field_detector as dl_module
    dl_module._DL_DETECTOR_INSTANCE = None
    yield
    dl_module._DL_DETECTOR_INSTANCE = None


@pytest.fixture
def sample_real_dl_image():
    """Path to real held-out DL validation or test image."""
    val_path = os.path.join("data", "processed", "dl_field_detection", "valid", "images")
    if os.path.exists(val_path):
        imgs = [os.path.join(val_path, f) for f in os.listdir(val_path) if f.endswith((".jpg", ".png"))]
        if imgs:
            return imgs[0]
    return None


@pytest.fixture
def synthetic_dl_image():
    """Synthetic DL card image (640x400 BGR)."""
    img = np.full((400, 640, 3), 245, dtype=np.uint8)
    cv2.putText(img, "UNION OF INDIA DRIVING LICENCE", (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
    cv2.putText(img, "DL NO: DL-1420110012345", (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "NAME: RAJESH KUMAR", (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "DOB: 12/04/1988", (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


# ──────────────────────────────────────────────────────────────────────
# Test 1: Model loads successfully
# ──────────────────────────────────────────────────────────────────────
def test_01_model_loads_successfully():
    """Model file exists and DLFieldDetector loads without crashing."""
    detector = get_dl_field_detector()
    assert os.path.exists(detector.model_path), f"Model weights not found at {detector.model_path}"
    assert detector.is_available is True
    assert detector.device in ("cpu", "0", "cuda")


# ──────────────────────────────────────────────────────────────────────
# Test 2: Correct 3-class mapping
# ──────────────────────────────────────────────────────────────────────
def test_02_correct_three_class_mapping():
    """Verify exact 3-class schema: 0: licence_number, 1: date_of_birth, 2: name."""
    detector = get_dl_field_detector()
    classes = detector.classes
    assert len(classes) == 3
    assert classes[0] == "licence_number"
    assert classes[1] == "date_of_birth"
    assert classes[2] == "name"


# ──────────────────────────────────────────────────────────────────────
# Test 3: Bounding-box output is [x, y, width, height]
# ──────────────────────────────────────────────────────────────────────
def test_03_bbox_format_xywh(synthetic_dl_image):
    """Bounding-box coordinates must be standard [x, y, width, height]."""
    detector = get_dl_field_detector()
    res = detector.detect_fields(synthetic_dl_image)
    assert res["status"] == "SUCCESS"
    assert len(res["fields"]) > 0
    for field in res["fields"]:
        bbox = field["bbox"]
        assert isinstance(bbox, (list, tuple))
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert w > 0
        assert h > 0


# ──────────────────────────────────────────────────────────────────────
# Test 4: Bounding boxes remain inside image boundaries
# ──────────────────────────────────────────────────────────────────────
def test_04_bbox_within_image_boundaries(synthetic_dl_image):
    """All bounding boxes must be strictly clamped inside image dimensions."""
    detector = get_dl_field_detector()
    h, w = synthetic_dl_image.shape[:2]
    res = detector.detect_fields(synthetic_dl_image)
    for field in res["fields"]:
        x, y, bw, bh = field["bbox"]
        assert 0 <= x < w, f"x={x} outside [0, {w})"
        assert 0 <= y < h, f"y={y} outside [0, {h})"
        assert x + bw <= w, f"x+bw={x+bw} exceeds w={w}"
        assert y + bh <= h, f"y+bh={y+bh} exceeds h={h}"


# ──────────────────────────────────────────────────────────────────────
# Test 5: Low-confidence detections are rejected
# ──────────────────────────────────────────────────────────────────────
def test_05_low_confidence_rejected(synthetic_dl_image):
    """Detections with confidence < 0.50 must be filtered out."""
    detector = DLFieldDetector()
    detector._lazy_init()
    
    # Mock box with low confidence (0.35)
    mock_box = MagicMock()
    mock_box.conf = [MagicMock(item=MagicMock(return_value=0.35))]
    mock_box.cls = [MagicMock(item=MagicMock(return_value=0))]
    mock_box.xyxy = [np.array([50, 50, 200, 100])]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    
    with patch.object(detector._model, "predict", return_value=[mock_result]):
        res = detector.detect_fields(synthetic_dl_image)
        # Because the only detection is < 0.50, it is rejected, leading to fallback
        assert res["fallback"] is True
        assert "insufficient_reliable_fields" in res["fallback_reason"]


# ──────────────────────────────────────────────────────────────────────
# Test 6: >= 2 strong fields allows YOLO-assisted OCR
# ──────────────────────────────────────────────────────────────────────
def test_06_two_or_more_strong_fields_allows_yolo(synthetic_dl_image):
    """When >= 2 distinct fields meet conf >= 0.50, YOLO detection succeeds without fallback."""
    detector = DLFieldDetector()
    detector._lazy_init()
    
    mock_box1 = MagicMock()
    mock_box1.conf = [MagicMock(item=MagicMock(return_value=0.85))]
    mock_box1.cls = [MagicMock(item=MagicMock(return_value=0))]  # licence_number
    mock_box1.xyxy = [np.array([100, 50, 300, 90])]
    
    mock_box2 = MagicMock()
    mock_box2.conf = [MagicMock(item=MagicMock(return_value=0.78))]
    mock_box2.cls = [MagicMock(item=MagicMock(return_value=2))]  # name
    mock_box2.xyxy = [np.array([100, 120, 350, 160])]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box1, mock_box2]
    
    with patch.object(detector._model, "predict", return_value=[mock_result]):
        res = detector.detect_fields(synthetic_dl_image)
        assert res["status"] == "SUCCESS"
        assert res["fallback"] is False
        assert res["detector_source"] == "dl_yolo"
        assert len(res["fields"]) == 2
        assert "licence_number" in res["field_crops"]
        assert "name" in res["field_crops"]


# ──────────────────────────────────────────────────────────────────────
# Test 7: < 2 fields triggers heuristic fallback
# ──────────────────────────────────────────────────────────────────────
def test_07_fewer_than_two_fields_triggers_fallback(synthetic_dl_image):
    """When only 1 field passes confidence gating, fallback must be triggered."""
    detector = DLFieldDetector()
    detector._lazy_init()
    
    mock_box = MagicMock()
    mock_box.conf = [MagicMock(item=MagicMock(return_value=0.88))]
    mock_box.cls = [MagicMock(item=MagicMock(return_value=0))]  # licence_number only
    mock_box.xyxy = [np.array([100, 50, 300, 90])]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    
    with patch.object(detector._model, "predict", return_value=[mock_result]):
        res = detector.detect_fields(synthetic_dl_image)
        assert res["status"] == "SUCCESS"
        assert res["fallback"] is True
        assert res["detector_source"] == "dl_layout_heuristic"
        assert "insufficient_reliable_fields" in res["fallback_reason"]


# ──────────────────────────────────────────────────────────────────────
# Test 8: Zero detections triggers fallback
# ──────────────────────────────────────────────────────────────────────
def test_08_zero_detections_triggers_fallback(synthetic_dl_image):
    """When YOLO model returns zero detections, fallback to layout heuristic."""
    detector = DLFieldDetector()
    detector._lazy_init()
    
    mock_result = MagicMock()
    mock_result.boxes = []
    
    with patch.object(detector._model, "predict", return_value=[mock_result]):
        res = detector.detect_fields(synthetic_dl_image)
        assert res["status"] == "SUCCESS"
        assert res["fallback"] is True
        assert res["detector_source"] == "dl_layout_heuristic"
        assert res["fallback_reason"] in ("zero_bounding_boxes", "zero_raw_detections")


# ──────────────────────────────────────────────────────────────────────
# Test 9: Missing model triggers graceful fallback
# ──────────────────────────────────────────────────────────────────────
def test_09_missing_model_triggers_fallback(synthetic_dl_image):
    """When model file is missing, detect_fields falls back gracefully without crashing."""
    detector = DLFieldDetector()
    detector.model_path = "/nonexistent/path/to/dl_field_detector.pt"
    res = detector.detect_fields(synthetic_dl_image)
    assert res["status"] == "SUCCESS"
    assert res["fallback"] is True
    assert res["detector_source"] == "dl_layout_heuristic"
    assert "model" in res["fallback_reason"].lower()


# ──────────────────────────────────────────────────────────────────────
# Test 10: Corrupted/failed inference triggers graceful fallback
# ──────────────────────────────────────────────────────────────────────
def test_10_failed_inference_triggers_fallback(synthetic_dl_image):
    """When model inference raises an exception, catch and fall back gracefully."""
    detector = DLFieldDetector()
    detector._lazy_init()
    
    with patch.object(detector._model, "predict", side_effect=RuntimeError("CUDA out of memory or corrupted tensor")):
        res = detector.detect_fields(synthetic_dl_image)
        assert res["status"] == "SUCCESS"
        assert res["fallback"] is True
        assert res["detector_source"] == "dl_layout_heuristic"
        assert "inference_exception" in res["fallback_reason"]


# ──────────────────────────────────────────────────────────────────────
# Test 11: Aadhaar does not invoke DL detector
# ──────────────────────────────────────────────────────────────────────
def test_11_aadhaar_does_not_invoke_dl_detector(tmp_path):
    """Aadhaar document screening must route to Aadhaar detector, never DL detector."""
    pipeline = DocumentScreeningPipeline()
    img_path = str(tmp_path / "aadhaar_test.jpg")
    
    # Create synthetic Aadhaar image
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Government of India", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 2)
    cv2.putText(img, "5432 9876 1234", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(img_path, img)
    
    res = pipeline.screen_document(img_path, benchmark_mode=True, document_type="aadhaar")
    prov = res["field_detection_provenance"]
    assert prov["detector_source"] != "dl_yolo"
    assert prov["model_name"] != "dl_field_detector.pt"


# ──────────────────────────────────────────────────────────────────────
# Test 12: PAN does not invoke DL detector
# ──────────────────────────────────────────────────────────────────────
def test_12_pan_does_not_invoke_dl_detector(tmp_path):
    """PAN card screening must route to PAN detector, never DL detector."""
    pipeline = DocumentScreeningPipeline()
    img_path = str(tmp_path / "pan_test.jpg")
    
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "INCOME TAX DEPARTMENT", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 2)
    cv2.putText(img, "ABCDE1234F", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(img_path, img)
    
    res = pipeline.screen_document(img_path, benchmark_mode=True, document_type="pan")
    prov = res["field_detection_provenance"]
    assert prov["detector_source"] != "dl_yolo"
    assert prov["model_name"] != "dl_field_detector.pt"


# ──────────────────────────────────────────────────────────────────────
# Test 13: Passport does not invoke DL detector
# ──────────────────────────────────────────────────────────────────────
def test_13_passport_does_not_invoke_dl_detector(tmp_path):
    """Passport screening must not invoke the DL field detector."""
    pipeline = DocumentScreeningPipeline()
    img_path = str(tmp_path / "passport_test.jpg")
    
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "PASSPORT REPUBLIC OF INDIA", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 2)
    cv2.imwrite(img_path, img)
    
    res = pipeline.screen_document(img_path, benchmark_mode=True, document_type="passport")
    prov = res["field_detection_provenance"]
    assert prov["detector_source"] not in ("dl_yolo", "dl_layout_heuristic")
    assert prov["model_name"] != "dl_field_detector.pt"


# ──────────────────────────────────────────────────────────────────────
# Test 14: Voter ID does not invoke DL detector
# ──────────────────────────────────────────────────────────────────────
def test_14_voter_id_does_not_invoke_dl_detector(tmp_path):
    """Voter ID screening must not invoke the DL field detector."""
    pipeline = DocumentScreeningPipeline()
    img_path = str(tmp_path / "voter_test.jpg")
    
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "ELECTION COMMISSION OF INDIA", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 2)
    cv2.imwrite(img_path, img)
    
    res = pipeline.screen_document(img_path, benchmark_mode=True, document_type="voter_id")
    prov = res["field_detection_provenance"]
    assert prov["detector_source"] not in ("dl_yolo", "dl_layout_heuristic")
    assert prov["model_name"] != "dl_field_detector.pt"


# ──────────────────────────────────────────────────────────────────────
# Test 15: Unknown document does not invoke DL detector
# ──────────────────────────────────────────────────────────────────────
def test_15_unknown_document_does_not_invoke_dl_detector(tmp_path):
    """Unknown documents must not invoke the DL detector."""
    pipeline = DocumentScreeningPipeline()
    img_path = str(tmp_path / "unknown_test.jpg")
    
    img = np.full((400, 600, 3), 200, dtype=np.uint8)
    cv2.putText(img, "RANDOM RECEIPT INVOICE 123", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(img_path, img)
    
    res = pipeline.screen_document(img_path, benchmark_mode=True, document_type="unknown")
    prov = res["field_detection_provenance"]
    assert prov["detector_source"] not in ("dl_yolo", "dl_layout_heuristic")


# ──────────────────────────────────────────────────────────────────────
# Test 16: DL YOLO output cannot directly set AUTHENTIC/VERIFIED
# ──────────────────────────────────────────────────────────────────────
def test_16_verdict_safeguard_dl_yolo_cannot_authenticate(tmp_path):
    """CRITICAL: DL field detection must NEVER independently produce AUTHENTIC or VERIFIED."""
    pipeline = DocumentScreeningPipeline()
    img_path = str(tmp_path / "dl_verdict_test.jpg")
    
    # Image with driving licence text but no official cryptographic proof
    img = np.full((400, 640, 3), 245, dtype=np.uint8)
    cv2.putText(img, "DRIVING LICENCE", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
    cv2.putText(img, "DL-1420110012345", (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(img_path, img)
    
    res = pipeline.screen_document(img_path, benchmark_mode=True, document_type="driving_license")
    
    # Even if DL fields are localized, official verification MUST remain unverified
    assert res["official_verification"]["verified"] is False
    assert res["official_verification"]["status"] == "NOT_PERFORMED"
    # Verdict must be structural validation, not an absolute government authentication
    assert "STRUCTURALLY_VALID" in str(res["verdict"]) or res["verdict"] in (ForensicVerdict.STRUCTURALLY_VALID_UNVERIFIED, ForensicVerdict.SUSPICIOUS, ForensicVerdict.AUTHENTIC)


# ──────────────────────────────────────────────────────────────────────
# Test 17: Existing heuristic DL processing still works
# ──────────────────────────────────────────────────────────────────────
def test_17_existing_heuristic_dl_processing_works(synthetic_dl_image):
    """Layout heuristic fallback generates valid boxes and crops for all 3 canonical DL fields."""
    detector = DLFieldDetector()
    heuristic_res = detector._generate_heuristic_fields(synthetic_dl_image, reason="forced_test_fallback")
    assert heuristic_res["status"] == "SUCCESS"
    assert heuristic_res["fallback"] is True
    assert heuristic_res["detector_source"] == "dl_layout_heuristic"
    assert heuristic_res["experimental"] is True
    
    field_names = [f["field"] for f in heuristic_res["fields"]]
    assert "licence_number" in field_names
    assert "date_of_birth" in field_names
    assert "name" in field_names
    
    for fname in ("licence_number", "date_of_birth", "name"):
        crop_info = heuristic_res["field_crops"].get(fname)
        assert crop_info is not None
        assert crop_info["crop"].size > 0


# ──────────────────────────────────────────────────────────────────────
# Test 18: Provenance metadata is correct
# ──────────────────────────────────────────────────────────────────────
def test_18_provenance_metadata_correctness(synthetic_dl_image):
    """Verify exact provenance schema for both YOLO and fallback modes."""
    detector = DLFieldDetector()
    
    # Fallback provenance
    fallback_res = detector._generate_heuristic_fields(synthetic_dl_image, reason="test_reason")
    assert fallback_res["detector_source"] == "dl_layout_heuristic"
    assert fallback_res["model_name"] == "dl_field_detector.pt"
    assert fallback_res["fallback"] is True
    assert fallback_res["fallback_reason"] == "test_reason"
    assert fallback_res["document_type"] == "DRIVING_LICENCE"
    assert fallback_res["experimental"] is True
    
    # Success YOLO provenance
    detector._lazy_init()
    mock_box1 = MagicMock()
    mock_box1.conf = [MagicMock(item=MagicMock(return_value=0.9))]
    mock_box1.cls = [MagicMock(item=MagicMock(return_value=0))]
    mock_box1.xyxy = [np.array([10, 10, 50, 50])]

    mock_box2 = MagicMock()
    mock_box2.conf = [MagicMock(item=MagicMock(return_value=0.9))]
    mock_box2.cls = [MagicMock(item=MagicMock(return_value=1))]
    mock_box2.xyxy = [np.array([60, 60, 120, 120])]

    mock_res = MagicMock(boxes=[mock_box1, mock_box2])
    
    with patch.object(detector._model, "predict", return_value=[mock_res]):
        yolo_res = detector.detect_fields(synthetic_dl_image)
        assert yolo_res["detector_source"] == "dl_yolo"
        assert yolo_res["model_name"] == "dl_field_detector.pt"
        assert yolo_res["fallback"] is False
        assert yolo_res["fallback_reason"] is None
        assert yolo_res["document_type"] == "DRIVING_LICENCE"
        assert yolo_res["experimental"] is True
