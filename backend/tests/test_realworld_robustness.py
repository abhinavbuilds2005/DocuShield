"""
Real-World Distortion & False-Positive Regression Test Suite
Validates that genuine documents under real-world conditions
(WhatsApp recompression, camera capture, perspective, resizing, screenshots)
remain AUTHENTIC and do not trigger false tampering alarms.
Also verifies that deterministic tamper cases remain strictly caught.
"""

import os
import io
import pytest
import numpy as np
import cv2
from PIL import Image

from backend.fusion import DocumentScreeningPipeline
from backend.nlp.field_validator import DocumentFieldValidator
from backend.forensics.copy_move import CopyMoveDetector
from backend.forensics.ela import ErrorLevelAnalysis

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROBUST_DIR = os.path.join(DATA_DIR, "robustness")


@pytest.fixture(scope="module")
def pipeline():
    return DocumentScreeningPipeline()


def test_genuine_pan_not_flagged(pipeline):
    """Verifies that a genuine PAN card is classified as AUTHENTIC with high score."""
    path = os.path.join(DATA_DIR, "mock_pan_01_genuine.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    assert res["diagnostic_status"] in ["CLEAN", "ANOMALY_DETECTED"]


def test_genuine_aadhaar_not_flagged(pipeline):
    """Verifies that a genuine Aadhaar card is classified as AUTHENTIC."""
    path = os.path.join(DATA_DIR, "mock_aadhaar_01_genuine.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    assert res["diagnostic_status"] in ["CLEAN", "ANOMALY_DETECTED"]


def test_whatsapp_pan_not_flagged(pipeline, tmp_path):
    """
    Simulates WhatsApp image compression pipeline:
    Resizes and recompresses a genuine PAN card with aggressive JPEG quantization (Q=65).
    Verifies that global recompression does not falsely flag the card as tampered.
    """
    orig_path = os.path.join(DATA_DIR, "mock_pan_01_genuine.png")
    with Image.open(orig_path) as img:
        # WhatsApp typically resizes to max 1280px and saves at quality 60-70
        w, h = img.size
        resized = img.resize((int(w * 0.85), int(h * 0.85)), Image.Resampling.BILINEAR)
        wa_path = str(tmp_path / "whatsapp_pan.jpg")
        resized.save(wa_path, "JPEG", quality=65)

    res = pipeline.screen_document(wa_path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0


def test_camera_photo_not_flagged(pipeline):
    """Verifies that a camera photograph with lighting variation remains AUTHENTIC."""
    path = os.path.join(ROBUST_DIR, "unseen_05_genuine_perspective_lighting.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0


def test_perspective_document_not_flagged(pipeline):
    """Verifies that a document captured at an angle remains AUTHENTIC."""
    path = os.path.join(ROBUST_DIR, "unseen_05_genuine_perspective_lighting.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"


def test_jpeg_compressed_document_not_flagged(pipeline):
    """Verifies that a genuine JPEG compressed document (Q=70) remains AUTHENTIC."""
    path = os.path.join(ROBUST_DIR, "unseen_02_genuine_jpeg_compressed.jpg")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0


def test_normal_portrait_boundary_not_tampering():
    """Verifies that a normal genuine photo boundary does not generate cut-seam false alarms."""
    ela = ErrorLevelAnalysis()
    path = os.path.join(DATA_DIR, "mock_aadhaar_01_genuine.png")
    res = ela.analyze(path)
    # Genuine card portrait should not have cut seams
    assert not any("cut seam" in b.get("reason", "").lower() for b in res.get("flagged_boxes", []))


def test_qr_code_not_copy_move():
    """Verifies that genuine QR codes / logos do not trigger copy-move false positives."""
    cm = CopyMoveDetector()
    path = os.path.join(DATA_DIR, "mock_aadhaar_01_genuine.png")
    res = cm.analyze(path)
    assert res["copy_move_detected"] is False
    assert res["score"] >= 90.0


def test_repeated_logo_not_copy_move():
    """Verifies that repeated headers / emblems in genuine PAN card do not trigger copy-move."""
    cm = CopyMoveDetector()
    path = os.path.join(DATA_DIR, "mock_pan_01_genuine.png")
    res = cm.analyze(path)
    assert res["copy_move_detected"] is False


def test_low_confidence_checksum_returns_uncertain():
    """Verifies that low OCR confidence on an invalid UID returns UNCERTAIN, not deterministic failure."""
    validator = DocumentFieldValidator()
    ocr_low_conf = {
        "full_text": "Aadhaar Mera Aadhaar 5428 9182 3814",
        "tokens": [
            {"text": "54289182", "box": [10, 10, 50, 20], "confidence": 0.32},
            {"text": "3814", "box": [65, 10, 30, 20], "confidence": 0.35}
        ]
    }
    res = validator.validate(ocr_low_conf)
    field = next(f for f in res["fields"] if "Aadhaar" in f["field"])
    assert field["status"] == "UNCERTAIN"
    assert field["is_deterministic"] is False
    assert field["evidence_level"] == "WEAK"


def test_high_confidence_invalid_checksum_returns_suspicious(pipeline):
    """Verifies that deterministic Verhoeff checksum failure correctly flags the document."""
    path = os.path.join(ROBUST_DIR, "unseen_06_tampered_checksum.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["diagnostic_status"] == "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"
    assert res["authenticity_score"] <= 65.0


def test_legitimate_duplicate_qr_codes_clean():
    """Verifies that legitimate duplicate QR codes (e.g. multi-section Aadhaar letters) are CLEAN."""
    enc = cv2.QRCodeEncoder.create()
    qr_mat = enc.encode("https://uidai.gov.in/verify")
    qr_img = cv2.resize(qr_mat, (120, 120), interpolation=cv2.INTER_NEAREST)
    qr_bgr = cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)

    canvas = np.full((800, 800, 3), 255, dtype=np.uint8)
    canvas[150:270, 150:270] = qr_bgr
    canvas[500:620, 400:520] = qr_bgr

    cm = CopyMoveDetector()
    res = cm.analyze(canvas)
    assert res["copy_move_detected"] is False
    assert res["score"] >= 90.0


def test_cloned_non_qr_graphic_detected():
    """Verifies that cloned non-QR graphics are strongly detected as copy-move."""
    canvas = np.full((800, 800, 3), 255, dtype=np.uint8)
    np.random.seed(42)
    texture = np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)
    cv2.circle(texture, (60, 60), 40, (0, 0, 0), 4)
    cv2.rectangle(texture, (20, 20), (100, 100), (255, 0, 0), 3)

    canvas[150:270, 150:270] = texture
    canvas[500:620, 400:520] = texture

    cm = CopyMoveDetector()
    res = cm.analyze(canvas)
    assert res["copy_move_detected"] is True
    assert res["score"] <= 40.0
    assert len(res["flagged_boxes"]) >= 2


def test_document_with_qr_and_cloned_graphic_flags_cloned_graphic():
    """Verifies that a document with both a QR code and a cloned graphic flags the cloned graphic."""
    canvas = np.full((1000, 1000, 3), 255, dtype=np.uint8)

    # Legitimate QR code
    enc = cv2.QRCodeEncoder.create()
    qr_mat = enc.encode("https://uidai.gov.in/verify")
    qr_img = cv2.resize(qr_mat, (120, 120), interpolation=cv2.INTER_NEAREST)
    canvas[150:270, 700:820] = cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)

    # Cloned non-QR texture
    np.random.seed(42)
    texture = np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)
    cv2.circle(texture, (60, 60), 40, (0, 0, 0), 4)
    cv2.rectangle(texture, (20, 20), (100, 100), (255, 0, 0), 3)
    canvas[150:270, 150:270] = texture
    canvas[500:620, 400:520] = texture

    cm = CopyMoveDetector()
    res = cm.analyze(canvas)
    assert res["copy_move_detected"] is True
    assert res["score"] <= 40.0

