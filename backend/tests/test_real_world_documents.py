"""
Real-World Document Forensics & Robustness Test Suite
Validates the full screening pipeline across diverse real-world capture conditions:
Category A: Genuine Aadhaar (Authentic verdict, clean / minor natural artifacts)
Category B: Genuine PAN (Authentic verdict, anti-aliasing / typography tolerance)
Category C: Genuine Driving Licence (Authentic verdict, layout / header tolerance)
Category D: WhatsApp / Recompressed documents (Uniform compression awareness)
Category E: Screenshots (Low noise / clean edge tolerance)
Category F: Mobile Camera Photographs (Lighting / perspective tolerance)
Category G: Rotated Documents (0°, 90°, 180°, 270° orientation handling)
Category H: Resized Documents (Scale normalization)
Category I: Blurred Documents (Blur / focus compensation)
Category J: Clearly Tampered Documents:
  - Photo swap (portrait cut seam detection)
  - Checksum alteration (Verhoeff checksum deterministic trigger)
  - Text splice (intra-line typography mismatch)
  - Copy-move (RANSAC geometric cloning verification)
  - Font splice (sharpness / anti-aliasing discontinuity)
  - Metadata tampering (editing software signatures)

Tests exercise the actual screening pipeline and inspect structured evidence outputs.
No hardcoding based on filename is permitted.
"""

import os
import cv2
import pytest
import numpy as np
from typing import Dict, Any

from backend.fusion import DocumentScreeningPipeline
from backend.forensics.condition_analyzer import DocumentConditionAnalyzer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROBUST_DIR = os.path.join(DATA_DIR, "robustness")


@pytest.fixture(scope="module")
def pipeline():
    return DocumentScreeningPipeline()


# =========================================================================
# Category A: Genuine Aadhaar
# =========================================================================
def test_category_a_genuine_aadhaar(pipeline):
    """Verifies that authentic Aadhaar cards pass screening without false alarms."""
    img_path = os.path.join(DATA_DIR, "mock_aadhaar_01_genuine.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    assert res["risk_score"] <= 20.0
    assert res["diagnostic_status"] in ["CLEAN", "ANOMALY_DETECTED"]
    assert "why_this_verdict" in res
    assert len(res["why_this_verdict"]["critical_evidence"]) == 0
    assert len(res["why_this_verdict"]["positive_checks"]) >= 1


# =========================================================================
# Category B: Genuine PAN
# =========================================================================
def test_category_b_genuine_pan(pipeline):
    """Verifies that authentic PAN cards are not flagged by font or border heuristics."""
    img_path = os.path.join(DATA_DIR, "mock_pan_01_genuine.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    assert res["diagnostic_status"] in ["CLEAN", "ANOMALY_DETECTED"]
    assert res["why_this_verdict"]["detector_explanations"]["field_validation"]["status"] in ["CLEAN", "WEAK"]


# =========================================================================
# Category C: Genuine Driving Licence
# =========================================================================
def test_category_c_genuine_dl(pipeline):
    """Verifies that authentic DL documents pass screening cleanly."""
    img_path = os.path.join(DATA_DIR, "mock_dl_01_genuine.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    assert res["diagnostic_status"] in ["CLEAN", "ANOMALY_DETECTED"]


# =========================================================================
# Category D: WhatsApp-Compressed Documents
# =========================================================================
def test_category_d_whatsapp_compressed_document(pipeline):
    """Verifies that uniform compression does not trigger tampering flags."""
    img_path = os.path.join(ROBUST_DIR, "unseen_02_genuine_jpeg_compressed.jpg")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    # Global compression should be reported in cautions, not critical evidence
    assert any("compression" in c.lower() or "jpeg" in c.lower() for c in res["why_this_verdict"]["cautions"] + res["why_this_verdict"]["positive_checks"])
    assert len(res["why_this_verdict"]["critical_evidence"]) == 0


# =========================================================================
# Category E: Screenshots
# =========================================================================
def test_category_e_screenshot_document(pipeline):
    """Verifies that digital screenshot captures are handled without false alarms."""
    img_path = os.path.join(ROBUST_DIR, "unseen_04_genuine_screenshot.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    assert res["diagnostic_status"] in ["CLEAN", "ANOMALY_DETECTED"]


# =========================================================================
# Category F: Mobile Camera Photographs
# =========================================================================
def test_category_f_camera_photograph(pipeline):
    """Verifies that mobile camera perspective and uneven lighting are tolerated."""
    img_path = os.path.join(ROBUST_DIR, "unseen_05_genuine_perspective_lighting.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    assert res["diagnostic_status"] in ["CLEAN", "ANOMALY_DETECTED"]


# =========================================================================
# Category G: Rotated Documents (0°, 90°, 180°, 270°)
# =========================================================================
def test_category_g_rotated_documents():
    """Verifies orientation analysis and transformation helpers on rotated documents."""
    analyzer = DocumentConditionAnalyzer()

    # Create a synthetic horizontal card image (W > H)
    card_h, card_w = 300, 500
    card = np.full((card_h, card_w, 3), 245, dtype=np.uint8)
    cv2.putText(card, "GOVERNMENT OF INDIA", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(card, "AADHAAR CARD", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(card, "1234 5678 9012", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Test 0° (Natural landscape orientation)
    cond_0 = analyzer.analyze(card)
    assert cond_0["orientation"] in [0, 180]

    # Test 90° rotation (Vertical card: H > W)
    rot_90 = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    cond_90 = analyzer.analyze(rot_90)
    assert cond_90["orientation"] in [90, 270]

    # Test 270° rotation (Vertical card: H > W)
    rot_270 = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cond_270 = analyzer.analyze(rot_270)
    assert cond_270["orientation"] in [90, 270]

    # Test remap_box_to_original coordinate inversion
    orig_box = [50, 60, 200, 30]
    # For 90° clockwise: new_x = (orig_h - orig_y - orig_bh), new_y = orig_x
    rotated_box = [card_h - 60 - 30, 50, 30, 200]
    remapped = analyzer.remap_box_to_original(rotated_box, 90, card_w, card_h)
    assert remapped[0] == 50
    assert remapped[1] == 60


# =========================================================================
# Category H: Resized Documents
# =========================================================================
def test_category_h_resized_document(pipeline):
    """Verifies that downscaled or upscaled genuine documents remain Authentic."""
    img_path = os.path.join(ROBUST_DIR, "unseen_03_genuine_resized.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0


# =========================================================================
# Category I: Blurred Documents
# =========================================================================
def test_category_i_blurred_document(pipeline, tmp_path):
    """Verifies that camera defocus blur routes to human review caution without tampering flag."""
    base_path = os.path.join(DATA_DIR, "mock_aadhaar_01_genuine.png")
    img = cv2.imread(base_path)
    # Apply mild Gaussian blur mimicking camera defocus
    blurred = cv2.GaussianBlur(img, (7, 7), 2.5)
    blurred_path = str(tmp_path / "blurred_card.png")
    cv2.imwrite(blurred_path, blurred)

    res = pipeline.screen_document(blurred_path, benchmark_mode=True)
    # Should not produce a false positive tamper verdict
    assert res["verdict"] in ["AUTHENTIC", "SUSPICIOUS"]
    assert res["diagnostic_status"] != "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"
    # Condition assessment should record blur
    assert res["condition_assessment"]["is_blurry"] is True


# =========================================================================
# Category J: Clearly Tampered Documents
# =========================================================================
def test_category_j_tampered_photoswap(pipeline):
    """Verifies detection of portrait photo swap with seam boundary."""
    img_path = os.path.join(DATA_DIR, "mock_aadhaar_01_tampered_photoswap.jpg")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["diagnostic_status"] == "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"
    assert any("seam" in s.lower() or "portrait" in s.lower() for s in res["critical_triggers"])


def test_category_j_tampered_checksum(pipeline):
    """Verifies deterministic failure on mathematically corrupted UID."""
    img_path = os.path.join(DATA_DIR, "mock_aadhaar_02_tampered_checksum.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert any("checksum" in s.lower() for s in res["critical_triggers"])


def test_category_j_tampered_textsplice(pipeline):
    """Verifies detection of spliced date / field text."""
    img_path = os.path.join(DATA_DIR, "mock_pan_02_tampered_datesplice.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]


def test_category_j_tampered_copymove(pipeline):
    """Verifies RANSAC detection of cloned image regions."""
    img_path = os.path.join(DATA_DIR, "mock_aadhaar_06_tampered_copymove.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["signals"]["copy_move"]["detected"] is True


def test_category_j_tampered_fontsplice(pipeline):
    """Verifies detection of typography / sharpness mismatch."""
    img_path = os.path.join(DATA_DIR, "mock_aadhaar_04_tampered_fontsplice.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]


def test_category_j_tampered_metadata(pipeline):
    """Verifies deterministic detection of editing software signature."""
    img_path = os.path.join(DATA_DIR, "mock_dl_02_tampered_photoshop_meta.png")
    res = pipeline.screen_document(img_path, benchmark_mode=True)

    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["signals"]["metadata_forensics"]["detected_software"] is not None
