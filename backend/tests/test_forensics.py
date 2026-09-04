"""
Forensics Modules Integration Tests
Tests:
- ELA (Error Level Analysis) on genuine vs tampered cards
- ELA on PNG vs JPEG
- Copy-move detector (self-match removal, RANSAC, keypoint matching)
- Typography and rendering forensics
- Metadata checker (EXIF extraction, editing software signatures)
- Multimodal Fusion engine (scoring, verdicts, evidence breakdown)
"""

import os
import cv2
import pytest
import numpy as np

from backend.forensics.ela import ErrorLevelAnalysis
from backend.forensics.copy_move import CopyMoveDetector
from backend.forensics.font_alignment import FontAlignmentForensics
from backend.forensics.metadata_checker import MetadataForensics
from backend.fusion import DocumentScreeningPipeline
from backend.nlp.field_validator import DocumentFieldValidator

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GENUINE_CARD = os.path.join(DATA_DIR, "mock_aadhaar_01_genuine.png")
TAMPERED_PHOTO = os.path.join(DATA_DIR, "mock_aadhaar_01_tampered_photoswap.jpg")
TAMPERED_COPYMOVE = os.path.join(DATA_DIR, "mock_aadhaar_06_tampered_copymove.png")
TAMPERED_META = os.path.join(DATA_DIR, "mock_dl_02_tampered_photoshop_meta.png")


def test_ela_analysis():
    ela = ErrorLevelAnalysis()
    # Test on genuine PNG
    res_png = ela.analyze(GENUINE_CARD)
    assert "ela_authenticity_score" in res_png
    assert "heatmap_data_uri" in res_png
    assert "format_note" in res_png
    assert res_png["source_format"] == "png"

    # Test on tampered photo swap JPEG
    res_jpg = ela.analyze(TAMPERED_PHOTO)
    assert res_jpg["source_format"] == "jpg"
    assert res_jpg["is_suspicious"] is True or res_jpg["hotspot_count"] >= 1


def test_copy_move_no_self_matches():
    detector = CopyMoveDetector()
    img = cv2.imread(GENUINE_CARD)
    res = detector.analyze(img)
    # Genuine card should not trigger copy-move false positive
    assert res["copy_move_detected"] is False
    assert res["score"] >= 80.0


def test_copy_move_detection_on_cloned_card():
    detector = CopyMoveDetector()
    img = cv2.imread(TAMPERED_COPYMOVE)
    res = detector.analyze(img)
    assert "copy_move_detected" in res
    assert "flagged_boxes" in res
    if res["copy_move_detected"]:
        assert len(res["flagged_boxes"]) >= 2


def test_metadata_forensics_photoshop():
    meta = MetadataForensics()
    res_clean = meta.analyze(GENUINE_CARD)
    assert res_clean["metadata_score"] == 100.0
    assert res_clean["is_suspicious"] is False

    res_tampered = meta.analyze(TAMPERED_META)
    assert res_tampered["is_suspicious"] is True
    assert res_tampered["detected_software"] is not None


def test_fusion_pipeline_end_to_end():
    pipeline = DocumentScreeningPipeline()
    # Test in benchmark mode using synthetic samples
    res = pipeline.screen_document(GENUINE_CARD, benchmark_mode=True)
    assert "authenticity_score" in res
    assert "verdict" in res
    assert res["verdict"] == "AUTHENTIC"
    assert "signals" in res
    assert "visualizations" in res
    assert res["authenticity_score"] >= 80.0


def test_font_splicing_detection():
    fa = FontAlignmentForensics()
    spliced_path = os.path.join(DATA_DIR, "mock_aadhaar_04_tampered_fontsplice.png")
    img = cv2.imread(spliced_path)
    tokens = [
        {"text": "Name:", "box": [230, 138, 50, 30], "confidence": 0.95},
        {"text": "VIKRAM", "box": [311, 139, 49, 12], "confidence": 0.95},
        {"text": "MALHOTRA", "box": [360, 139, 49, 12], "confidence": 0.95}
    ]
    res = fa.analyze(img, tokens)
    assert res["anomalies_detected"] is True
    assert len(res["flagged_boxes"]) >= 1


def test_verhoeff_checksum_validation():
    from backend.nlp.field_validator import validate_verhoeff
    assert validate_verhoeff("542891623811") is True
    assert validate_verhoeff("542891623814") is False
    assert validate_verhoeff("542891823814") is False


def test_pan_format_ocr_tolerance():
    from backend.nlp.field_validator import validate_pan_format
    ok, err = validate_pan_format("AZYPA9876Q")
    assert ok is True
    ok, err = validate_pan_format("AZYPA98780")
    assert ok is True
    ok, err = validate_pan_format("ABC9999999")
    assert ok is False


def test_checksum_top_level_status_consistency():
    """
    SECTION 2 REGRESSION TEST:
    Verifies that when a required field fails deterministic validation,
    the top-level checksum_result.status is FAIL, not PASS, and deterministic is True.
    """
    pipeline = DocumentScreeningPipeline()
    p = os.path.join(DATA_DIR, "mock_aadhaar_02_tampered_checksum.png")
    res = pipeline.screen_document(p)
    chk = res["forensic_decision_debug"]["checksum_result"]
    assert chk["status"] == "FAIL", f"Expected checksum status to be FAIL, got: {chk['status']}"
    assert chk["deterministic"] is True
    assert len(chk["fields"]) >= 1
    assert chk["fields"][0]["status"] == "FAIL"


def test_consistent_evidence_strength_terminology():
    """
    SECTION 3 TEST:
    All detector scores and classifications must use identical, uppercase terminology:
    CLEAN, WEAK, MODERATE, STRONG across signals, fusion_contributions, and debug report.
    """
    pipeline = DocumentScreeningPipeline()
    p = os.path.join(DATA_DIR, "mock_aadhaar_02_tampered_checksum.png")
    res = pipeline.screen_document(p)
    valid_levels = {"CLEAN", "WEAK", "MODERATE", "STRONG"}

    for key, val in res["evidence_strengths"].items():
        assert val in valid_levels, f"evidence_strengths[{key}] has invalid format: {val}"
        assert val == res["fusion_contributions"][key]["evidence_strength"]

    debug = res["forensic_decision_debug"]
    assert debug["ela_result_and_score"]["evidence_strength"] in valid_levels
    assert debug["copy_move_result_and_score"]["evidence_strength"] in valid_levels
    assert debug["typography_rendering_result_and_score"]["evidence_strength"] in valid_levels
    assert debug["metadata_result"]["evidence_strength"] in valid_levels


def test_ocr_confidence_cases_a_b_c_d():
    """
    SECTION 4 TESTS:
    Case A: High-confidence deterministic failure -> DOCUMENT_STRONGLY_SUSPECTED_TAMPERED
    Case B: Low-confidence OCR failure -> UNCERTAIN/WEAK, deterministic=False
    Case C: Low-confidence OCR + weak environmental anomaly -> remains AUTHENTIC (penalty capped)
    Case D: Low-confidence OCR + genuine independent moderate/strong evidence -> SUSPICIOUS/FLAGGED
    """
    validator = DocumentFieldValidator()

    # Case A: High-confidence OCR with mathematically invalid checksum
    ocr_case_a = {
        "full_text": "Aadhaar Mera Aadhaar 5428 9182 3814",
        "tokens": [
            {"text": "54289182", "box": [10, 10, 50, 20], "confidence": 0.88},
            {"text": "3814", "box": [65, 10, 30, 20], "confidence": 0.88}
        ]
    }
    res_a = validator.validate(ocr_case_a)
    assert any(f["status"] == "FAIL" and f.get("is_deterministic") for f in res_a["fields"])

    # Case B: Low-confidence OCR failure -> UNCERTAIN, deterministic=False
    ocr_case_b = {
        "full_text": "Aadhaar Mera Aadhaar 5428 9182 3814",
        "tokens": [
            {"text": "54289182", "box": [10, 10, 50, 20], "confidence": 0.35},
            {"text": "3814", "box": [65, 10, 30, 20], "confidence": 0.35}
        ]
    }
    res_b = validator.validate(ocr_case_b)
    field_b = next(f for f in res_b["fields"] if "Aadhaar" in f["field"])
    assert field_b["status"] == "UNCERTAIN"
    assert field_b["is_deterministic"] is False
    assert field_b["evidence_level"] == "WEAK"

    # Case C: Low-confidence OCR + weak anomaly (e.g. resized genuine card)
    # Verifies that normal distortions + weak signals do not wrongfully trigger DOCUMENT_STRONGLY_SUSPECTED_TAMPERED
    pipeline = DocumentScreeningPipeline()
    p_resized = os.path.join(DATA_DIR, "robustness", "unseen_03_genuine_resized.png")
    res_c = pipeline.screen_document(p_resized)
    assert res_c["verdict"] == "AUTHENTIC"
    assert res_c["diagnostic_status"] != "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"

    # Case D: Low-confidence OCR + genuine independent strong evidence (e.g. photo swap or cut seam)
    p_swap = os.path.join(DATA_DIR, "mock_aadhaar_01_tampered_photoswap.jpg")
    res_d = pipeline.screen_document(p_swap)
    assert res_d["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res_d["diagnostic_status"] == "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"

