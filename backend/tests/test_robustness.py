"""
Automated Robustness and Generalization Test Suite
Tests 11 realistic unseen document variations:
1. Genuine original document
2. Genuine JPEG-compressed document (Q=70)
3. Genuine resized document
4. Genuine screenshot / simulated screen photo
5. Genuine perspective / lighting variation
6. Fake checksum alteration
7. Fake text splice
8. Fake copy-move
9. Fake font splice
10. Fake category / text replacement
11. Fake realistic photo-swap (single normal JPEG re-save, no artificial double compression)

Also tests complete presence of structured forensic_decision_debug report.
Uses strictly REAL OCR MODE (benchmark_mode=False). No sidecars.
"""

import os
import pytest
from backend.fusion import DocumentScreeningPipeline

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROBUST_DIR = os.path.join(DATA_DIR, "robustness")


@pytest.fixture(scope="module")
def pipeline():
    return DocumentScreeningPipeline()


def test_robustness_genuine_original(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_01_genuine_original.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    assert res["diagnostic_status"] in ["CLEAN", "ANOMALY_DETECTED"]


def test_robustness_genuine_jpeg_compressed(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_02_genuine_jpeg_compressed.jpg")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0
    assert res["diagnostic_status"] in ["CLEAN", "ANOMALY_DETECTED"]


def test_robustness_genuine_resized(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_03_genuine_resized.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0


def test_robustness_genuine_screenshot(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_04_genuine_screenshot.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0


def test_robustness_genuine_perspective_lighting(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_05_genuine_perspective_lighting.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0


def test_robustness_tampered_checksum(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_06_tampered_checksum.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["diagnostic_status"] == "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"


def test_robustness_tampered_textsplice(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_07_tampered_textsplice.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["diagnostic_status"] in ["ANOMALY_DETECTED", "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"]


def test_robustness_tampered_copymove(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_08_tampered_copymove.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["diagnostic_status"] in ["ANOMALY_DETECTED", "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"]


def test_robustness_tampered_fontsplice(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_09_tampered_fontsplice.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["diagnostic_status"] in ["ANOMALY_DETECTED", "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"]


def test_robustness_tampered_categorysplice(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_10_tampered_categorysplice.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["diagnostic_status"] in ["ANOMALY_DETECTED", "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"]


def test_robustness_tampered_realistic_photoswap(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_11_tampered_realistic_photoswap.jpg")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["diagnostic_status"] in ["ANOMALY_DETECTED", "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"]


def test_forensic_decision_debug_structure(pipeline):
    path = os.path.join(ROBUST_DIR, "unseen_01_genuine_original.png")
    res = pipeline.screen_document(path, benchmark_mode=False)
    assert "forensic_decision_debug" in res
    debug = res["forensic_decision_debug"]

    # Verify all 14 required forensic debug fields are present
    required_fields = [
        "final_verdict", "final_risk_score", "confidence",
        "diagnostic_status", "ocr_status_and_confidence",
        "checksum_result", "regex_field_validation_result",
        "ela_result_and_score", "copy_move_result_and_score",
        "typography_rendering_result_and_score", "metadata_result",
        "detector_contributions", "triggered_signals", "decision_threshold",
        "weak_or_uncertain_signals"
    ]
    for field in required_fields:
        assert field in debug, f"Missing debug field: {field}"

    # Section 9: verify diagnostic_status strictly uses only the intended categories
    assert debug["diagnostic_status"] in [
        "CLEAN",
        "ANOMALY_DETECTED",
        "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"
    ]
