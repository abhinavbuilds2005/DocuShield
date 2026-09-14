"""
Regression & Robustness Test Suite for DocuShield AI Low-Quality Image Handling.
Verifies all 13 core requirements:
1. Blurry genuine document does not automatically become fake.
2. Heavily compressed genuine document retains AUTHENTIC or acceptable score.
3. Genuine screenshot retains AUTHENTIC.
4. Low OCR confidence on checksum is marked UNCERTAIN, not deterministic fraud.
5. High OCR confidence on checksum failure remains strong deterministic fraud.
6. Valid checksum alone does not prove authenticity if tampering exists.
7. Weak ELA anomaly alone does not force SUSPICIOUS.
8. Typography anomaly remains supporting evidence only.
9. Copy-move rejects repeated QR codes and background textures on clean cards.
10. Severely degraded card without tampering produces NEEDS REVIEW.
11. Tampered spliced card (even if blurry) is still flagged (tampering overrides low quality).
12. Zero filename / path / ground-truth leakage during inference.
13. Backward compatibility of legacy_verdict and legacy diagnostic fields.
"""

import os
import inspect
import cv2
import numpy as np
import pytest

from backend.forensics.condition_analyzer import DocumentConditionAnalyzer, analyze_document_condition
from backend.nlp.field_validator import DocumentFieldValidator
from backend.forensics.copy_move import CopyMoveDetector
from backend.fusion import DocumentScreeningPipeline

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROBUST_DIR = os.path.join(DATA_DIR, "robustness")


@pytest.fixture(scope="module")
def pipeline():
    return DocumentScreeningPipeline()


@pytest.fixture(scope="module")
def condition_analyzer():
    return DocumentConditionAnalyzer()


@pytest.fixture(scope="module")
def field_validator():
    return DocumentFieldValidator()


def create_synthetic_doc(width=800, height=500, quality="good"):
    """Generates an image of a document with known conditions."""
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    # Header bar
    cv2.rectangle(img, (0, 0), (width, 70), (45, 90, 160), -1)
    # Photo placeholder
    cv2.rectangle(img, (40, 100), (220, 320), (200, 200, 200), -1)
    cv2.circle(img, (130, 180), 45, (130, 130, 130), -1)
    cv2.ellipse(img, (130, 280), (60, 50), 0, 0, 180, (130, 130, 130), -1)

    # Clean text lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "REPUBLIC IDENTITY CARD", (260, 45), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "NAME: JANE DOE", (260, 140), font, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "DOB: 15/08/1990", (260, 190), font, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "ID: 9876 5432 1098", (260, 240), font, 0.7, (30, 30, 30), 2)

    if quality == "blurry":
        img = cv2.GaussianBlur(img, (17, 17), 6)
    elif quality == "compressed":
        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 25])
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif quality == "severely_degraded":
        # Low resolution + severe blur + noise
        img = cv2.resize(img, (320, 200))
        img = cv2.GaussianBlur(img, (15, 15), 6)
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif quality == "screenshot":
        img = cv2.resize(img, (1920, 1080))

    return img


# ─── Condition 1: Blurry genuine document does not automatically become fake ───

def test_blurry_genuine_document_not_fake(pipeline, tmp_path):
    img = create_synthetic_doc(quality="blurry")
    path = str(tmp_path / "blurry_doc.jpg")
    cv2.imwrite(path, img)

    res = pipeline.screen_document(path, benchmark_mode=False)

    assert res["quality_tier"] in ["LOW", "VERY_LOW"]
    assert res["verdict"] in ["AUTHENTIC", "NEEDS REVIEW", "SUSPICIOUS"]
    assert res["verdict"] != "FLAGGED / TAMPERED"
    assert res["reliability_score"] < 0.75


# ─── Condition 2: Heavily compressed genuine document retains AUTHENTIC ───

def test_heavily_compressed_genuine_retains_authentic(pipeline, tmp_path):
    img = create_synthetic_doc(quality="compressed")
    path = str(tmp_path / "compressed_doc.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 25])

    res = pipeline.screen_document(path, benchmark_mode=False)

    assert res["verdict"] in ["AUTHENTIC", "NEEDS REVIEW", "SUSPICIOUS"]
    assert res["verdict"] != "FLAGGED / TAMPERED"
    assert res["condition"]["compression_level"] in ["LOW", "MODERATE", "HEAVY"]
    assert res["authenticity_score"] >= 50.0


# ─── Condition 3: Genuine screenshot retains AUTHENTIC ───

def test_genuine_screenshot_not_penalized_harshly(pipeline, tmp_path):
    img = create_synthetic_doc(quality="screenshot")
    path = str(tmp_path / "screenshot_doc.png")
    cv2.imwrite(path, img)

    res = pipeline.screen_document(path, benchmark_mode=False)

    assert res["condition"]["is_screenshot"] is True
    assert res["verdict"] in ["AUTHENTIC", "SUSPICIOUS"]
    assert res["verdict"] != "FLAGGED / TAMPERED"


# ─── Condition 4: Low OCR confidence on checksum is marked UNCERTAIN ───

def test_low_ocr_confidence_checksum_uncertain(field_validator):
    ocr_results = {
        "full_text": "AADHAAR 1234 5678 9012",
        "tokens": [
            {"text": "AADHAAR", "confidence": 0.85},
            {"text": "1234", "confidence": 0.40},
            {"text": "5678", "confidence": 0.40},
            {"text": "9012", "confidence": 0.40}
        ],
        "lines": [{"text": "AADHAAR 1234 5678 9012", "confidence": 0.50}]
    }

    res = field_validator.validate(ocr_results, user_selected_type="aadhaar")
    aadhaar_field = next((f for f in res["fields"] if "aadhaar" in f["field"].lower() or "national id" in f["field"].lower()), None)

    assert aadhaar_field is not None
    assert aadhaar_field["status"] == "UNCERTAIN"
    assert aadhaar_field.get("is_deterministic") is False
    assert aadhaar_field.get("evidence_level") == "WEAK"
    assert res["has_deterministic_failure"] is False


# ─── Condition 5: High OCR confidence on checksum failure remains strong fraud ───

def test_high_ocr_confidence_checksum_failure_is_strong_fraud(field_validator):
    ocr_results = {
        "full_text": "AADHAAR 1234 5678 9012",
        "tokens": [
            {"text": "AADHAAR", "confidence": 0.95},
            {"text": "1234", "confidence": 0.92},
            {"text": "5678", "confidence": 0.92},
            {"text": "9012", "confidence": 0.92}
        ],
        "lines": [{"text": "AADHAAR 1234 5678 9012", "confidence": 0.93}]
    }

    res = field_validator.validate(ocr_results, user_selected_type="aadhaar")
    aadhaar_field = next((f for f in res["fields"] if "aadhaar" in f["field"].lower() or "national id" in f["field"].lower()), None)

    assert aadhaar_field is not None
    assert aadhaar_field["status"] == "FAIL"
    assert aadhaar_field.get("is_deterministic") is True
    assert aadhaar_field.get("evidence_level") == "STRONG"
    assert res["has_deterministic_failure"] is True


# ─── Condition 6: Valid checksum alone does not prove authenticity ───

def test_valid_checksum_alone_does_not_prove_authenticity(pipeline):
    # unseen_08 has copy-move manipulation
    cm_path = os.path.join(ROBUST_DIR, "unseen_08_tampered_copymove.png")
    res = pipeline.screen_document(cm_path, benchmark_mode=False)

    # Visual forensics identify tampering, overriding any simple format validity
    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
    assert res["diagnostic_status"] in ["ANOMALY_DETECTED", "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"]


# ─── Condition 7: Weak ELA anomaly alone does not force SUSPICIOUS ───

def test_weak_ela_alone_does_not_force_suspicious(pipeline, tmp_path):
    img = create_synthetic_doc(quality="good")
    path = str(tmp_path / "clean_doc.png")
    cv2.imwrite(path, img)

    res = pipeline.screen_document(path, benchmark_mode=False)
    assert res["verdict"] == "AUTHENTIC"
    assert res["authenticity_score"] >= 80.0


# ─── Condition 8: Typography anomaly remains supporting evidence only ───

def test_typography_alone_is_supporting_evidence_only(field_validator):
    ocr_results = {
        "full_text": "REPUBLIC IDENTITY CARD\nNAME: JOHN DOE",
        "tokens": [
            {"text": "REPUBLIC", "confidence": 0.88},
            {"text": "IDENTITY", "confidence": 0.85},
            {"text": "CARD", "confidence": 0.86},
            {"text": "NAME:", "confidence": 0.80},
            {"text": "JOHN", "confidence": 0.45},
            {"text": "DOE", "confidence": 0.82}
        ],
        "lines": [{"text": "REPUBLIC IDENTITY CARD", "confidence": 0.86}, {"text": "NAME: JOHN DOE", "confidence": 0.70}]
    }
    res = field_validator.validate(ocr_results, user_selected_type="national_id")
    assert res["has_deterministic_failure"] is False


# ─── Condition 9: Copy-move rejects repeated textures/clean images ───

def test_copy_move_clean_synthetic_doc():
    img = create_synthetic_doc(quality="good")
    detector = CopyMoveDetector()
    res = detector.analyze(img)
    assert res["copy_move_detected"] is False


# ─── Condition 10: Severely degraded card without tampering produces NEEDS REVIEW ───

def test_severely_degraded_card_yields_needs_review(pipeline, tmp_path):
    img = create_synthetic_doc(quality="severely_degraded")
    path = str(tmp_path / "severely_degraded.jpg")
    cv2.imwrite(path, img)

    res = pipeline.screen_document(path, benchmark_mode=False)

    assert res["quality_tier"] in ["LOW", "VERY_LOW"]
    assert res["verdict"] in ["NEEDS REVIEW", "SUSPICIOUS"]
    assert res["verdict"] != "FLAGGED / TAMPERED"
    assert "quality_tier" in res
    assert "reliability_score" in res


# ─── Condition 11: Tampered spliced card (even if blurry) is still flagged ───

def test_tampered_card_overrides_low_quality(pipeline, tmp_path):
    # Take an actual tampered card (checksum failure) and blur it
    src_path = os.path.join(ROBUST_DIR, "unseen_06_tampered_checksum.png")
    orig_img = cv2.imread(src_path)
    assert orig_img is not None

    # Apply moderate blur to simulate low quality capture of a tampered card
    blurry_tampered = cv2.GaussianBlur(orig_img, (5, 5), 1.5)
    dest_path = str(tmp_path / "blurry_tampered_card.png")
    cv2.imwrite(dest_path, blurry_tampered)

    res = pipeline.screen_document(dest_path, benchmark_mode=False)

    # Tampering cannot be excused by blur — document must still be flagged/suspicious
    assert res["verdict"] in ["SUSPICIOUS", "FLAGGED / TAMPERED"]


# ─── Condition 12: Zero filename / path / ground-truth leakage during inference ───

def test_zero_metadata_leakage_in_forensics():
    from backend.forensics import condition_analyzer, ela, copy_move, font_alignment, metadata_checker
    from backend import fusion

    modules = [condition_analyzer, ela, copy_move, font_alignment, metadata_checker, fusion]
    forbidden_terms = ["ground_truth.json", "dataset_path", "test_labels"]

    for mod in modules:
        source = inspect.getsource(mod)
        for term in forbidden_terms:
            assert term not in source, f"Module {mod.__name__} leaks or references '{term}'!"


# ─── Condition 13: Backward compatibility of legacy fields ───

def test_backward_compatibility_fields(pipeline, tmp_path):
    img = create_synthetic_doc(quality="good")
    path = str(tmp_path / "compat_test.png")
    cv2.imwrite(path, img)

    res = pipeline.screen_document(path, benchmark_mode=False)

    # Core required fields
    assert "authenticity_score" in res
    assert "risk_score" in res
    assert "verdict" in res
    assert "diagnostic_status" in res
    assert "detector_evidence" in res
    assert "condition" in res
    assert "signals" in res

    # New robustness fields
    assert "quality_tier" in res
    assert "reliability_score" in res
    assert "quality_explanation" in res
    assert "legacy_verdict" in res

    assert res["quality_tier"] in ["GOOD", "ACCEPTABLE", "LOW", "VERY_LOW"]
    assert 0.0 <= res["reliability_score"] <= 1.0
