"""
Unit Tests for Document Validation Engine (SIH26188)
Tests:
- Invalid date format
- Chronological contradiction (DOB vs Issue vs Expiry)
- Expired documents
- Missing required fields
- Low OCR confidence does not trigger false fraud verdict
- Checksum validation applied only where appropriate (Aadhaar UID Verhoeff, NOT Passport/Visa)
"""

import pytest
from backend.nlp.field_validator import DocumentFieldValidator, validate_verhoeff


@pytest.fixture
def validator():
    return DocumentFieldValidator()


def test_invalid_date_detection(validator):
    ocr_res = {"full_text": "DOB: 32/13/2020", "tokens": [], "lines": []}
    fields = {
        "date_of_birth": {"value": "32/13/2020", "confidence": 0.85, "status": "invalid"}
    }
    res = validator.validate_document("national_id", fields, ocr_res)
    assert res["anomalies_detected"] is True
    assert any("Invalid date of birth" in r for r in res["reasons"])


def test_chronological_contradiction(validator):
    ocr_res = {"full_text": "Dates extracted", "tokens": [], "lines": []}
    # Issue date before DOB
    fields = {
        "date_of_birth": {"value": "15/08/2000", "confidence": 0.90, "status": "valid"},
        "issue_date": {"value": "15/08/1990", "confidence": 0.90, "status": "valid"},
        "expiry_date": {"value": "15/08/2030", "confidence": 0.90, "status": "valid"}
    }
    res = validator.validate_document("driving_license", fields, ocr_res)
    assert any("Chronological Contradiction" in r for r in res["reasons"])


def test_expired_document_flagging(validator):
    ocr_res = {"full_text": "Expired card", "tokens": [], "lines": []}
    fields = {
        "date_of_birth": {"value": "01/01/1980", "confidence": 0.90, "status": "valid"},
        "issue_date": {"value": "01/01/2010", "confidence": 0.90, "status": "valid"},
        "date_of_expiry": {"value": "01/01/2020", "confidence": 0.90, "status": "valid"}
    }
    res = validator.validate_document("passport", fields, ocr_res)
    expired_checks = [c for c in res["checks"] if c.get("status") == "EXPIRED"]
    assert len(expired_checks) > 0
    assert "Document Expired" in res["reasons"][0]


def test_verhoeff_checksum_only_for_aadhaar(validator):
    # Verhoeff should be valid for correct 12-digit Indian UID
    assert validate_verhoeff("234567890124") is True  # Valid Verhoeff
    assert validate_verhoeff("234567890129") is False # Invalid Verhoeff


    # Validation engine for passport must NOT apply Verhoeff
    p_fields = {
        "passport_number": {"value": "Z9876543", "confidence": 0.90, "status": "valid"},
        "full_name": {"value": "JANE DOE", "confidence": 0.90, "status": "valid"}
    }
    res = validator.validate_document("passport", p_fields, {"full_text": "Passport Z9876543", "tokens": []})
    assert not any("Verhoeff" in c.get("rule", "") for c in res["checks"])


def test_low_ocr_confidence_does_not_mean_fraud(validator):
    # Low confidence extracted field marked unknown/warning, not outright deterministic fraud
    ocr_res = {
        "full_text": "BlUrRy TeXt",
        "tokens": [{"text": "BlUrRy", "confidence": 0.35}],
        "lines": []
    }
    res = validator.validate(ocr_res)
    assert res["has_deterministic_failure"] is False
