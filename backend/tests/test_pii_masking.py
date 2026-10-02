"""
Unit and Integration Tests for PII Masking and Output Sanitization

Tests:
1. Aadhaar number masking (spaced, unspaced, dashed, idempotent)
2. PAN number masking (standard format, idempotent)
3. Driving Licence number masking (multiple Indian formats, idempotent)
4. Personal name masking (single/multi-token, idempotent)
5. Date of birth masking (DD/MM/YYYY and ISO formats, idempotent)
6. Freeform text pattern scrubbing (OCR text, QR text)
7. Full response sanitization (nested schema_fields, structural_checks, signals)
8. Preservation of non-PII metadata, scores, verdicts, and boolean flags
9. API integration verification: /api/screen returns masked PII only
"""

import pytest
from fastapi.testclient import TestClient

from backend.nlp.pii_masking import (
    mask_aadhaar,
    mask_pan,
    mask_driving_license,
    mask_name,
    mask_dob,
    mask_passport,
    mask_pii_text,
    sanitize_screening_response,
)
from backend.app import app


client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Aadhaar Masking Tests
# ---------------------------------------------------------------------------
def test_mask_aadhaar_continuous_digits():
    raw = "542891623811"
    masked = mask_aadhaar(raw)
    assert masked == "XXXX XXXX 3811"
    assert "5428" not in masked
    assert "9162" not in masked


def test_mask_aadhaar_spaced_and_dashed():
    assert mask_aadhaar("5428 9162 3811") == "XXXX XXXX 3811"
    assert mask_aadhaar("5428-9162-3811") == "XXXX XXXX 3811"


def test_mask_aadhaar_idempotence():
    already_masked = "XXXX XXXX 3811"
    assert mask_aadhaar(already_masked) == already_masked
    # Double pass
    assert mask_aadhaar(mask_aadhaar("542891623811")) == "XXXX XXXX 3811"


# ---------------------------------------------------------------------------
# 2. PAN Masking Tests
# ---------------------------------------------------------------------------
def test_mask_pan_standard():
    raw = "ABCDE1234F"
    masked = mask_pan(raw)
    assert masked == "ABCDE****F"
    assert "1234" not in masked


def test_mask_pan_idempotence():
    already_masked = "ABCDE****F"
    assert mask_pan(already_masked) == already_masked
    assert mask_pan(mask_pan("ABCDE1234F")) == "ABCDE****F"


# ---------------------------------------------------------------------------
# 3. Driving Licence Masking Tests
# ---------------------------------------------------------------------------
def test_mask_driving_license_standard():
    # 2-letter state + 2-digit RTO + 11-digit serial
    raw = "BR0120190012345"
    masked = mask_driving_license(raw)
    assert masked.startswith("BR01")
    assert "20190012345" not in masked
    assert "*" in masked


def test_mask_driving_license_with_dashes():
    raw = "DL-1420110012345"
    masked = mask_driving_license(raw)
    assert masked.startswith("DL-14")
    assert "20110012345" not in masked
    assert "*" in masked


def test_mask_driving_license_idempotence():
    masked = mask_driving_license("BR0120190012345")
    assert mask_driving_license(masked) == masked


# ---------------------------------------------------------------------------
# 4. Personal Name Masking Tests
# ---------------------------------------------------------------------------
def test_mask_name():
    assert mask_name("Aarav Sharma") == "A**** S*****"
    assert mask_name("John Doe") == "J*** D**"
    assert mask_name("Not Detected") == "Not Detected"
    assert mask_name("—") == "—"


def test_mask_name_idempotence():
    masked = mask_name("Aarav Sharma")
    assert mask_name(masked) == masked


# ---------------------------------------------------------------------------
# 5. Date of Birth Masking Tests
# ---------------------------------------------------------------------------
def test_mask_dob():
    assert mask_dob("15/08/1990") == "**/**/1990"
    assert mask_dob("15-08-1990") == "**-**-1990"
    assert mask_dob("1990-08-15") == "1990-**-**"
    assert mask_dob("Not Detected") == "Not Detected"


def test_mask_dob_idempotence():
    masked = mask_dob("15/08/1990")
    assert mask_dob(masked) == masked


# ---------------------------------------------------------------------------
# 6. Freeform Text Masking Tests
# ---------------------------------------------------------------------------
def test_mask_pii_text_mixed_ocr():
    ocr_sample = (
        "GOVERNMENT OF INDIA\n"
        "Name: Aarav Sharma\n"
        "Aadhaar Number: 5428 9162 3811\n"
        "PAN Number: ABCDE1234F\n"
        "DL Number: BR0120190012345\n"
    )
    scrubbed = mask_pii_text(ocr_sample)
    assert "5428 9162 3811" not in scrubbed
    assert "XXXX XXXX 3811" in scrubbed
    assert "ABCDE1234F" not in scrubbed
    assert "ABCDE****F" in scrubbed
    assert "20190012345" not in scrubbed


# ---------------------------------------------------------------------------
# 7. Nested Response Sanitizer Tests
# ---------------------------------------------------------------------------
def test_sanitize_screening_response():
    mock_response = {
        "status": "STRUCTURALLY_VALID_UNVERIFIED",
        "verdict": "AUTHENTIC",
        "risk_score": 5.0,
        "authenticity_score": 95.0,
        "model_name": "yolov8n_aadhaar_detector",
        "schema_fields": {
            "aadhaar_number": {"value": "5428 9162 3811", "status": "valid", "confidence": 0.95},
            "name": {"value": "Aarav Sharma", "status": "valid", "confidence": 0.88},
            "dob": {"value": "15/08/1990", "status": "valid", "confidence": 0.82},
            "gender": {"value": "MALE", "status": "valid", "confidence": 0.99}
        },
        "structural_checks": {
            "format_valid": True,
            "field_details": [
                {"field": "Aadhaar Number Checksum", "status": "PASS", "value": "5428 9162 3811"},
                {"field": "Holder Name", "status": "PASS", "value": "Aarav Sharma"}
            ]
        },
        "signals": {
            "nlp_validation": {
                "score": 95.0,
                "extracted_full_text": "Government of India Aarav Sharma 5428 9162 3811 DOB 15/08/1990",
                "field_checks": [
                    {"field": "Aadhaar Checksum", "value": "5428 9162 3811", "details": "Verhoeff check passed for 5428 9162 3811"}
                ],
                "reasons": []
            }
        },
        "qr_analysis": {
            "status": "VERIFIED_MATCH",
            "extracted_number": "542891623811",
            "raw_text": "Aadhaar: 542891623811 Name: Aarav Sharma"
        },
        "qr_extracted_number": "542891623811"
    }

    sanitized = sanitize_screening_response(mock_response)

    # 1. Verify schema fields masked
    assert sanitized["schema_fields"]["aadhaar_number"]["value"] == "XXXX XXXX 3811"
    assert sanitized["schema_fields"]["name"]["value"] == "A**** S*****"
    assert sanitized["schema_fields"]["dob"]["value"] == "**/**/1990"
    assert sanitized["schema_fields"]["gender"]["value"] == "MALE"  # non-PII preserved

    # 2. Verify structural checks masked
    assert sanitized["structural_checks"]["field_details"][0]["value"] == "XXXX XXXX 3811"
    assert sanitized["structural_checks"]["field_details"][1]["value"] == "A**** S*****"

    # 3. Verify signals masked
    assert "5428 9162 3811" not in sanitized["signals"]["nlp_validation"]["extracted_full_text"]
    assert "XXXX XXXX 3811" in sanitized["signals"]["nlp_validation"]["extracted_full_text"]
    assert sanitized["signals"]["nlp_validation"]["field_checks"][0]["value"] == "XXXX XXXX 3811"
    assert "5428 9162 3811" not in sanitized["signals"]["nlp_validation"]["field_checks"][0]["details"]

    # 4. Verify QR masked
    assert sanitized["qr_analysis"]["extracted_number"] == "XXXX XXXX 3811"
    assert "542891623811" not in sanitized["qr_analysis"]["raw_text"]
    assert sanitized["qr_extracted_number"] == "XXXX XXXX 3811"

    # 5. Verify technical metadata and scores preserved
    assert sanitized["status"] == "STRUCTURALLY_VALID_UNVERIFIED"
    assert sanitized["verdict"] == "AUTHENTIC"
    assert sanitized["risk_score"] == 5.0
    assert sanitized["model_name"] == "yolov8n_aadhaar_detector"


# ---------------------------------------------------------------------------
# 8. API Boundary Integration Test
# ---------------------------------------------------------------------------
def test_api_screen_boundary_pii_sanitization():
    """Validates that POST /api/screen returns masked PII when screening a sample."""
    res = client.post("/api/screen", data={"sample_id": "mock_aadhaar_01_genuine.png"})
    assert res.status_code == 200
    data = res.json()

    # Raw 12-digit number 542891623811 must not appear anywhere unmasked
    raw_num = "542891623811"
    raw_num_spaced = "5428 9162 3811"
    
    # Check schema fields
    if "schema_fields" in data and "aadhaar_number" in data["schema_fields"]:
        val = data["schema_fields"]["aadhaar_number"]["value"]
        assert val != raw_num
        assert val != raw_num_spaced
        assert "XXXX" in val

    # Check structural checks
    for item in data.get("structural_checks", {}).get("field_details", []):
        assert item.get("value") != raw_num
        assert item.get("value") != raw_num_spaced

    # Check extracted full text
    nlp_sig = data.get("signals", {}).get("nlp_validation", {})
    full_text = nlp_sig.get("extracted_full_text", "")
    assert raw_num not in full_text
    assert raw_num_spaced not in full_text
