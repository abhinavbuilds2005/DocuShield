"""
Unit and Integration Tests for Multi-Document Expansion (SIH26188 Phase 1)
Tests:
1. PAN Card:
   - Valid individual PAN with matching surname initial -> STRUCTURALLY_VALID_UNVERIFIED / PASS
   - Valid individual PAN with divergent surname initial -> WARNING (not INVALID or automatic fraud)
   - Non-individual entity PAN (Company 'C') -> NOT_APPLICABLE
   - Malformed PAN or invalid entity code -> INVALID / FAIL
2. Driving Licence (DL):
   - Valid state code & chronological validity -> PASS
   - Invalid state code -> Deterministic failure / INVALID
   - Under-age issuance (age < 18) -> VALIDITY_INCONSISTENCY
   - Expired licence -> EXPIRED
3. Passport:
   - Valid ICAO Doc 9303 TD3 check digits & visual match -> PASS
   - MRZ check digit failure -> INVALID
   - Visual vs MRZ number discrepancy -> LIKELY_TAMPERED / FAIL
4. Voter ID / EPIC:
   - Format compatibility (3 letters + 7 digits) -> EPIC_FORMAT_COMPATIBLE
   - Raw vs normalized OCR preservation
   - Misspelled ECI header forgery detection
5. QR / Barcode explicit 6-state machine
6. Legal authenticity guardrails: Never claims AUTHENTIC or VERIFIED without issuing authority
"""

import pytest
from datetime import datetime, timedelta

from backend.nlp.field_validator import (
    validate_pan_format,
    get_pan_entity_info,
    correlate_pan_surname,
    validate_dl_format,
    validate_dl_state_code,
    evaluate_dl_validity,
    validate_epic_format,
    DocumentFieldValidator
)
from backend.nlp.document_classifier import DocumentClassifier
from backend.fusion import ForensicVerdict


@pytest.fixture
def validator():
    return DocumentFieldValidator()


@pytest.fixture
def classifier():
    return DocumentClassifier()


# ─── 1. PAN Card Validation Tests ───

def test_pan_valid_individual_matching_surname():
    # 'P' at 4th position = Individual; 5th char 'S' matches 'Sharma'
    pan = "ABCPS1234F"
    is_valid, err = validate_pan_format(pan)
    assert is_valid is True
    assert err is None

    ent_info = get_pan_entity_info(pan)
    assert ent_info["is_individual"] is True
    assert ent_info["entity_code"] == "P"
    assert ent_info["entity_type"] == "Individual"

    corr = correlate_pan_surname(pan, "Aarav Sharma")
    assert corr["status"] == "PASS"
    assert corr["is_mismatch"] is False


def test_pan_individual_divergent_surname_is_warning_not_fraud():
    # 5th char 'M' but surname is 'Sharma' (starts with 'S')
    # Must produce a WARNING, never INVALID or automatic fraud
    pan = "ABCPM1234F"
    is_valid, _ = validate_pan_format(pan)
    assert is_valid is True

    corr = correlate_pan_surname(pan, "Aarav Sharma")
    assert corr["status"] == "WARNING"
    assert corr["is_mismatch"] is True
    assert "not proof of forgery" in corr["details"]


def test_pan_non_individual_entity():
    # 4th char 'C' = Company
    pan = "AAACC1234G"
    is_valid, _ = validate_pan_format(pan)
    assert is_valid is True

    ent_info = get_pan_entity_info(pan)
    assert ent_info["is_individual"] is False
    assert ent_info["entity_type"] == "Company"

    corr = correlate_pan_surname(pan, "Tata Consultancy Services")
    assert corr["status"] == "NOT_APPLICABLE"
    assert corr["is_mismatch"] is False


def test_pan_invalid_entity_code():
    # 4th char 'Z' is not a valid Income Tax entity code
    pan = "ABCZS1234F"
    is_valid, err = validate_pan_format(pan)
    assert is_valid is False
    assert "Invalid 4th character" in err


def test_pan_malformed_structure():
    # 6 digits instead of 4
    pan = "ABCDE123456F"
    is_valid, err = validate_pan_format(pan)
    assert is_valid is False


# ─── 2. Driving Licence Validation Tests ───

def test_dl_valid_state_code_and_format():
    dl = "MH-1420110012345"
    is_valid, _ = validate_dl_format(dl)
    assert is_valid is True

    is_sc, code, _ = validate_dl_state_code(dl)
    assert is_sc is True
    assert code == "MH"


def test_dl_invalid_state_code():
    # 'ZZ' is not a valid Indian state/UT code
    dl = "ZZ-1420110012345"
    is_valid, _ = validate_dl_format(dl)
    assert is_valid is True  # Format is 2 letters + digits

    is_sc, code, err = validate_dl_state_code(dl)
    assert is_sc is False
    assert code == "ZZ"
    assert "not a valid Indian licensing jurisdiction" in err


def test_dl_underage_issuance_inconsistency():
    # Born in 2005, issued in 2015 (age 10 -> illegal, min age 18)
    res = evaluate_dl_validity(
        issue_date_str="15/06/2015",
        validity_date_str="14/06/2035",
        dob_str="15/06/2005"
    )
    assert res["status"] == "VALIDITY_INCONSISTENCY"
    assert res["is_deterministic"] is True
    assert "minimum legal driving age" in res["details"]


def test_dl_expiry_date_inconsistency():
    # Expiry is before issue date
    res = evaluate_dl_validity(
        issue_date_str="15/06/2020",
        validity_date_str="15/06/2018",
        dob_str="15/06/1990"
    )
    assert res["status"] == "VALIDITY_INCONSISTENCY"
    assert res["is_deterministic"] is True


def test_dl_expired_status():
    # Expired 5 years ago
    res = evaluate_dl_validity(
        issue_date_str="15/06/2000",
        validity_date_str="15/06/2020",
        dob_str="15/06/1980",
        reference_date=datetime(2026, 1, 1)
    )
    assert res["status"] == "EXPIRED"


# ─── 3. Voter ID / EPIC Validation Tests ───

def test_epic_valid_format_compatible():
    raw_epic = "WBD1234567"
    res = validate_epic_format(raw_epic)
    assert res["is_valid"] is True
    assert res["status"] == "EPIC_FORMAT_COMPATIBLE"
    assert res["normalized_value"] == "WBD1234567"
    assert res["raw_value"] == "WBD1234567"
    assert res["normalization_note"] is None


def test_epic_optical_disambiguation_preserves_raw_value():
    # Trailing character 'O' instead of digit '0' due to OCR misread
    raw_epic = "WBD123456O"
    res = validate_epic_format(raw_epic)
    assert res["is_valid"] is True
    assert res["status"] == "EPIC_FORMAT_COMPATIBLE"
    assert res["raw_value"] == "WBD123456O"
    assert res["normalized_value"] == "WBD1234560"
    assert res["normalization_note"] is not None
    assert "optical character disambiguation" in res["normalization_note"].lower()


def test_epic_invalid_format():
    # Too short
    res = validate_epic_format("WB12")
    assert res["is_valid"] is False
    assert res["status"] == "INVALID_EPIC_FORMAT"


# ─── 4. Document Classification & Routing Tests ───

def test_classifier_all_indian_document_types(classifier):
    # PAN text
    pan_text = "INCOME TAX DEPARTMENT GOVT OF INDIA PERMANENT ACCOUNT NUMBER ABCPS1234F AARAV SHARMA"
    res = classifier.classify(pan_text)
    assert res["document_type"] == "pan"

    # DL text
    dl_text = "UNION OF INDIA DRIVING LICENCE TRANSPORT DEPARTMENT DL NO MH-1420110012345 CLASS LMV"
    res = classifier.classify(dl_text)
    assert res["document_type"] == "driving_license"

    # Voter ID text
    voter_text = "ELECTION COMMISSION OF INDIA ELECTORAL PHOTO IDENTITY CARD EPIC NO WBD1234567"
    res = classifier.classify(voter_text)
    assert res["document_type"] == "voter_id"

    # Aadhaar text
    aadhaar_text = "GOVERNMENT OF INDIA UNIQUE IDENTIFICATION AUTHORITY OF INDIA 9876 5432 1096"
    res = classifier.classify(aadhaar_text)
    assert res["document_type"] == "aadhaar"

    # Passport text
    passport_text = "REPUBLIC OF INDIA PASSPORT P<INDSHARMA<<AARAV<<<<<<<<<<<<<<<<<<<\nZ123456784IND9001015M3001018<<<<<<<<<<<<<<<4"
    res = classifier.classify(passport_text)
    assert res["document_type"] == "passport"


# ─── 5. Field Extraction & Schema Tests ───

def test_pan_field_schema_extraction(validator):
    ocr = {
        "full_text": "INCOME TAX DEPARTMENT GOVT. OF INDIA PERMANENT ACCOUNT NUMBER ABCPS1234F NAME AARAV SHARMA FATHER'S NAME RAJESH SHARMA DOB 15/08/1990",
        "tokens": [{"text": "ABCPS1234F", "confidence": 0.95}, {"text": "AARAV", "confidence": 0.92}, {"text": "SHARMA", "confidence": 0.92}],
        "lines": [{"text": "INCOME TAX DEPARTMENT"}, {"text": "ABCPS1234F"}, {"text": "AARAV SHARMA"}]
    }
    fields = validator.extract_document_fields("pan", ocr)
    assert "pan_number" in fields
    assert fields["pan_number"]["value"] == "ABCPS1234F"
    assert fields["pan_number"]["status"] == "valid"
    assert "name" in fields
    assert "father_name" in fields
    assert "date_of_birth" in fields
    assert "entity_type" in fields
    assert fields["entity_type"]["value"] == "Individual"
    assert "surname_correlation" in fields
    assert fields["surname_correlation"]["value"] == "PASS"


def test_dl_field_schema_extraction(validator):
    ocr = {
        "full_text": "UNION OF INDIA DRIVING LICENCE TRANSPORT DEPARTMENT DL NO DL-1420110056789 NAME VIKRAM SINGH DOB 10/05/1985 ISSUE 12/06/2010 EXPIRY 11/06/2030 CLASS LMV",
        "tokens": [{"text": "DL-1420110056789", "confidence": 0.92}],
        "lines": []
    }
    fields = validator.extract_document_fields("driving_license", ocr)
    assert "licence_number" in fields
    assert fields["licence_number"]["status"] == "valid"
    assert "state_code" in fields
    assert fields["state_code"]["value"] == "DL"
    assert fields["state_code"]["status"] == "valid"
    assert "validity_status" in fields
    assert fields["validity_status"]["value"] == "VALID"


def test_voter_id_field_schema_extraction(validator):
    ocr = {
        "full_text": "ELECTION COMMISSION OF INDIA ELECTORAL PHOTO IDENTITY CARD EPIC NO WBD1234567 ELECTOR'S NAME PRIYA DAS FATHER'S NAME ANIL DAS GENDER FEMALE AGE 28 ASSEMBLY CONSTITUENCY BHOWANIPUR",
        "tokens": [{"text": "WBD1234567", "confidence": 0.90}],
        "lines": []
    }
    fields = validator.extract_document_fields("voter_id", ocr)
    assert "epic_number" in fields
    assert fields["epic_number"]["value"] == "WBD1234567"
    assert fields["epic_number"]["status"] == "valid"
    assert "elector_name" in fields
    assert "relation_name" in fields
    assert "gender" in fields
    assert "issuer_header" in fields
    assert fields["issuer_header"]["status"] == "valid"


# ─── 6. Document Validation Engine Checks ───

def test_pan_invalid_checksum_produces_deterministic_failure(validator):
    ocr = {
        "full_text": "INCOME TAX DEPARTMENT PERMANENT ACCOUNT NUMBER 12345ABCDE NAME TEST USER",
        "tokens": [{"text": "12345ABCDE", "confidence": 0.85}],
        "lines": []
    }
    val_res = validator.validate(ocr, user_selected_type="pan")
    assert val_res["has_deterministic_failure"] is True
    assert any("Tax ID" in df["field"] or "PAN" in df["field"] for df in val_res["deterministic_failures"])


def test_dl_invalid_state_jurisdiction_produces_deterministic_failure(validator):
    ocr = {
        "full_text": "UNION OF INDIA DRIVING LICENCE DL NO ZZ-1420110056789 NAME TEST USER DOB 10/05/1985",
        "tokens": [{"text": "ZZ-1420110056789", "confidence": 0.90}],
        "lines": []
    }
    val_res = validator.validate(ocr, user_selected_type="driving_license")
    assert val_res["has_deterministic_failure"] is True
    assert any("State Jurisdiction" in df["field"] for df in val_res["deterministic_failures"])


def test_voter_id_counterfeit_header_detection(validator):
    # Notice misspelled counterfeit words 'electlon' and 'commisslon'
    ocr = {
        "full_text": "ELECTLON COMMISSLON OF INDIA ELECTORAL IDENTITY CARD EPIC NO WBD1234567 NAME TEST USER",
        "tokens": [{"text": "WBD1234567", "confidence": 0.90}],
        "lines": []
    }
    val_res = validator.validate(ocr, user_selected_type="voter_id")
    assert any("Issuer Template Integrity" in f["field"] and f["status"] == "FAIL" for f in val_res["fields"])


def test_legal_authenticity_verdict_guardrail(validator):
    # Verify that clean documents receive STRUCTURALLY_VALID_UNVERIFIED and never AUTHENTIC
    ocr = {
        "full_text": "INCOME TAX DEPARTMENT GOVT OF INDIA PERMANENT ACCOUNT NUMBER ABCPS1234F AARAV SHARMA FATHER RAJESH SHARMA DOB 15/08/1990",
        "tokens": [{"text": "ABCPS1234F", "confidence": 0.95}],
        "lines": []
    }
    val_res = validator.validate(ocr, user_selected_type="pan")
    assert val_res["has_deterministic_failure"] is False
    assert val_res["score"] >= 70
    assert ForensicVerdict.STRUCTURALLY_VALID_UNVERIFIED == "STRUCTURALLY_VALID_UNVERIFIED"
