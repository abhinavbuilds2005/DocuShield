"""
Unit Tests for Document Type Classification and Field Schemas (SIH26188)
Tests:
- Passport classification and schema extraction
- Visa classification and schema extraction
- National ID classification and schema extraction
- Driving Licence classification and schema extraction
- Permit classification and schema extraction
- Unknown document type fallback
"""

import pytest
from backend.nlp.document_classifier import DocumentClassifier, DOCUMENT_TYPES
from backend.nlp.field_validator import DocumentFieldValidator


@pytest.fixture
def classifier():
    return DocumentClassifier()


@pytest.fixture
def validator():
    return DocumentFieldValidator()


def test_supported_document_types_constant():
    assert "passport" in DOCUMENT_TYPES
    assert "visa" in DOCUMENT_TYPES
    assert "national_id" in DOCUMENT_TYPES
    assert "driving_license" in DOCUMENT_TYPES
    assert "permit" in DOCUMENT_TYPES


def test_classify_user_selected_override(classifier):
    res = classifier.classify("Random uninformative text", user_selected_type="passport")
    assert res["document_type"] == "passport"
    assert res["confidence"] == 1.0
    assert res["method"] == "user_selected"


def test_classify_passport_mrz(classifier):
    text = "REPUBLIC OF UTOPIA PASSPORT\nP<UTODOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nU123456784UTO8808154M3208148<<<<<<<<<<<<<<<4"
    res = classifier.classify(text)
    assert res["document_type"] == "passport"
    assert res["confidence"] >= 0.70


def test_classify_visa_keywords(classifier):
    text = "ENTRY VISA TYPE TOURIST VALID FOR MULTIPLE ENTRIES DURATION OF STAY 90 DAYS CONSULATE GENERAL"
    res = classifier.classify(text)
    assert res["document_type"] == "visa"
    assert res["confidence"] >= 0.60


def test_classify_driving_licence(classifier):
    text = "UNION OF INDIA DRIVING LICENCE TRANSPORT DEPARTMENT DL NO DL-0420110056789 CLASS LMV"
    res = classifier.classify(text)
    assert res["document_type"] == "driving_license"
    assert res["confidence"] >= 0.60


def test_classify_permit(classifier):
    text = "SPECIAL TRAVEL AUTHORIZATION PERMIT NO TA-2026-98124 IMMIGRATION BORDER CONTROL"
    res = classifier.classify(text)
    assert res["document_type"] == "permit"
    assert res["confidence"] >= 0.60


def test_classify_unknown_document(classifier):
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    res = classifier.classify(text)
    assert res["document_type"] == "unknown"
    assert res["confidence"] <= 0.40
    assert "could not be determined confidently" in res["explanation"]


def test_passport_field_schema_extraction(validator):
    ocr_res = {
        "full_text": "REPUBLIC OF UTOPIA PASSPORT NO U12345678 CITIZEN JOHN DOE NATIONALITY UTOPIAN DOB 15/08/1988 EXPIRY 14/08/2032 MALE\nP<UTODOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nU123456784UTO8808154M3208148<<<<<<<<<<<<<<<4",
        "tokens": [{"text": "U12345678", "confidence": 0.95}],
        "lines": [
            {"text": "P<UTODOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"},
            {"text": "U123456784UTO8808154M3208148<<<<<<<<<<<<<<<4"}
        ]
    }
    fields = validator.extract_document_fields("passport", ocr_res)
    assert "passport_number" in fields
    assert "full_name" in fields
    assert "nationality" in fields
    assert "date_of_birth" in fields
    assert "date_of_expiry" in fields
    assert "gender" in fields
    assert "issuing_country" in fields
    assert "mrz" in fields
    assert fields["passport_number"]["value"] == "U12345678"
    assert fields["passport_number"]["status"] == "valid"


def test_visa_field_schema_extraction(validator):
    ocr_res = {
        "full_text": "VISA NO V87654321 HOLDER JANE SMITH NATIONALITY UTOPIAN TOURIST MULTIPLE ENTRIES 01/01/2025 31/12/2025",
        "tokens": [{"text": "V87654321", "confidence": 0.92}],
        "lines": []
    }
    fields = validator.extract_document_fields("visa", ocr_res)
    assert "visa_number" in fields
    assert "holder_name" in fields
    assert "nationality" in fields
    assert "visa_type" in fields
    assert "entries" in fields
    assert fields["visa_number"]["value"] == "V87654321"


def test_permit_field_schema_extraction(validator):
    ocr_res = {
        "full_text": "TRAVEL AUTHORIZATION PERMIT NO TA-998877 ROBERTO SILVA NATIONALITY BRAZILIAN 15/02/2025 14/08/2026",
        "tokens": [{"text": "TA-998877", "confidence": 0.90}],
        "lines": []
    }
    fields = validator.extract_document_fields("permit", ocr_res)
    assert "document_number" in fields
    assert "name" in fields
    assert "nationality" in fields
    assert "permit_type" in fields
    assert fields["document_number"]["value"] == "TA-998877"
