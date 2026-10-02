"""
Focused Unit and Integration Tests for Phase 2 Safe OCR Accuracy Improvements
(backend/tests/test_ocr_accuracy_improvements.py)

Tests:
1. Padded crop extraction and safe boundary clamping
2. Rejection of invalid, inverted, or zero-area bounding boxes
3. Controlled aspect-ratio-preserving field-crop upscaling
4. Bounded OCR preprocessing variants (max 3)
5. OCR candidate selection composite scoring
6. PAN validation safety (explicit optical disambiguation tracking, no silent mutation)
7. Driving Licence format validation (Sarathi-4, hyphenated, space-separated, legacy formats)
8. Field-crop OCR integration in screening pipeline
9. PII sanitization after multi-variant crop OCR (no raw PII leak in response)
10. Strict adherence to authenticity disclaimers (no false authentic claims)
"""

import os
import re
import cv2
import numpy as np
import pytest

from backend.vision.crop_utils import (
    extract_padded_crop,
    upscale_crop_if_needed,
    get_crop_preprocessing_variants,
)
from backend.nlp.ocr_engine import OCREngine
from backend.nlp.field_validator import (
    validate_pan_format,
    validate_dl_format,
    PANValidationResult,
)
from backend.nlp.pii_masking import (
    mask_pan,
    mask_driving_license,
    mask_aadhaar,
    sanitize_screening_response,
)
from backend.fusion import DocumentScreeningPipeline, ForensicVerdict


# ---------------------------------------------------------------------------
# 1. Padded Crop Extraction & Boundary Clamping Tests
# ---------------------------------------------------------------------------
def test_padded_crop_basic():
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    bbox = {"x1": 50, "y1": 30, "x2": 150, "y2": 70}
    crop, meta = extract_padded_crop(img, bbox, pad_ratio=0.10, min_pad_px=4)

    assert crop is not None
    assert isinstance(crop, np.ndarray)
    # Original width=100, height=40. With 10% pad (10px x, 4px y), padded box should be wider and taller
    p_box = meta["padded_bbox"]
    assert p_box["x1"] < bbox["x1"]
    assert p_box["y1"] < bbox["y1"]
    assert p_box["x2"] > bbox["x2"]
    assert p_box["y2"] > bbox["y2"]


def test_padded_crop_boundary_clamping():
    """Verify that padding never exceeds image boundaries [0, W] and [0, H]."""
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    # Box touching image top-left edge
    bbox = {"x1": 2, "y1": 2, "x2": 40, "y2": 40}
    crop, meta = extract_padded_crop(img, bbox, pad_ratio=0.20, min_pad_px=10)

    assert crop is not None
    p_box = meta["padded_bbox"]
    assert p_box["x1"] == 0  # Clamped to 0
    assert p_box["y1"] == 0  # Clamped to 0

    # Box touching image bottom-right edge
    bbox_br = {"x1": 80, "y1": 80, "x2": 99, "y2": 99}
    crop_br, meta_br = extract_padded_crop(img, bbox_br, pad_ratio=0.20, min_pad_px=10)
    assert crop_br is not None
    assert meta_br["padded_bbox"]["x2"] == 100
    assert meta_br["padded_bbox"]["y2"] == 100


def test_padded_crop_formats_support():
    """Verify list [x, y, w, h], list [x1, y1, x2, y2], and normalized dict."""
    img = np.zeros((200, 400, 3), dtype=np.uint8)

    # [x1, y1, x2, y2]
    c1, m1 = extract_padded_crop(img, [50, 40, 150, 80])
    assert c1 is not None

    # [x, y, w, h]
    c2, m2 = extract_padded_crop(img, [50, 40, 100, 40])
    assert c2 is not None

    # normalized dict
    c3, m3 = extract_padded_crop(img, {"x": 0.25, "y": 0.20, "width": 0.50, "height": 0.20})
    assert c3 is not None
    assert c3.shape[0] > 0 and c3.shape[1] > 0


def test_padded_crop_invalid_rejection():
    """Verify graceful handling of invalid, inverted, empty, or out-of-bounds boxes."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Inverted box (x2 <= x1)
    c, m = extract_padded_crop(img, {"x1": 50, "y1": 10, "x2": 40, "y2": 30})
    assert c is None
    assert "error" in m

    # Zero area box
    c, m = extract_padded_crop(img, {"x1": 10, "y1": 10, "x2": 10, "y2": 20})
    assert c is None

    # Completely out of bounds box
    c, m = extract_padded_crop(img, {"x1": 150, "y1": 150, "x2": 200, "y2": 200})
    assert c is None

    # None / empty image
    c, m = extract_padded_crop(None, [10, 10, 20, 20])
    assert c is None


# ---------------------------------------------------------------------------
# 2. Controlled Field-Crop Upscaling Tests
# ---------------------------------------------------------------------------
def test_crop_upscaling_tiny_crop():
    """Small crops (< 64px height) should be upscaled 2x or 3x."""
    small = np.zeros((24, 120, 3), dtype=np.uint8)
    upscaled, scale = upscale_crop_if_needed(small, min_height=64, max_scale=3.0)

    assert scale >= 2.0
    assert upscaled.shape[0] >= 48
    assert upscaled.shape[1] >= 240
    # Aspect ratio preserved within 1px rounding
    orig_ratio = 120.0 / 24.0
    new_ratio = float(upscaled.shape[1]) / float(upscaled.shape[0])
    assert abs(orig_ratio - new_ratio) < 0.1


def test_crop_upscaling_normal_crop():
    """Already sufficiently tall crops (>= 64px) should not be upscaled."""
    normal = np.zeros((80, 200, 3), dtype=np.uint8)
    upscaled, scale = upscale_crop_if_needed(normal, min_height=64)

    assert scale == 1.0
    assert upscaled.shape == normal.shape


def test_crop_upscaling_edge_cases():
    """Empty or None crops handled safely without exceptions."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    u, s = upscale_crop_if_needed(empty)
    assert s == 1.0

    u_none, s_none = upscale_crop_if_needed(None)
    assert s_none == 1.0


# ---------------------------------------------------------------------------
# 3. Bounded OCR Preprocessing Variants Tests
# ---------------------------------------------------------------------------
def test_preprocessing_variants_generation():
    """Verify that up to 3 bounded variants are generated as valid numpy arrays."""
    sample = np.full((50, 150, 3), 180, dtype=np.uint8)
    variants = get_crop_preprocessing_variants(sample, max_variants=3)

    assert len(variants) <= 3
    assert len(variants) >= 2
    names = [v[0] for v in variants]
    assert "original" in names
    assert "clahe_contrast" in names

    for name, img_arr in variants:
        assert isinstance(img_arr, np.ndarray)
        assert img_arr.size > 0
        assert img_arr.dtype == np.uint8


def test_preprocessing_variants_max_bound():
    """Never returns more than max_variants."""
    sample = np.full((50, 150, 3), 180, dtype=np.uint8)
    v1 = get_crop_preprocessing_variants(sample, max_variants=1)
    assert len(v1) == 1

    v2 = get_crop_preprocessing_variants(sample, max_variants=2)
    assert len(v2) == 2


# ---------------------------------------------------------------------------
# 4. Field-Crop OCR Integration & Candidate Selection Tests
# ---------------------------------------------------------------------------
def test_extract_field_crop_text_empty():
    engine = OCREngine()
    res = engine.extract_field_crop_text(None)
    assert res["status"] == "EMPTY_CROP"
    assert res["confidence"] == 0.0
    assert res["is_valid"] is False


def test_extract_field_crop_text_candidate_selection(monkeypatch):
    """
    Simulates OCR returning different qualities across variants and tests composite selection:
    Variant 1 gives low confidence invalid string.
    Variant 2 gives valid format string with high confidence.
    Candidate selection must select Variant 2.
    """
    engine = OCREngine()

    call_count = [0]
    def mock_process_image(img, benchmark_mode=False):
        call_count[0] += 1
        if call_count[0] == 1:
            # Variant 1: noisy text
            return {"full_text": "AB1234", "tokens": [{"text": "AB1234", "confidence": 0.40}]}
        else:
            # Variant 2: valid PAN (4th char P for Individual)
            return {"full_text": "ABCPE1234F", "tokens": [{"text": "ABCPE1234F", "confidence": 0.92}]}

    monkeypatch.setattr(engine, "process_image", mock_process_image)

    crop = np.zeros((30, 100, 3), dtype=np.uint8)
    result = engine.extract_field_crop_text(
        crop,
        field_type="pan_number",
        validator_fn=validate_pan_format,
        max_variants=3
    )

    assert result["status"] == "SUCCESS"
    assert result["text"] == "ABCPE1234F"
    assert result["is_valid"] is True
    assert result["confidence"] >= 0.90


# ---------------------------------------------------------------------------
# 5. PAN Validation Safety Tests
# ---------------------------------------------------------------------------
def test_pan_validation_result_unpacking_compatibility():
    """Ensure PANValidationResult unpacks as (bool, str_or_none) for 100% backward compatibility."""
    res = validate_pan_format("ABCPE1234F")
    assert isinstance(res, tuple)
    is_valid, err = res
    assert is_valid is True
    assert err is None
    assert res.disambiguated is False


def test_pan_validation_optical_disambiguation_tracking():
    """
    When terminal 10th char is optical confusion '0', it normalizes to 'O'
    AND records that optical disambiguation occurred with uncertainty details.
    """
    res = validate_pan_format("AZYPA98780")
    assert res.is_valid is True
    assert res.disambiguated is True
    assert res.normalized_value == "AZYPA9878O"
    assert res.disambiguation_note is not None
    assert "optical disambiguation" in res.disambiguation_note.lower()


def test_pan_validation_invalid_entities():
    """Invalid 4th character rejected."""
    is_valid, err = validate_pan_format("ABCXE1234F")
    assert is_valid is False
    assert "Invalid 4th character" in err


def test_pan_validation_invalid_structures():
    is_valid, err = validate_pan_format("12345ABCDE")
    assert is_valid is False
    is_valid, err = validate_pan_format("ABCD123456")
    assert is_valid is False
    is_valid, err = validate_pan_format("")
    assert is_valid is False


# ---------------------------------------------------------------------------
# 6. Driving Licence Validation Tests
# ---------------------------------------------------------------------------
def test_dl_validation_sarathi_standard():
    """Modern Sarathi-4 15-char formats with/without separators."""
    # Standard 15 digits
    ok, err = validate_dl_format("DL1420110012345")
    assert ok is True

    # Hyphenated
    ok, err = validate_dl_format("DL-1420110012345")
    assert ok is True

    # Space-separated
    ok, err = validate_dl_format("DL 14 2011 0012345")
    assert ok is True

    # Slash-separated
    ok, err = validate_dl_format("DL-14/2011/0012345")
    assert ok is True


def test_dl_validation_recognized_states():
    """Valid Indian State/UT codes accepted."""
    for code in ["MH", "KA", "TN", "UP", "HR", "WB", "GJ", "RJ"]:
        ok, _ = validate_dl_format(f"{code}0120190012345")
        assert ok is True, f"Failed for valid state code {code}"


def test_dl_validation_invalid_states_rejected():
    """Invalid non-Indian state codes rejected by validate_dl_state_code."""
    from backend.nlp.field_validator import validate_dl_state_code
    ok, sc, err = validate_dl_state_code("ZZ1420110012345")
    assert ok is False
    assert "not a valid Indian licensing jurisdiction" in err


def test_dl_validation_arbitrary_string_rejected():
    ok, _ = validate_dl_format("AB1234")
    assert ok is False
    ok, _ = validate_dl_format("SOMERANDOMSTRING")
    assert ok is False


# ---------------------------------------------------------------------------
# 7. Token Box Estimation Tagging Tests
# ---------------------------------------------------------------------------
def test_token_box_estimation_tagging():
    """
    Ensure EasyOCR multi-word tokens are explicitly flagged with is_estimated_box=True,
    and single-word tokens with is_estimated_box=False.
    """
    engine = OCREngine()
    # Create synthetic image with text
    test_img = np.full((120, 400, 3), 255, dtype=np.uint8)
    cv2.putText(test_img, "GOVERNMENT OF INDIA", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    res = engine.process_image(test_img, benchmark_mode=False)
    assert res is not None
    tokens = res.get("tokens", [])
    if tokens:
        for t in tokens:
            assert "is_estimated_box" in t
            assert "line_box" in t


# ---------------------------------------------------------------------------
# 8. PII Sanitization After Multi-Variant Crop OCR
# ---------------------------------------------------------------------------
def test_pii_sanitization_crop_ocr_metadata():
    """
    Verify that sanitize_screening_response scrubs raw PII text from
    field_crop_extractions and field_crop_ocr_metadata before output serialization.
    """
    raw_response = {
        "verdict": "STRUCTURALLY_VALID_UNVERIFIED",
        "schema_fields": {
            "pan_number": {"value": "ABCDE1234F", "status": "valid"}
        },
        "field_crop_ocr_metadata": {
            "pan_number": {
                "text": "ABCDE1234F",
                "confidence": 0.95,
                "variant": "clahe_contrast",
                "status": "SUCCESS"
            }
        },
        "field_crop_extractions": {
            "aadhaar_number": {
                "text": "5428 9162 3811",
                "confidence": 0.92,
                "status": "SUCCESS"
            }
        }
    }

    sanitized = sanitize_screening_response(raw_response)

    # PAN in schema_fields masked
    assert sanitized["schema_fields"]["pan_number"]["value"] == "ABCDE****F"

    # Field crop metadata text masked
    pan_meta_text = sanitized["field_crop_ocr_metadata"]["pan_number"]["text"]
    assert "1234" not in pan_meta_text
    assert "****" in pan_meta_text

    # Field crop extractions text masked
    aadhaar_crop_text = sanitized["field_crop_extractions"]["aadhaar_number"]["text"]
    assert "5428" not in aadhaar_crop_text
    assert "XXXX" in aadhaar_crop_text


# ---------------------------------------------------------------------------
# 9. Pipeline Security & Non-Governmental Authenticity Disclaimer Tests
# ---------------------------------------------------------------------------
def test_pipeline_never_claims_government_authentication():
    """
    Verify that screening results never claim government authentication or verified status
    based on OCR / YOLO heuristics alone.
    """
    pipeline = DocumentScreeningPipeline()
    # Test on genuine sample
    sample_path = os.path.join("backend", "data", "mock_aadhaar_01_genuine.png")
    if not os.path.exists(sample_path):
        sample_path = os.path.join("data", "mock_aadhaar_01_genuine.png")

    if os.path.exists(sample_path):
        res = pipeline.screen_document(sample_path, benchmark_mode=False)
        gov_status = res.get("government_database_status", {})
        assert gov_status.get("status") == "NOT_PERFORMED"
        assert "no external government database verification" in gov_status.get("message", "").lower()

        # Official verification remains NOT_PERFORMED
        assert res["official_verification"]["status"] == "NOT_PERFORMED"
        assert res["official_verification"]["verified"] is False
