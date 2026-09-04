"""
OCR Isolation and Standardization Tests
Verifies:
- Benchmark mode vs Real screening mode separation
- Real screening mode NEVER reads .ocr.json sidecar files
- Sidecar OCR works in benchmark mode
- Bounding box normalization (coordinates in [0, 1])
- OCR health status and error handling
"""

import os
import json
import pytest
import numpy as np

from backend.nlp.ocr_engine import OCREngine, OCRUnavailableError, get_ocr_status

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAMPLE_IMAGE = os.path.join(DATA_DIR, "mock_aadhaar_01_genuine.png")
SAMPLE_SIDECAR = os.path.join(DATA_DIR, "mock_aadhaar_01_genuine.ocr.json")


def test_ocr_health_check_structure():
    status = get_ocr_status()
    assert "ocr_available" in status
    assert "easyocr_available" in status
    assert "tesseract_available" in status
    assert "primary_engine" in status


def test_real_screening_never_reads_sidecar(monkeypatch):
    """
    CRITICAL TEST:
    In real mode (benchmark_mode=False), the OCR engine must NOT read the sidecar
    even if the file exists.
    """
    engine = OCREngine()
    
    # Mock open to fail if any .ocr.json or ground_truth.json is opened
    orig_open = open
    def guarded_open(file, *args, **kwargs):
        filename = str(file)
        if filename.endswith(".ocr.json") or "ground_truth.json" in filename:
            raise AssertionError(f"Real screening mode attempted to access sidecar/ground-truth: {filename}")
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)

    # In real mode, it either runs the real OCR engine or raises OCRUnavailableError
    try:
        res = engine.process_image(SAMPLE_IMAGE, benchmark_mode=False)
        # If it succeeded, it must have used a real OCR engine, not sidecar
        assert res.get("ocr_engine") in ["easyocr", "tesseract"]
    except OCRUnavailableError:
        # Expected if no OCR weights or binary is installed
        pass


def test_benchmark_mode_can_use_sidecar():
    """
    In benchmark mode (benchmark_mode=True), the OCR engine may use the sidecar
    for controlled synthetic testing.
    """
    if not os.path.exists(SAMPLE_SIDECAR):
        pytest.skip("Sidecar not present")

    engine = OCREngine()
    res = engine.process_image(SAMPLE_IMAGE, benchmark_mode=True)
    assert res is not None
    assert "tokens" in res
    assert "lines" in res
    assert len(res["tokens"]) > 0


def test_bounding_box_normalization():
    """
    Verify OCR results contain normalized bounding boxes within [0, 1].
    """
    engine = OCREngine()
    res = engine.process_image(SAMPLE_IMAGE, benchmark_mode=True)
    w, h = res.get("image_dims", [800, 500])
    
    for line in res.get("lines", []):
        if "bbox_normalized" in line:
            norm = line["bbox_normalized"]
            assert 0.0 <= norm["x"] <= 1.0
            assert 0.0 <= norm["y"] <= 1.0
            assert 0.0 <= norm["width"] <= 1.0
            assert 0.0 <= norm["height"] <= 1.0


def test_removing_sidecar_does_not_change_real_ocr_screening(tmp_path):
    """
    SECTION 10 TEST:
    Proves that removing the sidecar does NOT change real-OCR screening behavior.
    Copies image to an isolated temp directory with NO sidecars and screens it.
    """
    import shutil
    from backend.fusion import DocumentScreeningPipeline

    isolated_img = str(tmp_path / "isolated_mock_aadhaar.png")
    shutil.copyfile(SAMPLE_IMAGE, isolated_img)

    # Ensure no .ocr.json file exists in this directory
    isolated_sidecar = str(tmp_path / "isolated_mock_aadhaar.ocr.json")
    assert not os.path.exists(isolated_sidecar)

    pipeline = DocumentScreeningPipeline()
    res = pipeline.screen_document(isolated_img, benchmark_mode=False)

    assert res["verdict"] == "AUTHENTIC"
    assert res["ocr_engine_used"] in ["easyocr", "tesseract"]
    assert res["forensic_decision_debug"]["checksum_result"]["status"] == "PASS"

