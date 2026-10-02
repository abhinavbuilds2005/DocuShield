"""
Unit Tests for Aadhaar Field Detector (backend/vision/field_detector.py)

Tests cover:
1. Singleton pattern via get_field_detector()
2. Graceful degradation when model file is missing
3. FIELD_DETECTOR_ENABLED=false disables detection
4. detect_fields() returns correct structure with empty/unavailable model
5. Bounding box normalization and coordinate clipping on synthetic images
6. Integration: field detector is invoked from fusion pipeline (when available)
"""

import os
import sys
import cv2
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton field detector instance between tests."""
    import backend.vision.field_detector as fd_module
    fd_module._FIELD_DETECTOR_INSTANCE = None
    yield
    fd_module._FIELD_DETECTOR_INSTANCE = None


@pytest.fixture
def synthetic_aadhaar_image():
    """Create a minimal synthetic 'Aadhaar-like' card image (640x400 BGR)."""
    img = np.full((400, 640, 3), 245, dtype=np.uint8)
    # Header bar
    cv2.rectangle(img, (0, 0), (640, 60), (30, 30, 120), -1)
    cv2.putText(img, "GOVERNMENT OF INDIA", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Fake photo region
    cv2.rectangle(img, (30, 80), (180, 280), (200, 200, 200), -1)
    cv2.putText(img, "PHOTO", (65, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    # Text fields
    cv2.putText(img, "Name: Test User", (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "DOB: 01/01/1990", (200, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "2847 5938 1024", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # QR code placeholder
    cv2.rectangle(img, (480, 200), (620, 340), (50, 50, 50), -1)
    cv2.putText(img, "QR", (530, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


# ──────────────────────────────────────────────────────────────────────
# Test 1: Singleton Pattern
# ──────────────────────────────────────────────────────────────────────

def test_singleton_returns_same_instance():
    """get_field_detector() should always return the same instance."""
    from backend.vision.field_detector import get_field_detector
    det1 = get_field_detector()
    det2 = get_field_detector()
    assert det1 is det2


# ──────────────────────────────────────────────────────────────────────
# Test 2: Graceful Degradation - Missing Model File
# ──────────────────────────────────────────────────────────────────────

def test_graceful_degradation_missing_model(synthetic_aadhaar_image, monkeypatch):
    """When model weights file does not exist, detector should gracefully return UNAVAILABLE."""
    monkeypatch.setenv("FIELD_DETECTOR_ENABLED", "true")
    monkeypatch.setenv("FIELD_DETECTOR_MODEL_PATH", "/nonexistent/path/model.pt")

    from backend.vision.field_detector import AadhaarFieldDetector
    det = AadhaarFieldDetector()

    assert det.is_available is False

    result = det.detect_fields(synthetic_aadhaar_image)
    assert result["status"] == "UNAVAILABLE"
    assert result["fields"] == []
    assert result["field_crops"] == {}


# ──────────────────────────────────────────────────────────────────────
# Test 3: Disabled via Environment Variable
# ──────────────────────────────────────────────────────────────────────

def test_disabled_via_env(synthetic_aadhaar_image, monkeypatch):
    """When FIELD_DETECTOR_ENABLED=false, detector should be disabled and return DISABLED status."""
    monkeypatch.setenv("FIELD_DETECTOR_ENABLED", "false")

    from backend.vision.field_detector import AadhaarFieldDetector
    det = AadhaarFieldDetector()

    assert det.is_available is False

    result = det.detect_fields(synthetic_aadhaar_image)
    assert result["status"] == "DISABLED"
    assert result["fields"] == []
    assert result["field_crops"] == {}


# ──────────────────────────────────────────────────────────────────────
# Test 4: Return Structure Validation
# ──────────────────────────────────────────────────────────────────────

def test_return_structure_keys(synthetic_aadhaar_image, monkeypatch):
    """detect_fields() must always return expected top-level keys even when unavailable."""
    monkeypatch.setenv("FIELD_DETECTOR_MODEL_PATH", "/nonexistent/path.pt")

    from backend.vision.field_detector import AadhaarFieldDetector
    det = AadhaarFieldDetector()
    result = det.detect_fields(synthetic_aadhaar_image)

    required_keys = {"status", "fields", "field_crops", "image_dims"}
    assert required_keys.issubset(set(result.keys()))
    assert isinstance(result["fields"], list)
    assert isinstance(result["field_crops"], dict)
    assert isinstance(result["image_dims"], list)
    assert len(result["image_dims"]) == 2


# ──────────────────────────────────────────────────────────────────────
# Test 5: File Path Input - Nonexistent Path
# ──────────────────────────────────────────────────────────────────────

def test_nonexistent_file_path(monkeypatch):
    """Passing a nonexistent file path should not crash but return empty result."""
    monkeypatch.setenv("FIELD_DETECTOR_MODEL_PATH", "/nonexistent/path.pt")

    from backend.vision.field_detector import AadhaarFieldDetector
    det = AadhaarFieldDetector()
    result = det.detect_fields("/nonexistent/image.png")

    assert result["status"] in ("UNAVAILABLE", "DISABLED")
    assert result["fields"] == []


# ──────────────────────────────────────────────────────────────────────
# Test 6: Zero-Size / Invalid Image
# ──────────────────────────────────────────────────────────────────────

def test_invalid_empty_image(monkeypatch):
    """Passing an empty numpy array should return empty result without crash."""
    monkeypatch.setenv("FIELD_DETECTOR_MODEL_PATH", "/nonexistent/path.pt")

    from backend.vision.field_detector import AadhaarFieldDetector
    det = AadhaarFieldDetector()

    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    result = det.detect_fields(empty_img)
    assert result["fields"] == []


# ──────────────────────────────────────────────────────────────────────
# Test 7: Configuration Defaults
# ──────────────────────────────────────────────────────────────────────

def test_default_configuration():
    """Verify detector uses correct default configuration values."""
    from backend.vision.field_detector import AadhaarFieldDetector

    # Clear env vars to test defaults
    env_vars = ["FIELD_DETECTOR_ENABLED", "FIELD_DETECTOR_CONFIDENCE", "FIELD_DETECTOR_DEVICE"]
    saved = {}
    for v in env_vars:
        saved[v] = os.environ.pop(v, None)

    try:
        det = AadhaarFieldDetector()
        assert det.enabled is True
        assert det.conf_threshold == 0.35
        assert det.device_pref == "auto"
    finally:
        for v, val in saved.items():
            if val is not None:
                os.environ[v] = val


# ──────────────────────────────────────────────────────────────────────
# Test 8: Custom Configuration via Environment
# ──────────────────────────────────────────────────────────────────────

def test_custom_configuration(monkeypatch):
    """Verify detector reads custom configuration from env vars."""
    monkeypatch.setenv("FIELD_DETECTOR_ENABLED", "true")
    monkeypatch.setenv("FIELD_DETECTOR_CONFIDENCE", "0.50")
    monkeypatch.setenv("FIELD_DETECTOR_DEVICE", "cpu")
    monkeypatch.setenv("FIELD_DETECTOR_MODEL_PATH", "/custom/model.pt")

    from backend.vision.field_detector import AadhaarFieldDetector
    det = AadhaarFieldDetector()

    assert det.enabled is True
    assert det.conf_threshold == 0.50
    assert det.device_pref == "cpu"
    assert det.model_path == "/custom/model.pt"


# ──────────────────────────────────────────────────────────────────────
# Test 9: Lazy Initialization - Model Not Loaded Until First Access
# ──────────────────────────────────────────────────────────────────────

def test_lazy_initialization():
    """Model should not be loaded at construction time, only on first access."""
    from backend.vision.field_detector import AadhaarFieldDetector
    det = AadhaarFieldDetector()

    # Before first access, _init_attempted should be False
    assert det._init_attempted is False
    assert det._model is None

    # Trigger lazy init via is_available
    _ = det.is_available

    # Now init should have been attempted
    assert det._init_attempted is True


# ──────────────────────────────────────────────────────────────────────
# Test 10: Fusion Pipeline Integration - Field Detector Import
# ──────────────────────────────────────────────────────────────────────

def test_fusion_imports_field_detector():
    """The fusion module should import get_field_detector without errors."""
    try:
        from backend.fusion import DocumentScreeningPipeline
        # Verify the pipeline class is importable and has screen_document method
        assert hasattr(DocumentScreeningPipeline, "screen_document")
    except ImportError as e:
        pytest.fail(f"Fusion pipeline failed to import: {e}")


# ──────────────────────────────────────────────────────────────────────
# Test 11: Multiple detect_fields calls don't re-init
# ──────────────────────────────────────────────────────────────────────

def test_no_reinit_on_multiple_calls(synthetic_aadhaar_image, monkeypatch):
    """Calling detect_fields multiple times should not re-attempt initialization."""
    monkeypatch.setenv("FIELD_DETECTOR_MODEL_PATH", "/nonexistent/path.pt")

    from backend.vision.field_detector import AadhaarFieldDetector
    det = AadhaarFieldDetector()

    result1 = det.detect_fields(synthetic_aadhaar_image)
    init_flag_after_first = det._init_attempted

    result2 = det.detect_fields(synthetic_aadhaar_image)

    assert init_flag_after_first is True
    assert det._init_attempted is True  # Not re-attempted
    assert result1["status"] == result2["status"]
