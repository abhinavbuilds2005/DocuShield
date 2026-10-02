"""
Unit and Integration Tests for Document Forensics Security Hardening
Tests:
- Upload size limits (> 10MB)
- Magic byte validation and unsupported MIME types
- Corrupt image handling
- Decompression bomb protection
- Path traversal attempts (../, encoded traversal, arbitrary paths)
- Whitelist validation for sample_id
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app, MAX_UPLOAD_BYTES, _ALLOWED_SAMPLE_IDS

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ocr" in data
    assert "ocr_available" in data["ocr"]


def test_upload_file_too_large():
    # Generate in-memory payload exceeding 10 MB limit
    oversized_data = b"x" * (MAX_UPLOAD_BYTES + 1024)
    response = client.post(
        "/api/screen",
        files={"file": ("huge.jpg", io.BytesIO(oversized_data), "image/jpeg")}
    )
    assert response.status_code == 413
    assert response.json()["detail"]["error"] == "FILE_TOO_LARGE"


def test_upload_unsupported_mime_and_magic_bytes():
    # Text or binary data pretending to be a JPG
    fake_data = b"<?php echo 'malicious'; ?>"
    response = client.post(
        "/api/screen",
        files={"file": ("fake.jpg", io.BytesIO(fake_data), "image/jpeg")}
    )
    assert response.status_code == 415
    assert response.json()["detail"]["error"] == "UNSUPPORTED_FORMAT"


def test_upload_corrupt_image():
    # JPEG header but corrupted content
    corrupt_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 30
    response = client.post(
        "/api/screen",
        files={"file": ("corrupt.jpg", io.BytesIO(corrupt_jpeg), "image/jpeg")}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "CORRUPT_IMAGE"


def test_decompression_bomb_protection():
    # Image exceeding MAX_IMAGE_DIMENSION (e.g. 9000 x 9000 pixels)
    img = Image.new("RGB", (9000, 10), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    response = client.post(
        "/api/screen",
        files={"file": ("large_dim.png", buf, "image/png")}
    )
    assert response.status_code == 413
    assert response.json()["detail"]["error"] == "IMAGE_TOO_LARGE"


def test_path_traversal_sample_id():
    # Attempting to access sensitive files using path traversal
    traversal_payloads = [
        "../../../../etc/passwd",
        "..\\..\\..\\Windows\\win.ini",
        "../app.py",
        "..%2F..%2Fapp.py",
        "nonexistent_sample.png"
    ]
    for payload in traversal_payloads:
        response = client.post(
            "/api/screen",
            data={"sample_id": payload}
        )
        assert response.status_code == 404 or response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error"] in ["SAMPLE_NOT_FOUND", "INVALID_PATH"]


def test_path_traversal_image_endpoint():
    traversal_payloads = [
        "../../app.py",
        "..\\app.py",
        "random_file.png"
    ]
    for payload in traversal_payloads:
        response = client.get(f"/api/image/{payload}")
        assert response.status_code in [404, 400]


def test_cors_allowed_local_origin():
    """Verifies that requests from allowed development origins receive valid CORS headers."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejected_unauthorized_origin():
    """Verifies that requests from untrusted origins are NOT granted access-control-allow-origin."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://unauthorized-malicious-domain.com",
            "Access-Control-Request-Method": "GET"
        }
    )
    # Starlette CORS middleware omits the header for unauthorized origins
    assert response.headers.get("access-control-allow-origin") != "https://unauthorized-malicious-domain.com"


def test_cors_origin_parser():
    """Tests normalisation and whitespace/trailing-slash stripping of comma-separated origins."""
    from backend.app import parse_allowed_origins
    parsed = parse_allowed_origins(" https://frontend.example.com/ , http://localhost:5173/ , ")
    assert parsed == ["https://frontend.example.com", "http://localhost:5173"]
    # Empty / whitespace fallback
    default_parsed = parse_allowed_origins("")
    assert "http://localhost:5173" in default_parsed

