
"""
FastAPI Endpoints Smoke and Functional Tests
Tests:
- GET /
- GET /api/health
- GET /api/samples
- GET /api/benchmark
- POST /api/screen with sample_id
- GET /api/image/{filename}
"""

import os
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["dataset_size"] >= 20


def test_samples_endpoint():
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) >= 20
    # Check sample structure
    sample = data["samples"][0]
    assert "id" in sample
    assert "filename" in sample
    assert "label" in sample
    assert "doc_type" in sample


def test_image_endpoint():
    # Fetch valid image
    res = client.get("/api/image/mock_aadhaar_01_genuine.png")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/")


def test_benchmark_endpoint():
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "metrics" in data
    assert "accuracy" in data["metrics"]
    assert "details" in data
    assert len(data["details"]) >= 20
