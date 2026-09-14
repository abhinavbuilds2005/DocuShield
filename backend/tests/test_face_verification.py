"""
Unit Tests for Face Verification Module (SIH26188)
Tests:
- Missing person image returns status NOT_PERFORMED with exact required disclaimer
- Image without face handled gracefully
- Matching face pair vs mismatching face pair
- Similarity score boundaries [0.0 - 1.0]
"""

import pytest
import numpy as np
import cv2
from backend.forensics.face_verifier import FaceVerifier


@pytest.fixture
def verifier():
    return FaceVerifier()


def test_missing_person_image(verifier):
    dummy_doc = np.full((300, 300, 3), 200, dtype=np.uint8)
    res = verifier.verify(dummy_doc, person_img=None)
    assert res["status"] == "NOT_PERFORMED"
    assert res["face_match"] is None
    assert "person image not provided" in res["explanation"]


def test_image_without_face(verifier):
    dummy_doc = np.zeros((300, 300, 3), dtype=np.uint8)
    dummy_person = np.zeros((300, 300, 3), dtype=np.uint8)
    res = verifier.verify(dummy_doc, person_img=dummy_person)
    assert res["status"] in ["NO_FACE_IN_DOCUMENT", "NO_FACE_IN_PERSON_IMAGE"]
    assert res["face_match"] is None


def test_face_similarity_computation(verifier):
    # Two identical mock faces
    mock_face = np.full((128, 128, 3), 150, dtype=np.uint8)
    sim = verifier.compute_similarity(mock_face, mock_face)
    assert 0.0 <= sim <= 1.0
    assert sim >= 0.80

    # Extremely different mock face
    diff_face = np.zeros((128, 128, 3), dtype=np.uint8)
    sim_diff = verifier.compute_similarity(mock_face, diff_face)
    assert sim_diff < sim
