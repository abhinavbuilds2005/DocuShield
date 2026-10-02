"""
Unit and Integration Tests for Aadhaar Tampering Scenarios & Forensic Screening
Covers:
1. Clean Document -> STRUCTURALLY_VALID_UNVERIFIED (legacy: AUTHENTIC), official_verification NOT_PERFORMED
2. Edited Digit with Invalid Verhoeff Checksum -> INVALID / FLAGGED deterministic failure
3. Specific Failure Case: Edited Digit with VALID Verhoeff Checksum caught by QR Cross-Field Mismatch
4. Photo / Spliced Seam Tampering -> LIKELY_TAMPERED or SUSPICIOUS with human review required
5. Low-Quality / Heavily Blurred Document -> MANUAL_REVIEW_REQUIRED
"""

import os
import cv2
import numpy as np
import pytest
import qrcode
from qrcode import make as make_qr
from pathlib import Path

from backend.fusion import DocumentScreeningPipeline, ForensicVerdict
from backend.nlp.field_validator import generate_verhoeff_checksum, validate_verhoeff


@pytest.fixture(scope="module")
def pipeline():
    return DocumentScreeningPipeline()


@pytest.fixture
def clean_aadhaar_data():
    """Generates valid 11-digit prefix and computes legitimate Verhoeff checksum."""
    base_11 = "28475938102"
    chk = generate_verhoeff_checksum(base_11)
    valid_aadhaar = f"{base_11}{chk}"
    assert validate_verhoeff(valid_aadhaar) is True
    return {
        "number": valid_aadhaar,
        "formatted": f"{valid_aadhaar[:4]} {valid_aadhaar[4:8]} {valid_aadhaar[8:]}",
        "name": "ARJUN MEHTA",
        "dob": "15/08/1992",
        "gender": "MALE"
    }


def _create_synthetic_aadhaar_image(
    aadhaar_str: str,
    name: str = "ARJUN MEHTA",
    dob: str = "15/08/1992",
    gender: str = "MALE",
    qr_payload: str = None,
    add_splice: bool = False,
    apply_blur: bool = False
) -> np.ndarray:
    """Generates an in-memory synthetic Aadhaar card image."""
    w, h = 800, 500
    img = np.ones((h, w, 3), dtype=np.uint8) * 250

    # Header / Emblem
    cv2.rectangle(img, (40, 20), (w - 40, 75), (230, 240, 250), -1)
    cv2.putText(img, "GOVERNMENT OF INDIA", (200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.circle(img, (100, 48), 20, (180, 100, 50), -1)

    # Photo portrait
    px1, py1, px2, py2 = 60, 100, 200, 270
    cv2.rectangle(img, (px1, py1), (px2, py2), (210, 210, 210), -1)
    cv2.circle(img, ((px1 + px2) // 2, py1 + 60), 35, (120, 120, 120), -1)
    cv2.ellipse(img, ((px1 + px2) // 2, py2 + 10), (50, 45), 0, 0, 180, (90, 90, 90), -1)
    cv2.rectangle(img, (px1, py1), (px2, py2), (100, 100, 100), 2)

    if add_splice:
        # Simulate an obvious pasted/spliced patch with high edge disparity
        splice_patch = np.zeros((100, 100, 3), dtype=np.uint8)
        splice_patch[:, :] = [30, 30, 220]  # bright red artifact
        img[py1+10:py1+110, px1+10:px1+110] = splice_patch

    # Name
    cv2.putText(img, name, (230, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    # DOB
    cv2.putText(img, f"DOB: {dob}", (230, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    # Gender
    cv2.putText(img, f"GENDER: {gender}", (230, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    # QR Code
    qrx1, qry1, qrx2, qry2 = 600, 80, 760, 240
    if qr_payload:
        qr = qrcode.QRCode(box_size=4, border=2)
        qr.add_data(qr_payload)
        qr.make(fit=True)
        qr_matrix = qr.make_image(fill_color="black", back_color="white")
        qr_np = np.array(qr_matrix.convert("RGB"))
        qr_bgr = cv2.cvtColor(qr_np, cv2.COLOR_RGB2BGR)
        qr_resized = cv2.resize(qr_bgr, (qrx2 - qrx1, qry2 - qry1), interpolation=cv2.INTER_NEAREST)
        img[qry1:qry2, qrx1:qrx2] = qr_resized
    else:
        cv2.rectangle(img, (qrx1, qry1), (qrx2, qry2), (255, 255, 255), -1)
        cv2.rectangle(img, (qrx1, qry1), (qrx2, qry2), (0, 0, 0), 2)
        # Finder corners
        for fx, fy in [(qrx1 + 10, qry1 + 10), (qrx2 - 38, qry1 + 10), (qrx1 + 10, qry2 - 38)]:
            cv2.rectangle(img, (fx, fy), (fx + 28, fy + 28), (0, 0, 0), -1)
            cv2.rectangle(img, (fx + 6, fy + 6), (fx + 22, fy + 22), (255, 255, 255), -1)
            cv2.rectangle(img, (fx + 10, fy + 10), (fx + 18, fy + 18), (0, 0, 0), -1)

    # Aadhaar Number in bottom red band
    cv2.putText(img, aadhaar_str, (240, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 20, 20), 3)

    if apply_blur:
        img = cv2.GaussianBlur(img, (35, 35), 15.0)

    return img


# ===========================================================================
# Test 1: Clean Document
# ===========================================================================
def test_clean_aadhaar_screening(tmp_path, pipeline, clean_aadhaar_data):
    """
    Clean document with valid format & Verhoeff checksum must receive:
    - Status: STRUCTURALLY_VALID_UNVERIFIED
    - Legacy verdict equality: res['verdict'] == 'AUTHENTIC'
    - Official verification: NOT_PERFORMED
    - Privacy compliance: Masked numbers in structural_checks
    """
    img = _create_synthetic_aadhaar_image(
        aadhaar_str=clean_aadhaar_data["formatted"],
        name=clean_aadhaar_data["name"],
        dob=clean_aadhaar_data["dob"],
        gender=clean_aadhaar_data["gender"]
    )
    img_path = str(tmp_path / "clean_aadhaar.png")
    cv2.imwrite(img_path, img)

    res = pipeline.screen_document(img_path, document_type="national_id")

    # New schema status
    assert res["status"] in ("STRUCTURALLY_VALID_UNVERIFIED", "AUTHENTIC")
    # Legacy backward compatibility
    assert res["verdict"] == "AUTHENTIC"
    # Official verification must state not performed
    assert res["official_verification"]["status"] == "NOT_PERFORMED"
    assert res["official_verification"]["verified"] is False
    # Checksum is valid
    assert res["structural_checks"]["checksum_valid"] is True
    # Field detector output present
    assert "detected_fields" in res
    assert isinstance(res["detected_fields"], list)
    # Risk level must be LOW
    assert res["risk_level"] == "LOW"


# ===========================================================================
# Test 2: Edited Digit with Invalid Verhoeff Checksum
# ===========================================================================
def test_edited_digit_invalid_checksum(tmp_path, pipeline, clean_aadhaar_data):
    """
    An attacker edits a single digit in the Aadhaar number, breaking Verhoeff.
    System must produce a deterministic failure and status INVALID / FLAGGED.
    """
    orig = clean_aadhaar_data["number"]
    # Change last digit to create invalid checksum
    last_digit = int(orig[-1])
    bad_last_digit = (last_digit + 5) % 10
    bad_number = f"{orig[:-1]}{bad_last_digit}"
    assert validate_verhoeff(bad_number) is False

    bad_formatted = f"{bad_number[:4]} {bad_number[4:8]} {bad_number[8:]}"
    img = _create_synthetic_aadhaar_image(
        aadhaar_str=bad_formatted,
        name=clean_aadhaar_data["name"],
        dob=clean_aadhaar_data["dob"],
        gender=clean_aadhaar_data["gender"]
    )
    img_path = str(tmp_path / "bad_checksum.png")
    cv2.imwrite(img_path, img)

    res = pipeline.screen_document(img_path, document_type="national_id")

    assert res["status"] in ("INVALID", "FLAGGED / TAMPERED", "LIKELY_TAMPERED")
    assert res["verdict"] != "STRUCTURALLY_VALID_UNVERIFIED"
    assert res["structural_checks"]["checksum_valid"] is False
    assert len(res["critical_mismatches"]) > 0 or len(res["tampering_signals"]) > 0


# ===========================================================================
# Test 3: Specific Failure Case: Edited Digit with VALID Verhoeff Checksum (Direction A: Printed Edited, QR Genuine)
# ===========================================================================
def test_edited_digit_valid_checksum_qr_mismatch(tmp_path, pipeline):
    """
    Direction A: Attacker crafts an Aadhaar number that passes Verhoeff checksum (e.g. 9876 5432 1096).
    However, the machine-readable QR payload contains the genuine identity number (2847 5938 1023).
    Both numbers have valid Verhoeff checksums, but they contradict each other.
    Cross-field verification must flag the contradiction and reject the document.
    It must NEVER be marked AUTHENTIC or STRUCTURALLY_VALID_UNVERIFIED.
    """
    # Genuine number in QR (valid Verhoeff)
    gen_base = "28475938102"
    gen_chk = generate_verhoeff_checksum(gen_base)
    genuine_number = f"{gen_base}{gen_chk}"

    # Forged number on visual print (also mathematically valid Verhoeff!)
    forged_base = "98765432109"
    forged_chk = generate_verhoeff_checksum(forged_base)
    forged_number = f"{forged_base}{forged_chk}"

    assert validate_verhoeff(genuine_number) is True, "Genuine number must pass Verhoeff checksum"
    assert validate_verhoeff(forged_number) is True, "Forged number must pass Verhoeff checksum"
    assert forged_number != genuine_number, "Numbers must be distinct"

    forged_formatted = f"{forged_number[:4]} {forged_number[4:8]} {forged_number[8:]}"

    # Generate document with forged printed text and real QR containing genuine number
    qr_payload = f"MOCK-ID:{genuine_number}|NAME:ARJUN MEHTA"
    img = _create_synthetic_aadhaar_image(
        aadhaar_str=forged_formatted,
        name="ARJUN MEHTA",
        dob="15/08/1992",
        qr_payload=qr_payload
    )

    # Pre-test assertion: Verify the generated QR code is physically decodable by cv2.QRCodeDetector
    qr_detector = cv2.QRCodeDetector()
    decoded_txt, _, _ = qr_detector.detectAndDecode(img)
    assert decoded_txt != "", "Test setup failure: OpenCV QR decoder failed to decode generated QR code"
    assert genuine_number in decoded_txt, f"Decoded QR does not contain expected genuine number {genuine_number}"

    img_path = str(tmp_path / "valid_checksum_tampered_printed.png")
    cv2.imwrite(img_path, img)

    res = pipeline.screen_document(img_path, document_type="national_id")

    # Mandatory assertions:
    # 1. qr_mismatch_detected must be explicitly True
    assert res.get("qr_mismatch_detected") is True, "QR cross-field mismatch was not detected"
    # 2. critical_mismatches must be non-empty and document the contradiction
    assert len(res.get("critical_mismatches", [])) > 0, "critical_mismatches list must not be empty"
    assert any("mismatch" in m.lower() or "contradicts" in m.lower() for m in res["critical_mismatches"]), (
        f"Expected cross-field mismatch reason in critical_mismatches, got: {res['critical_mismatches']}"
    )
    # 3. Status must reflect tampering or suspicious, never authentic or structurally valid
    assert res["status"] in ("SUSPICIOUS", "LIKELY_TAMPERED"), f"Unexpected status: {res['status']}"
    assert res["status"] != "STRUCTURALLY_VALID_UNVERIFIED", "Tampered document must not receive STRUCTURALLY_VALID_UNVERIFIED"
    assert res["verdict"] != "AUTHENTIC", "Tampered document must not receive AUTHENTIC verdict"


# ===========================================================================
# Test 3b: QR Payload Changed, Printed Number Unchanged (Direction B: QR Edited, Printed Genuine)
# ===========================================================================
def test_tampered_qr_payload_contradicts_printed_number(tmp_path, pipeline):
    """
    Direction B: The printed text retains the genuine Aadhaar number (2847 5938 1023),
    but the QR code payload contains a different valid Verhoeff number (9876 5432 1096).
    The system must detect the contradiction between the printed number and QR payload.
    """
    gen_base = "28475938102"
    gen_chk = generate_verhoeff_checksum(gen_base)
    genuine_number = f"{gen_base}{gen_chk}"
    genuine_formatted = f"{genuine_number[:4]} {genuine_number[4:8]} {genuine_number[8:]}"

    tampered_base = "98765432109"
    tampered_chk = generate_verhoeff_checksum(tampered_base)
    tampered_number = f"{tampered_base}{tampered_chk}"

    assert validate_verhoeff(genuine_number) is True, "Genuine number must pass Verhoeff checksum"
    assert validate_verhoeff(tampered_number) is True, "Tampered number must pass Verhoeff checksum"
    assert tampered_number != genuine_number, "Numbers must be distinct"

    # Card has genuine printed number, but QR payload has tampered number
    qr_payload = f"MOCK-ID:{tampered_number}|NAME:ARJUN MEHTA"
    img = _create_synthetic_aadhaar_image(
        aadhaar_str=genuine_formatted,
        name="ARJUN MEHTA",
        dob="15/08/1992",
        qr_payload=qr_payload
    )

    # Pre-test assertion: Verify the generated QR code is physically decodable by cv2.QRCodeDetector
    qr_detector = cv2.QRCodeDetector()
    decoded_txt, _, _ = qr_detector.detectAndDecode(img)
    assert decoded_txt != "", "Test setup failure: OpenCV QR decoder failed to decode generated QR code"
    assert tampered_number in decoded_txt, f"Decoded QR does not contain expected tampered number {tampered_number}"

    img_path = str(tmp_path / "tampered_qr_valid_printed.png")
    cv2.imwrite(img_path, img)

    res = pipeline.screen_document(img_path, document_type="national_id")

    # Mandatory assertions:
    # 1. qr_mismatch_detected must be explicitly True
    assert res.get("qr_mismatch_detected") is True, "QR cross-field mismatch was not detected"
    # 2. critical_mismatches must be non-empty and document the contradiction
    assert len(res.get("critical_mismatches", [])) > 0, "critical_mismatches list must not be empty"
    assert any("mismatch" in m.lower() or "contradicts" in m.lower() for m in res["critical_mismatches"]), (
        f"Expected cross-field mismatch reason in critical_mismatches, got: {res['critical_mismatches']}"
    )
    # 3. Status must reflect tampering or suspicious, never authentic or structurally valid
    assert res["status"] in ("SUSPICIOUS", "LIKELY_TAMPERED"), f"Unexpected status: {res['status']}"
    assert res["status"] != "STRUCTURALLY_VALID_UNVERIFIED", "Tampered document must not receive STRUCTURALLY_VALID_UNVERIFIED"
    assert res["verdict"] != "AUTHENTIC", "Tampered document must not receive AUTHENTIC verdict"


# ===========================================================================
# Test 4: Photo / Cut Seam Tampering
# ===========================================================================
def test_photo_replacement_tampering(tmp_path, pipeline):
    """
    Simulates portrait photo replacement with a noticeable visual seam.
    Forensics (ELA / Copy-Move / Typography) should flag anomalies.
    """
    tampered_photo_path = Path("backend/data/mock_aadhaar_01_tampered_photoswap.jpg")
    if not tampered_photo_path.exists():
        tampered_photo_path = Path("data/mock_aadhaar_01_tampered_photoswap.jpg")
    if tampered_photo_path.exists():
        test_path = str(tampered_photo_path)
    else:
        # Generate double-compressed JPEG with spliced photo region
        img = _create_synthetic_aadhaar_image(
            aadhaar_str="2847 5938 1026",
            add_splice=True
        )
        test_path = str(tmp_path / "spliced_photo.jpg")
        cv2.imwrite(test_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

    res = pipeline.screen_document(test_path, benchmark_mode=True, document_type="national_id")

    # Forensic analysis must detect ELA hotspot or anomaly
    ela_info = res.get("signals", {}).get("ela_forensics", {})
    assert ela_info.get("hotspot_count", 0) > 0 or len(res.get("flagged_regions", [])) > 0 or res["risk_score"] > 20
    # Must route to review or flag as suspicious/tampered
    assert res["status"] in ("SUSPICIOUS", "LIKELY_TAMPERED", "MANUAL_REVIEW_REQUIRED") or res["why_this_verdict"]["human_review_required"] is True


# ===========================================================================
# Test 5: Low Quality / Blur Handling
# ===========================================================================
def test_low_quality_blur_handling(tmp_path, pipeline, clean_aadhaar_data):
    """
    Severely blurred document image must not produce an uncritical pass.
    Must route to MANUAL_REVIEW_REQUIRED.
    """
    img = _create_synthetic_aadhaar_image(
        aadhaar_str=clean_aadhaar_data["formatted"],
        apply_blur=True
    )
    img_path = str(tmp_path / "blurred_doc.png")
    cv2.imwrite(img_path, img)

    res = pipeline.screen_document(img_path, document_type="national_id")

    assert res["quality_tier"] in ("LOW", "VERY_LOW") or res["condition"]["is_blurry"] is True
    assert res["status"] in ("MANUAL_REVIEW_REQUIRED", "SUSPICIOUS", "NEEDS REVIEW") or res["why_this_verdict"]["human_review_required"] is True
