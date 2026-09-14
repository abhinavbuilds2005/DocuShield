import os
import sys
import json
import requests

API_URL = "http://127.0.0.1:8008/api/screen"

test_files = [
    ("Genuine Passport", r"D:\original test data\WhatsApp Image 2026-09-10 at 10.26.52 AM.jpeg", "AUTHENTIC"),
    ("Fake Passport", r"D:\fake data\images (2).jpg", "SUSPICIOUS"),
    ("Genuine Aadhaar", r"D:\original test data\WhatsApp Image 2026-09-04 at 10.18.34 AM.jpeg", "AUTHENTIC"),
    ("Fake Aadhaar", r"D:\fake data\Screenshot 2026-09-03 224537.png", "SUSPICIOUS"),
    ("Genuine PAN", r"D:\original test data\WhatsApp Image 2026-09-06 at 3.40.00 PM (1).jpeg", "AUTHENTIC"),
    ("Fake PAN", r"D:\fake data\Screenshot 2026-09-03 225315.png", "SUSPICIOUS"),
    ("Genuine DL", r"D:\original test data\WhatsApp Image 2026-09-06 at 3.40.00 PM (2).jpeg", "AUTHENTIC"),
    ("Fake DL", r"D:\fake data\ASSAM_1zVZW8a.jpg", "FLAGGED / TAMPERED"),
]

print("="*75)
print("DOCUSHIELD AI — WEB API VERIFICATION (8 TARGET DOCUMENTS)")
print("="*75)

all_passed = True

for label, file_path, expected_verdict in test_files:
    if not os.path.exists(file_path):
        print(f"[FAIL] File not found: {file_path}")
        all_passed = False
        continue

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "image/jpeg" if file_path.lower().endswith((".jpg", ".jpeg")) else "image/png")}
        resp = requests.post(API_URL, files=files)
    
    if resp.status_code != 200:
        print(f"[ERROR] HTTP {resp.status_code} for {label}: {resp.text}")
        all_passed = False
        continue

    data = resp.json()
    verdict = data.get("verdict")
    auth_score = data.get("authenticity_score")
    risk_score = data.get("risk_score")
    doc_type = data.get("document_type")
    diag_status = data.get("diagnostic_status")

    # In DocuShield AI:
    # AUTHENTIC (risk <= 20)
    # SUSPICIOUS / FLAGGED / TAMPERED (risk > 20)
    matches_expected = (verdict == expected_verdict) or (expected_verdict in ["SUSPICIOUS", "FLAGGED / TAMPERED"] and verdict in ["SUSPICIOUS", "FLAGGED / TAMPERED"])

    status_tag = "PASS" if matches_expected else "FAIL"
    if not matches_expected:
        all_passed = False

    print(f"[{status_tag}] {label:18} | File: {os.path.basename(file_path)[:25]:25} | Verdict: {verdict:18} | Auth: {auth_score:5.1f}% | Risk: {risk_score:5.1f}% | Type: {doc_type:10}")

print("="*75)
print(f"WEB API VERIFICATION RESULT: {'ALL 8 CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print("="*75)
