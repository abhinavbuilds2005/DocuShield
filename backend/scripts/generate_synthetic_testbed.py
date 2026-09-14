"""
Synthetic Testbed Generator for SIH26188
Generates reproducible synthetic prototype test samples across 5 document categories:
1. Passport (ICAO Doc 9303 TD3 standard with valid and invalid MRZ)
2. Visa (Single/Multiple entries, stay duration)
3. National ID (12-digit Indian UID with Verhoeff, Tax ID PAN)
4. Driving Licence (State codes, vehicle categories)
5. Permit / Travel Authorization (Authorization number, dates)

CRITICAL NOTE:
All samples are explicitly watermarked and labeled:
"SYNTHETIC TEST SAMPLE — NOT A REAL GOVERNMENT DOCUMENT"
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")

# Subdirectories for each category
SUBDIRS = [
    "raw", "authentic", "tampered", "passport", "visa",
    "national_id", "driving_license", "permit", "processed", "annotations"
]

for sub in SUBDIRS:
    os.makedirs(os.path.join(DATASETS_DIR, sub), exist_ok=True)


def create_synthetic_card(
    title: str,
    fields: list,
    output_path: str,
    bg_color=(240, 245, 250),
    is_tampered=False,
    tamper_note=""
):
    """Creates a synthetic credential image with watermarking."""
    w, h = 800, 500
    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Outer security border
    draw.rectangle([10, 10, w - 10, h - 10], outline=(60, 90, 140), width=3)
    draw.rectangle([16, 16, w - 16, h - 16], outline=(180, 200, 220), width=1)

    # Header bar
    draw.rectangle([20, 20, w - 20, 75], fill=(30, 58, 100))
    draw.text((35, 32), title.upper(), fill=(255, 255, 255))

    # Synthetic disclaimer watermark
    disclaimer = "SYNTHETIC TEST SAMPLE — NOT A REAL GOVERNMENT DOCUMENT"
    draw.text((35, h - 35), disclaimer, fill=(180, 40, 40))

    # Mock portrait box
    draw.rectangle([40, 100, 180, 260], fill=(210, 220, 230), outline=(100, 120, 140), width=2)
    # Simple avatar graphic
    draw.ellipse([80, 130, 140, 190], fill=(160, 175, 195))
    draw.chord([60, 180, 160, 270], 180, 360, fill=(120, 140, 170))
    draw.text((65, 270), "PORTRAIT", fill=(100, 120, 140))

    # Text fields
    y = 105
    for label, val in fields:
        draw.text((210, y), f"{label}:", fill=(90, 105, 120))
        draw.text((350, y), str(val), fill=(20, 25, 35))
        y += 36

    if is_tampered and tamper_note:
        # Subtle manipulation artifact simulation
        draw.rectangle([345, 102, 550, 138], outline=(230, 80, 80), width=1)
        draw.text((w - 280, 85), f"[MANIPULATION: {tamper_note}]", fill=(200, 40, 40))

    img.save(output_path, quality=92)


def generate_all_samples():
    """Generates authentic and tampered samples across 5 document types."""
    manifest = {}

    # 1. PASSPORT (Authentic & Tampered)
    p_auth = os.path.join(DATASETS_DIR, "passport", "passport_authentic_sample.png")
    p_fields = [
        ("DOCUMENT TYPE", "PASSPORT (P)"),
        ("COUNTRY", "REPUBLIC OF UTOPIA (UTO)"),
        ("PASSPORT NO", "U12345678"),
        ("FULL NAME", "CITIZEN JOHN DOE"),
        ("NATIONALITY", "UTOPIAN"),
        ("DATE OF BIRTH", "15/08/1988"),
        ("EXPIRY DATE", "14/08/2032"),
        ("MRZ LINE 1", "P<UTODOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"),
        ("MRZ LINE 2", "U123456784UTO8808154M3208148<<<<<<<<<<<<<<<4")
    ]
    create_synthetic_card("REPUBLIC OF UTOPIA — PASSPORT", p_fields, p_auth)
    manifest["passport_authentic_sample.png"] = {"type": "passport", "label": "GENUINE"}

    p_tamp = os.path.join(DATASETS_DIR, "passport", "passport_tampered_sample.png")
    p_tamp_fields = [
        ("DOCUMENT TYPE", "PASSPORT (P)"),
        ("COUNTRY", "REPUBLIC OF UTOPIA (UTO)"),
        ("PASSPORT NO", "U99999999"),  # Altered passport number
        ("FULL NAME", "CITIZEN JOHN DOE"),
        ("NATIONALITY", "UTOPIAN"),
        ("DATE OF BIRTH", "15/08/1988"),
        ("EXPIRY DATE", "14/08/2032"),
        ("MRZ LINE 1", "P<UTODOE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"),
        ("MRZ LINE 2", "U123456784UTO8808154M3208148<<<<<<<<<<<<<<<4")  # MRZ still has old number -> mismatch!
    ]
    create_synthetic_card("REPUBLIC OF UTOPIA — PASSPORT", p_tamp_fields, p_tamp, is_tampered=True, tamper_note="MRZ Number Mismatch")
    manifest["passport_tampered_sample.png"] = {"type": "passport", "label": "TAMPERED", "tamper": "MRZ Mismatch"}

    # 2. VISA
    v_auth = os.path.join(DATASETS_DIR, "visa", "visa_authentic_sample.png")
    v_fields = [
        ("VISA NUMBER", "V87654321"),
        ("HOLDER NAME", "JANE SMITH"),
        ("NATIONALITY", "UTOPIAN"),
        ("VISA TYPE", "TOURIST (T-1)"),
        ("ENTRIES", "MULTIPLE"),
        ("STAY DURATION", "90 DAYS"),
        ("ISSUE DATE", "01/01/2025"),
        ("EXPIRY DATE", "31/12/2025"),
        ("AUTHORITY", "CONSULATE GENERAL")
    ]
    create_synthetic_card("ENTRY VISA — STATE OF PACIFICA", v_fields, v_auth)
    manifest["visa_authentic_sample.png"] = {"type": "visa", "label": "GENUINE"}

    # 3. NATIONAL ID
    n_auth = os.path.join(DATASETS_DIR, "national_id", "national_id_sample.png")
    n_fields = [
        ("NATIONAL ID NO", "9876 5432 1098"),
        ("NAME", "ALEXANDER CHEN"),
        ("DATE OF BIRTH", "22/11/1992"),
        ("GENDER", "MALE"),
        ("ISSUE DATE", "10/05/2018"),
        ("AUTHORITY", "NATIONAL IDENTITY COMMISSION")
    ]
    create_synthetic_card("NATIONAL IDENTITY CARD", n_fields, n_auth)
    manifest["national_id_sample.png"] = {"type": "national_id", "label": "GENUINE"}

    # 4. DRIVING LICENCE
    d_auth = os.path.join(DATASETS_DIR, "driving_license", "driving_license_sample.png")
    d_fields = [
        ("LICENCE NO", "DL-0420110056789"),
        ("HOLDER NAME", "PRIYA SHARMA"),
        ("DATE OF BIRTH", "04/04/1990"),
        ("VEHICLE CLASS", "LMV / MCWG"),
        ("ISSUE DATE", "12/03/2015"),
        ("EXPIRY DATE", "11/03/2035"),
        ("AUTHORITY", "REGIONAL TRANSPORT OFFICE")
    ]
    create_synthetic_card("UNION DRIVING LICENCE", d_fields, d_auth)
    manifest["driving_license_sample.png"] = {"type": "driving_license", "label": "GENUINE"}

    # 5. PERMIT / TRAVEL AUTHORIZATION
    perm_auth = os.path.join(DATASETS_DIR, "permit", "travel_permit_sample.png")
    perm_fields = [
        ("AUTHORIZATION NO", "TA-2026-98124"),
        ("NAME", "ROBERTO SILVA"),
        ("NATIONALITY", "BRAZILIAN"),
        ("PERMIT TYPE", "TRANSIT & WORK AUTHORIZATION"),
        ("ISSUE DATE", "15/02/2025"),
        ("EXPIRY DATE", "14/08/2026"),
        ("AUTHORITY", "IMMIGRATION & BORDER CONTROL")
    ]
    create_synthetic_card("TRAVEL & BORDER AUTHORIZATION PERMIT", perm_fields, perm_auth)
    manifest["travel_permit_sample.png"] = {"type": "permit", "label": "GENUINE"}

    # Save manifest in annotations
    ann_path = os.path.join(DATASETS_DIR, "annotations", "synthetic_testbed_manifest.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[TESTBED] Successfully generated {len(manifest)} synthetic test samples in datasets/")


if __name__ == "__main__":
    generate_all_samples()
