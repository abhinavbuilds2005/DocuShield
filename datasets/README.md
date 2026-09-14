# DocuShield Dataset Interface & Synthetic Testbed Specification
**SIH 2026 Problem Statement: SIH26188 — AI-Based Fake Identity & Document Screening System**

---

## 1. Directory Structure

```
datasets/
├── raw/                 # Unprocessed raw test images
├── authentic/           # Authentic baseline samples across 5 document types
├── tampered/            # Controlled manipulated samples across 5 document types
├── passport/            # Sample passport bio pages (ICAO TD3 format)
├── visa/                # Sample visas with stay and entry metadata
├── national_id/         # Sample national ID cards (Aadhaar, PAN, national citizen IDs)
├── driving_license/     # Sample driving licenses (state codes, vehicle categories)
├── permit/              # Sample permits & travel authorizations
├── processed/           # Normalized, cropped, and preprocessed samples
├── annotations/         # Ground truth metadata and forgery annotations
└── README.md            # This documentation file
```

---

## 2. Dataset Ethical Guidelines & Data Privacy Statement

> [!IMPORTANT]
> **Strict Cybersecurity & Privacy Policy**:
> 1. In compliance with data protection laws (such as GDPR and the Digital Personal Data Protection Act), **no real-world personal identity documents, live Aadhaar cards, real passports, or un-redacted citizen credentials are stored in this repository**.
> 2. All samples generated or stored within `datasets/` for developer testing are **strictly synthetic or simulated prototypes** generated for algorithmic verification.
> 3. DO NOT commit, distribute, or log real citizen PII.

---

## 3. Synthetic Manipulation Taxonomy

For testing forensic detectors in controlled laboratory environments, synthetic samples undergo controlled manipulations:

| Manipulation Type | Target Detector | Controlled Modification Technique |
| :--- | :--- | :--- |
| **Text Replacement** | Typography / OCR Consistency | Splicing new alphanumeric text onto card body |
| **Date Inversion** | Field Validation / Chronology | Setting Expiry date < Issue date or post-dating |
| **Photo Swap** | ELA Cut Seam / Face Verifier | Splicing alternate portrait into document frame |
| **Cloned Seal / Stamp** | Copy-Move (ORB+RANSAC) | Duplicating an emblem, watermark, or seal patch |
| **JPEG Recompression** | Condition Analyzer / ELA | Compressing image at quality 50-65 to simulate web transfer |
| **MRZ Alteration** | MRZ Parser Check Digits | Modifying document number without updating ICAO 7-3-1 check digit |

---

## 4. Controlled Testbed Generator

To generate reproducible synthetic prototype test samples across all 5 document categories, run:

```bash
python backend/scripts/generate_synthetic_testbed.py
```

All generated files are clearly stamped with:
`[SYNTHETIC TEST SAMPLE — NOT A REAL GOVERNMENT DOCUMENT]`
