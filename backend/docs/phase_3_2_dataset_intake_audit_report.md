# Phase 3.2: Approved Dataset Intake Audit Report

## 1. Executive Summary & Approval Verdict

An intake validation audit was performed against all candidate datasets available within the workspace (`data/` and `datasets/`) in accordance with the requirements established in [`backend/docs/independent_dataset_intake_specification.md`](file:///d:/Document%20detector/backend/docs/independent_dataset_intake_specification.md).

### Approval Verdict
> [!WARNING]
> **VERDICT: REJECTED for Independent OCR Accuracy Evaluation**  
> **VERDICT: CONDITIONALLY APPROVED ONLY for YOLO Bounding-Box Detection Check**
>
> **Core Deficiencies Preventing Full Approval:**
> 1. **Zero Ground-Truth OCR Text**: None of the candidate datasets provide paired ground-truth text strings (no true names, identity numbers, DOBs, or salted HMAC-SHA256 hashes). OCR Exact Match Ratio (EMR) and Character Error Rate (CER) cannot be calculated.
> 2. **Missing Metadata Records**: No companion `.eval.json` files exist; none of the samples define acquisition categories, compression status, or rotation angles.
> 3. **Unverified Individual Consent**: Datasets sourced from public Roboflow exports contain images of real Indian citizens without documented opt-in consent under DPDPA 2023.
> 4. **Distribution Leakage**: The test splits in `data/` were drawn from the identical source pools used to train the YOLO field detectors and do not constitute an independent benchmark.
> 5. **Insufficient Scale**: Only 19 total test-split images exist across the three trained classes (Aadhaar: 15, PAN: 0, DL: 4, Voter ID: 0, Passport: 0), falling far short of the 500-document statistical threshold.

---

## 2. Candidate Dataset Audit & Provenance

### A. Evaluated Candidate Repositories

| Candidate Path | Declared Provenance | License | Document Types | Primary Annotation Format |
| :--- | :--- | :--- | :--- | :--- |
| `data/aadhaar_field_detection/` | Roboflow Universe (`cutm-iwh4a/aadhaar-card-details`) | Public web upload | Aadhaar | YOLO Bounding Boxes (7 classes) |
| `data/processed/pan_field_detection/` | Roboflow Universe (`smartxtract/pan-card-ygz7o`) | CC BY 4.0 | PAN Card | YOLO Bounding Boxes (5 classes) |
| `data/processed/dl_field_detection/` | Roboflow Universe (`autodoc-kkdka/indian-driving-licence-reader`) | CC BY 4.0 | Driving Licence | YOLO Bounding Boxes (6 classes) |
| `data/acceptance_staging/` | Internal PAN acceptance suite | Internal | PAN Card | None (Image-only staging) |
| `datasets/` | Synthetic prototype testbed | Internal demo | Passport, Visa, DL, National ID | Minimal manifest (`synthetic_testbed_manifest.json`) |
| `backend/data/` | Internal mock regression cards | Internal testbed | Aadhaar, PAN, DL | Functional test metadata (`ground_truth.json`) |

---

## 3. Detailed Document Counts

An audit of image assets across all candidate directories produced the following exact counts:

```
Candidate Asset Distribution:
├── data/aadhaar_field_detection/
│   ├── train/images: 105 samples
│   ├── valid/images: 30 samples
│   └── test/images:  15 samples (150 total)
├── data/processed/pan_field_detection/
│   ├── train/images: 107 samples
│   └── valid/images: 44 samples (151 total, NO test split)
├── data/processed/dl_field_detection/
│   ├── train/images: 28 samples
│   ├── valid/images: 8 samples
│   └── test/images:  4 samples (40 total)
├── data/acceptance_staging/
│   └── root:         10 samples (acceptance test cases TC-01 to TC-10)
├── datasets/ (prototype testbed)
│   ├── driving_license: 1 sample
│   ├── national_id:     1 sample
│   ├── passport:        2 samples
│   ├── permit:          1 sample
│   └── visa:            1 sample (6 total)
└── backend/data/ (internal regression suite)
    ├── root:         21 mock cards
    └── robustness:   11 perturbation variants (32 total)
```

### Breakdown by Independent Test Partition Availability

| Document Class | Test-Split Count | Ground-Truth Text Available? | Bounding Boxes Available? | Meets Intake Spec? |
| :--- | :---: | :---: | :---: | :---: |
| **Aadhaar** | 15 | **NO** | YES (YOLO `.txt`) | **NO** |
| **PAN Card** | 0 | **NO** | YES (train/valid only) | **NO** |
| **Driving Licence** | 4 | **NO** | YES (YOLO `.txt`) | **NO** |
| **Voter ID** | 0 | **NO** | **NO** | **NO** |
| **Passport** | 0 | **NO** | **NO** | **NO** |
| **Total Test Samples** | **19** | **0** | **19** | **REJECTED** |

---

## 4. Privacy, Legal & Consent Findings

1. **Unconsented Citizen PII Risk**:
   - Inspection of raw images in `data/raw/pan_card/` and `data/raw/driving_licence/` confirmed they are camera photographs and flatbed scans of genuine Indian identity cards uploaded to public Roboflow repositories by third-party users.
   - While Roboflow exports attach a CC BY 4.0 license, the underlying subjects did not provide documented, legally verifiable consent under India's Digital Personal Data Protection Act (DPDPA 2023).
   - Ingesting and publishing benchmarks on unconsented real identity cards creates significant legal exposure.
2. **PII Masking Compliance in Repo**:
   - The repository code strictly enforces runtime PII masking at the FastAPI boundary (`sanitize_screening_response`).
   - However, storing unhashed, unredacted citizen cards on disk for long-term evaluation violates the [Zero Raw PII Storage Policy](file:///d:/Document%20detector/backend/docs/independent_dataset_intake_specification.md#5-ground-truth-schema--zero-raw-pii-storage-policy).

---

## 5. Annotation Completeness & Data Quality

1. **OCR Text Ground Truth**:
   - **Completely Missing**: Zero text transcription files exist for any image in `data/aadhaar_field_detection/`, `data/processed/pan_field_detection/`, or `data/processed/dl_field_detection/`.
   - YOLO bounding box annotations only record normalized geometry `[class_id, x_center, y_center, width, height]`.
   - Therefore, no ground-truth text exists against which to measure OCR Exact Match, Character Error Rate (CER), or Levenshtein distance.
2. **Negative Controls**:
   - Neither the Aadhaar, PAN, nor DL splits contain designated negative controls (e.g., blank cards, utility bills, business cards).
3. **Bounding Box Quality**:
   - Aadhaar: 7 classes annotated cleanly (Aadhaar number, Name, DOB, Gender, Photo, QR code, Emblem header).
   - PAN: 5 classes annotated (PAN number, Name, Father's Name, DOB, Photo).
   - DL: 6 classes annotated (Licence number, Name, DOB, Issue Date, Expiry Date, Photo).
   - YOLO boxes are intact and suitable for detection mAP testing, but cannot support text extraction evaluation.

---

## 6. Duplicate & Cross-Split Leakage Scan

1. **Exact Duplicate Image Scan**:
   - An MD5 cryptographic hash scan across all 389 image files in `data/` and `datasets/` revealed **zero exact byte-level duplicates** across splits.
2. **Near-Duplicate & Distribution Leakage**:
   - The test sets (`aadhaar_field_detection/test` and `dl_field_detection/test`) were partitioned directly from the same Roboflow project sources as their corresponding training sets.
   - Because they share identical camera sensors, backgrounds, and acquisition environments as the training sets, evaluating on them would measure in-distribution memorization rather than independent generalizability.

---

## 7. Approval Checklist Audit

| Item | Requirement | Status | Audit Note |
| :---: | :--- | :---: | :--- |
| **1** | **License or Consent Verified** | **FAIL** | CC BY 4.0 on Roboflow, but individual citizen consent unverified. |
| **2** | **Privacy Review Completed** | **FAIL** | Real identity cards present without individual consent agreements. |
| **3** | **Zero Raw PII Storage Approved** | **FAIL** | No salted HMAC-SHA256 schema implemented for candidate files. |
| **4** | **Annotation Quality Checked** | **PARTIAL** | YOLO boxes valid; OCR ground-truth text completely missing. |
| **5** | **Deduplication Scan Completed** | **PASS** | Zero exact duplicates found across splits. |
| **6** | **Dataset Split Finalized** | **FAIL** | PAN has 0 test images; DL has only 4 test images; cross-split source overlap exists. |
| **7** | **Evaluation Set Locked** | **FAIL** | Test sets not locked via cryptographic manifest. |

---

## 8. Summary of Risks and Limitations

1. **Risk of False Validation**: Running OCR evaluation against images without ground-truth text would require relying on heuristic regex validation, which verifies structural syntax rather than extraction correctness.
2. **Legal Non-Compliance**: Benchmarking against unconsented third-party identity images risks regulatory non-compliance under DPDPA 2023.
3. **Incomplete Scope**: Voter ID and Passport categories have zero real evaluation samples in the candidate splits.

---

## 9. Next Steps Requiring User Approval

To advance to quantitative evaluation, one of the following two approved pathways must be selected:

- **Pathway A (Recommended — High-Fidelity Synthetic Evaluation Testbed)**:
  Generate an independent, legally unencumbered evaluation corpus of 500 documents across all 5 classes using the synthetic generator framework. Every sample will be paired with the compliant HMAC-SHA256 schema from [`evaluation_sample_schema.json`](file:///d:/Document%20detector/backend/docs/examples/evaluation_sample_schema.json), providing 100% ground-truth text, bounding boxes, zero PII exposure, and full coverage of real-world degradations (blur, compression, rotation).
- **Pathway B (Consented Real-World Private Intake)**:
  Ingest an externally provided archive of explicitly consented participant scans accompanied by signed consent records and pre-hashed metadata.

*Execution is halted. Awaiting explicit user direction and approval before proceeding.*
