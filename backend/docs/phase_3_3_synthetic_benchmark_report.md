# Phase 3.3A: Synthetic Benchmark Generation & Validation Report

## 1. Executive Summary & Status

Phase 3.3A (Synthetic Benchmark Generation and Validation) has been completed under strict privacy, safety, and non-disclosure constraints:
- **500 Synthetic Evaluation Samples Generated**: Exactly 100 samples across 5 document classes (Aadhaar, PAN, Driving Licence, Passport, Voter ID).
- **100% Fictional Data**: All personae, identifiers, and layouts are synthetic mockups with zero real citizen information.
- **Mandatory Disclaimer Watermark**: Every sample carries the permanent visual watermark:  
  `[SYNTHETIC TEST SAMPLE — NOT A REAL GOVERNMENT DOCUMENT]`
- **Locked Test Partition Frozen**: The 150 locked test samples remain **completely unevaluated** and frozen under a cryptographic SHA-256 manifest.
- **Evaluation Harness Validated**: The standalone benchmark evaluator was verified on the DEV/VAL partitions only.
- **Production Integrity Preserved**: Zero model weights retrained or modified (all model SHA-256 hashes remain unchanged); 219/219 backend regression tests passing.

---

## 2. Files Created & Modified

### A. Created Files
1. `backend/scripts/generate_synthetic_evaluation_dataset.py`:
   - Deterministic generation engine utilizing master seed `20260918`.
   - Generates fictional personae, valid/invalid checksums, and renders realistic card mockups across 5 degradation profiles.
   - Automatically writes images (`.jpg`), YOLO bounding boxes (`.txt`), and evaluation schemas (`.eval.json`).
2. `backend/scripts/validate_synthetic_evaluation_dataset.py`:
   - Validates all 500 records against split quotas, coordinates, mandatory fields, and negative controls.
   - Performs exact duplicate scans (MD5/SHA-256) and perceptual near-duplicate scans (dHash).
   - Verifies zero cross-split lineage leakage.
   - Generates immutable manifests: `test_manifest_sha256.json` and `benchmark_manifest.json`.
3. `backend/scripts/evaluate_synthetic_benchmark.py`:
   - Isolated evaluation harness computing classification accuracy, confusion matrices, field detection IoU/P/R, OCR Exact Match Ratio (EMR), Character Error Rate (CER), Levenshtein similarity, latency, and peak memory.
   - Features a hard permission gate that strictly prohibits locked test execution unless `--allow-locked-test` is explicitly passed.
4. `data/evaluation_synthetic/`:
   - `images/`: 500 images partitioned into `dev/` (300), `val/` (50), `test/` (150).
   - `annotations/`: 500 YOLO label files partitioned into `dev/`, `val/`, `test/`.
   - `metadata/`: 500 companion `.eval.json` records, plus `test_manifest_sha256.json` and `benchmark_manifest.json`.
5. `reports/evaluation_synthetic_qa/`:
   - Contact sheet manifests and 15 annotated bounding-box visual review artifacts from DEV and VAL only.

### B. Modified Files
- `backend/docs/phase_3_3_synthetic_benchmark_report.md`: Updated with full empirical validation results.
- **Zero production code or model files were modified.**

---

## 3. Dataset Distribution & Counts

### A. Document Class Distribution (100 per class = 500 total)
| Document Class | Development (60%) | Validation (10%) | Locked Test (30%) | Total Samples |
| :--- | :---: | :---: | :---: | :---: |
| **Aadhaar** | 60 | 10 | 30 | 100 |
| **PAN Card** | 60 | 10 | 30 | 100 |
| **Driving Licence** | 60 | 10 | 30 | 100 |
| **Passport** | 60 | 10 | 30 | 100 |
| **Voter ID (EPIC)** | 60 | 10 | 30 | 100 |
| **Total** | **300** | **50** | **150** | **500** |

### B. Acquisition & Degradation Category Distribution
| Degradation Category | Dev (300) | Val (50) | Test (150) | Total Corpus | Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pristine Flatbed Scan** | 90 | 15 | 45 | 150 | 30.0% |
| **Mobile Camera Simulation** | 75 | 12 | 38 | 125 | 25.0% |
| **Compressed Transmission** | 60 | 10 | 30 | 100 | 20.0% |
| **Degraded Physical Card** | 45 | 8 | 22 | 75 | 15.0% |
| **Negative Control / Edge Case** | 30 | 5 | 15 | 50 | 10.0% |
| **Total** | **300** | **50** | **150** | **500** | **100.0%** |

---

## 4. Dataset Validation & Integrity Findings

The automated validation suite (`validate_synthetic_evaluation_dataset.py`) executed across all 500 samples with the following results:

1. **File Existence & Split Quotas**:
   - Exactly 500 image files verified (`.jpg`).
   - Exactly 500 YOLO annotation files verified (`.txt`).
   - Exactly 500 metadata records verified (`.eval.json`).
   - Split distribution matches exactly: 300 DEV, 50 VAL, 150 TEST.
2. **Schema & Field Integrity**:
   - `synthetic: true` verified on 100% of samples.
   - `locked_test: true` verified on 100% of TEST split samples.
   - `watermark_present: true` verified on 100% of samples.
   - Zero coordinate out-of-bounds errors: all bounding boxes clamped within pixel bounds and normalized YOLO coordinates in $[0.0, 1.0]$.
   - Negative controls verified: 50/50 samples correctly tagged with anomaly categories (invalid checksums, missing photo/ID, typography splicing).
3. **Deduplication & Cross-Split Leakage Results**:
   - **Exact Image Duplicates**: **0 found** (MD5 & SHA-256).
   - **Exact Metadata Duplicates**: **0 found**.
   - **Cross-Split Lineage Leakage**: **0 found**. Every base document and its derived variants exist exclusively within one split.
   - **Perceptual Near-Duplicates**: No cross-split leakage observed; all derived mobile/compressed variations share unique lineage IDs isolated from other splits.

---

## 5. Manual Visual QA Findings (DEV & VAL Only)

Visual inspection of 15 annotated review samples in `reports/evaluation_synthetic_qa/` confirmed:
1. **Perimeter Watermark**: Clear, unclipped, and legible on every sample across all document types.
2. **Bounding-Box Alignment**: YOLO bounding boxes tightly encompass target text lines, portrait boxes, and emblem headers.
3. **Typography & Readability**: Pristine and mobile camera samples exhibit sharp, high-contrast characters suitable for OCR; compressed samples display realistic JPEG blocking; degraded samples display authentic Gaussian blur.
4. **Negative Controls**: Bounding boxes appropriately omit missing fields on negative control samples, and anomalous text is properly bounded for forensic inspection.
5. **No TEST Contamination**: Routine QA was strictly confined to DEV and VAL samples. Zero locked TEST samples were exposed.

---

## 6. DEV / VAL Evaluator Smoke-Test Results

A smoke test of `evaluate_synthetic_benchmark.py` was conducted on a 15-sample subset of the VAL split to confirm evaluator functionality:
- **Document Classification Accuracy**: **100.0%** (15/15 correctly classified).
- **OCR Exact Match Ratio (EMR)**: **69.23%** (45/65 fields matched exactly across compressed transmission images).
- **Mean Character Error Rate (CER)**: **0.2774**.
- **Mean Levenshtein Similarity**: **0.7256**.
- **Peak Memory**: **176.4 MB** (safely compliant with the $< 512\,\text{MB}$ runtime ceiling).
- **Mean Latency**: $\approx 11.3\,\text{seconds}$ per document (comprehensive full-pipeline forensic analysis on CPU).
- **Screening Verdicts**: Correctly classified as `STRUCTURALLY_VALID_UNVERIFIED` for clean valid cards.
- **Locked Test Protection**: Verified that attempting to run on `test` without `--allow-locked-test` triggers an immediate `PermissionError`.

---

## 7. Model Integrity & Regression Suite Verification

1. **Model Weights Integrity**:
   - `models/aadhaar_field_detector.pt`: `D2ACF50D2935DCEA1524ACEE8E003EC7C91434236799CD1AF96F6DBC622A007E`
   - `models/dl_field_detector.pt`: `A62F354581F0C1066D7B0657ED1897C67338B369876EBA2B83A039C6A52EB06E`
   - `models/pan_field_detector.pt`: `472FBA9557096BC3155E45CF39B6E6A0936D5394D8D2E905E6F9B75CE181931C`
   *(All SHA-256 hashes are 100% identical to baseline — zero model weights modified).*
2. **Backend Regression Test Suite**:
   - **219 passed, 0 failed** in 345.46s (`python -m pytest backend/tests/ -q`).

---

## 8. Benchmark Manifest Verification

The locked test partition is frozen under:
- **Test Manifest**: `data/evaluation_synthetic/metadata/test_manifest_sha256.json`
- **Benchmark Manifest**: `data/evaluation_synthetic/metadata/benchmark_manifest.json`
- **Test Manifest SHA-256**: `68f014e6c8bc851a1fbd0d5049540aca9260d68a0c4b732e9951e485eefdd5a5`

The 150 locked test samples remain **completely untouched, uninspected, and unevaluated**.

---

## 9. Privacy, Legal & Ethical Observations

1. **Zero Real Citizen Data**: Verified that all names, numbers, and addresses were procedurally generated from fictional lookup dictionaries and math checksum tables.
2. **HMAC Key Protection**: `DOCUSHIELD_EVAL_KEY` is loaded strictly from environment variables; zero secret keys are stored in files or printed to logs.
3. **No Authenticity Claims**: All outputs and documentation preserve the strict principle that structural/checksum validity is not official authenticity.

---

## 10. Phase 1 vs. Phase 2 Comparison Note

As required by Task G, because an exact historical snapshot of the pre-Phase 2 codebase cannot be executed without rolling back working directory state, the harness reports:
`"Phase 1 comparative benchmark unavailable (reproducible versioned snapshot required)"`
and benchmarks the current Phase 2 pipeline cleanly without inventing historical numbers.

---

## 11. Final Status

> ### **READY FOR LOCKED-TEST EVALUATION**
>
> All generation, validation, near-duplicate analysis, lineage isolation, visual QA, manifest freezing, evaluator smoke testing, and regression suite verifications have passed without error.
>
> The locked test partition is frozen and ready for the one-time final benchmark run upon your explicit approval.
