# DocuShield AI — Accuracy Improvement Audit Report
**SIH 2026 Problem Statement: SIH26188 (AI-Based Identity Document Forensic Screening)**  
**Date:** September 2026  
**Auditor:** Antigravity AI  
**Scope:** Phase 1 Read-Only Codebase Inspection & Accuracy Bottleneck Analysis

---

## Executive Summary

A comprehensive, read-only architectural audit of DocuShield AI was conducted to identify real-world accuracy bottlenecks in field extraction and forensic screening without compromising system stability or security guardrails.

All **197/197 backend tests** are currently passing, the React/Vite frontend builds cleanly with zero errors, and model SHA-256 weight integrity is verified.

This audit details:
1. The end-to-end execution pipeline from upload to frontend rendering.
2. The exact architectural and algorithmic bottlenecks identified in the source code.
3. Safe, modular improvement opportunities for Phase 2.
4. Risky changes that must be strictly avoided.
5. Exact files requiring modification.
6. A proposed testing matrix.
7. Expected measurable impacts.
8. Non-code-change operational recommendations.

---

## 1. Existing Pipeline Summary

The DocuShield AI screening workflow spans the following end-to-end stages:

```
Image Upload (FastAPI /api/screen)
       │
       ▼
Security & Ingestion Validation
  - Size check (≤ 10 MB)
  - Magic byte verification (JPEG, PNG, WebP)
  - PIL decompression bomb check (≤ 67M px, max dim ≤ 8192px)
  - Downscaled to max 1280px (Lanczos) for 512MB RAM safety
       │
       ▼
Condition Assessment (`DocumentConditionAnalyzer.analyze`)
  - Laplacian variance blur detection (< 120.0 indicates blur)
  - JPEG compression & DCT grid artifact evaluation
  - Effective resolution & quality tier assignment (HIGH, MEDIUM, LOW, VERY_LOW)
       │
       ▼
Field Detection (YOLOv8n / Vision Layer)
  - Route by explicit user selection or document type:
      * Aadhaar: `AadhaarFieldDetector` (threshold: 0.35)
      * PAN: `PANFieldDetector` (threshold: 0.30) with layout heuristic fallback
      * DL: `DLFieldDetector` (threshold: 0.50, experimental) with Sarathi layout fallback
  - Extracts raw bounding boxes: `crop = img[y1:y2, x1:x2]`
       │
       ▼
OCR Text Extraction (`OCREngine.process_image`)
  - Primary: EasyOCR singleton (lazy-loaded, downscaled dynamically to max 960px)
  - Fallback: Tesseract OCR (where available)
  - Word tokens: Approximated by linearly dividing line bounding boxes (`bw // len(words)`)
       │
       ▼
Crop-Assisted Field OCR (`fusion.py`)
  - Ad-hoc regex search on raw crops (`aadhaar_number`, `pan_number`, `licence_number`)
  - Appends matched ID number string to `full_text` if not already present
       │
       ▼
Document Classification (`DocumentClassifier`)
  - Evaluates MRZ ICAO patterns (`P<`, `V<`)
  - Keyword and phrase heuristic matching across 8 document types
  - Post-classification detector re-routing for auto-detected PAN and DL
       │
       ▼
NLP Field & Rule Validation (`DocumentFieldValidator`)
  - Aadhaar: Verhoeff checksum algorithm with optical tolerance for crimson 6 ↔ 8
  - PAN: 10-char alphanumeric pattern (`[A-Z]{5}[0-9]{4}[A-Z]`), 4th char entity check, silent 10th char digit→letter substitution
  - Driving Licence: `^[A-Z]{2}[0-9]{11,14}$` regex + 2-letter state code + chronological state machine
  - Passport: ICAO 9303 MRZ parsing with 7-3-1 weight check-digit verification
  - Chronological consistency: Issue date, expiry date, holder DOB
       │
       ▼
Image Forensics Layer
  - Error Level Analysis (`ErrorLevelAnalysis`): JPEG recompression artifacts and boundary cut seams
  - Copy-Move Detection (`CopyMoveDetector`): SIFT/ORB keypoints + patch cross-correlation
  - Typography & Font Discrepancy (`FontAlignmentForensics`): Edge continuity and token bounding box alignment
  - Metadata Analysis (`MetadataForensics`): Software editing signatures and EXIF timestamps
  - Biometric Face Verification (`FaceVerifier`): OpenCV Haar / cosine feature distance (if selfie uploaded)
       │
       ▼
Multimodal Evidence Fusion (`DocumentScreeningPipeline`)
  - 5-Level Evidence Hierarchy (Level 1: Artifact → Level 5: Deterministic)
  - Correlation Matrix across Compression, Structural, and Content Integrity families
  - Multi-family corroboration requirement before escalating to tamper verdicts
  - Inconclusive/degraded inputs routed to `MANUAL_REVIEW_REQUIRED`
  - Canonical verdicts: `STRUCTURALLY_VALID_UNVERIFIED`, `SUSPICIOUS`, `LIKELY_TAMPERED`, `INVALID`, `MANUAL_REVIEW_REQUIRED`
       │
       ▼
Response Sanitization (`sanitize_screening_response`)
  - Deep-copy response scrubbing at API boundary
  - Masks Aadhaar (`XXXX XXXX 1234`), PAN (`ABCDE****F`), DL (`DL-14******`), names, DOBs
  - Sanitizes freeform text in OCR results, QR payloads, and diagnostic reasons
       │
       ▼
Frontend Presentation (`frontend/src/App.jsx`)
  - Interactive Canvas viewer with color-coded forensic bounding boxes
  - Explanatory "Why This Verdict?" evidence cards
  - Quality warnings and explicit non-governmental disclaimer
```

---

## 2. Actual Bottlenecks Found in Code

### Bottleneck A: Aggressive Double Downscaling of Input Images
- **Code Locations:** `backend/app.py` (lines 419–426), `backend/nlp/ocr_engine.py` (lines 256–265).
- **Issue:** An uploaded image is first downscaled to `max_dim = 1280` in `app.py`. When passed into `_run_easyocr`, it is scaled down a second time to `max_ocr_dim = 960`.
- **Impact:** High-resolution scans (e.g., 3000x2000) have small text regions (such as 8pt Aadhaar or DL numbers) shrunk to 12–18 pixels tall. At this resolution, character strokes merge, causing optical recognition failures on digits and characters (e.g., `8` vs `B`, `0` vs `D`).

### Bottleneck B: Unpadded, Raw YOLO Bounding Box Crops
- **Code Locations:** `backend/vision/field_detector.py` (line 211), `pan_field_detector.py` (line 268), `dl_field_detector.py` (line 241).
- **Issue:** Bounding boxes are sliced directly: `crop = img[y1:y2, x1:x2]`.
- **Impact:** Object detector bounding boxes for text lines frequently fit tightly against character contours. Without margin padding (3–8 pixels or 5–10%), ascenders, descenders, and the first/last characters get clipped or touch the crop border, causing EasyOCR to omit leading letters or trailing digits.

### Bottleneck C: Absence of Resolution Checks & Upscaling on Field Crops
- **Code Locations:** `backend/fusion.py` (lines 180–208, 245–275).
- **Issue:** When a field crop is passed to OCR, the code only checks `shape[0] >= 15 and shape[1] >= 40`. The crop is fed directly to `ocr_engine.process_image` without any upscaling.
- **Impact:** EasyOCR's CRAFT text detector and recognition network perform best on text with an x-height of 32–64 pixels. A 16px high crop has an x-height of under 10px, causing severe recognition dropouts or hallucinated punctuation.

### Bottleneck D: Zero Preprocessing Variants on Field Crops
- **Code Locations:** `backend/fusion.py` (lines 182, 193, 204), `backend/nlp/ocr_engine.py`.
- **Issue:** Field crops are processed purely in raw BGR format. No grayscale conversion, contrast enhancement (CLAHE), sharpening, or binarization is attempted.
- **Impact:** Cards with complex guilloche backgrounds, holographic overlays, or colored backgrounds (e.g., blue background on PAN cards or yellow/green gradients on DLs) fail OCR because the text lacks sufficient foreground-to-background contrast.

### Bottleneck E: Primitive Crop OCR Integration Logic
- **Code Locations:** `backend/fusion.py` (lines 213–233).
- **Issue:** When crop OCR completes, the code executes a basic regex check and appends the raw matched text to `ocr_result["full_text"]`:
  ```python
  ocr_result["full_text"] = f"{ocr_result.get('full_text', '')} {crop_num.group(1)}".strip()
  ```
- **Impact:**
  1. Crop OCR does not run on names, DOBs, or father's names.
  2. It does not replace low-confidence tokens in `ocr_result["tokens"]`.
  3. It does not associate the field detector's high-confidence bounding box with the extracted text.
  4. If full-image OCR extracted a corrupt 12-digit string, `crop_num and not full_num` evaluates to False, completely ignoring the superior crop OCR result!

### Bottleneck F: Approximate Word Token Bounding Boxes
- **Code Locations:** `backend/nlp/ocr_engine.py` (lines 319–334).
- **Issue:** EasyOCR returns line-level bounding boxes. `ocr_engine.py` splits words by linearly subdividing the line width:
  ```python
  avg_w = max(1, bw // len(words))
  for idx, word in enumerate(words):
      tok_bx = bx + (idx * avg_w)
  ```
- **Impact:** Proportional fonts have widely varying word lengths. Dividing the width uniformly creates inaccurate token coordinates. When `FontAlignmentForensics` samples the pixel borders around these approximated boxes (lines 95–145 of `font_alignment.py`), it may mistake misaligned token boundaries for digital cut seams.

### Bottleneck G: Silent Mutation in PAN Validation Without Uncertainty Tracking
- **Code Locations:** `backend/nlp/field_validator.py` (lines 108–114).
- **Issue:** `validate_pan_format` replaces trailing digits on 10th character (`0->O`, `1->I`, `8->B`, `5->S`, `6->G`) silently without returning or recording uncertainty flags.
- **Impact:** If an invalid PAN happened to end in an illegal digit, the system silently mutates it into a letter, potentially passing an altered credential without diagnostic transparency.

### Bottleneck H: Rigid DL Regex Assumption
- **Code Locations:** `backend/nlp/field_validator.py` (lines 126–134, 1835).
- **Issue:** `validate_dl_format` asserts `^[A-Z]{2}[0-9]{11,14}$`.
- **Impact:** Real Indian DLs across states (legacy vs Sarathi-4) use diverse numbering formats: e.g., `DL-0420110012345` (15 chars with hyphen), `MH02 20180004567`, or `KA-01/1998/0001234`. Legitimate driving licences can be flagged as invalid format because spaces/hyphens are stripped or state format variations are not accommodated.

### Bottleneck I: Benchmark Does Not Separate FP from FN Across Document Classes
- **Code Locations:** `backend/scripts/run_benchmark.py`, `backend/scripts/run_batch_test.py`.
- **Issue:** Benchmarks report overall accuracy, precision, recall, and a 20-sample confusion matrix.
- **Impact:** False positives (genuine card flagged as tampered) and false negatives (tampered card passed as genuine) are not broken down per document type or per field extraction accuracy (CER/WER).

---

## 3. Safe Improvement Opportunities (Phase 2 Roadmap)

### Opportunity 1: Dedicated Field-Crop OCR Pipeline
Implement a dedicated function `extract_field_crop_text(crop_bgr, field_type)`:
1. **Pad Bounding Box:** Add dynamic margin padding (5–10% with min 4px, max 16px) clamped safely to image boundaries before slicing.
2. **Resolution Check & Conditional Upscaling:** If crop height is below 64px, upscale 2x or 3x using `cv2.INTER_CUBIC` or `cv2.INTER_LANCZOS4`. Keep memory minimal by upscaling only the small field crop, never the 10MB full document.
3. **Targeted Safeguards:** Skip empty, zero-dimension, or oversized crops.

### Opportunity 2: Bounded Multi-Variant OCR for Low-Confidence Crops
For crops where raw OCR returns low confidence (< 0.70) or fails field-specific format validation:
1. Variant 0: Original padded crop.
2. Variant 1: Grayscale + CLAHE (Contrast Limited Adaptive Histogram Equalization).
3. Variant 2: Unsharp masking (mild sharpening: $\text{Gaussian blur} \times -0.5 + \text{original} \times 1.5$).
4. Variant 3: Otsu / adaptive Gaussian binarization (only if contrast is extremely low).
- **Enforce strict execution bounds:** Maximum 3 variants evaluated, stop immediately upon high-confidence format match, and enforce a processing timeout guard.

### Opportunity 3: Multi-Factor OCR Result Selection
Instead of blindly picking the highest raw EasyOCR confidence:
- Select the candidate based on a composite score:
  $$\text{Score} = (\text{OCR Confidence} \times 0.4) + (\text{Format Validity} \times 0.4) + (\text{Expected Length Match} \times 0.2)$$
- If crop OCR produces a valid format match while full-image OCR failed or produced an invalid checksum, prioritize the crop OCR result and update `schema_fields`.

### Opportunity 4: Audit-Preserving PAN and DL Validation
- **PAN:** When character disambiguation is required (e.g. `0` to `O` at index 9), flag the field as `disambiguated_from_ocr = True` with a documented reason, rather than silently mutating text.
- **DL:** Expand regex coverage to recognize standard Indian MoRTH / Sarathi formats (e.g., standard 15-char formats with state code + RTO code + 4-digit year + 7-digit sequence), preserving fallback behavior.

### Opportunity 5: Token Bounding Box Provenance
- In `ocr_engine.py`, tag approximated word boxes with `"is_estimated_box": True`.
- When field detector YOLO boxes are available, use the detector's precise coordinates as the authoritative field bounding box rather than the approximated token slice.

---

## 4. Risky Changes That Must Be Avoided

| Change | Risk Level | Reason to Avoid |
|---|---|---|
| **Modifying YOLO model weights** | **CRITICAL** | Violates strict safety rule 3, 4, 5. Models must not be retrained or replaced without explicit approval and independent datasets. |
| **Upscaling entire document images** | **HIGH** | Full images upscaled 2x/3x will exceed 512MB RAM on containerized hosts (Render/Docker) and trigger OOM crashes. |
| **Unbounded OCR loops** | **HIGH** | Running OCR on dozens of filter variations will cause request timeouts (> 30s) on CPU-based servers. |
| **Arbitrary fusion weight shifts** | **HIGH** | Altering base fusion weights (30% NLP, 25% ELA, 20% Font, 15% Copy-Move, 10% Meta) will invalidate current calibrations and break regression tests. |
| **Changing official verification semantics** | **CRITICAL** | System must never claim official authentication or verified status based on heuristic OCR or YOLO results. |
| **Bypassing PII sanitization** | **CRITICAL** | Raw PII must never leak to API responses or logs. |

---

## 5. Files That Would Need Modification

| File | Nature of Changes |
|---|---|
| `backend/nlp/ocr_engine.py` | Add modular field-crop preprocessing (`preprocess_crop_variants`), crop upscaling, padded box slicing helper, and token box estimation tagging. |
| `backend/nlp/field_validator.py` | Enhance `validate_pan_format` to track character ambiguity transparently. Support standardized Sarathi-4 DL formats alongside legacy formats. |
| `backend/vision/field_detector.py` | Add optional margin/padding parameter to crop extraction (`img[y1:y2, x1:x2]`). |
| `backend/vision/pan_field_detector.py` | Add optional margin/padding parameter to crop extraction. |
| `backend/vision/dl_field_detector.py` | Add optional margin/padding parameter to crop extraction. |
| `backend/fusion.py` | Upgrade crop-assisted OCR from naive string concatenation to structured multi-variant field extraction with composite scoring. |
| `backend/tests/test_ocr_accuracy_improvements.py` | [NEW] Comprehensive test suite covering upscaling, variants, padding, PAN/DL validation, and PII safety. |

---

## 6. Proposed Tests for Phase 4

A dedicated test suite `backend/tests/test_ocr_accuracy_improvements.py` will test:
1. **Padding & Crop Geometry:**
   - Verify crop padding expands box safely without exceeding image boundaries `[0, W]` and `[0, H]`.
   - Test handling of edge bounding boxes (e.g. at $(0, 0)$ or $(W, H)$).
2. **Crop Resolution & Upscaling:**
   - Verify small crops (< 64px height) are upscaled 2x/3x.
   - Verify already-large crops are not upscaled unnecessarily.
3. **Preprocessing Variants:**
   - Verify grayscale, CLAHE, and unsharp mask variant generators return valid uint8 numpy arrays.
   - Verify variant generator terminates cleanly within maximum iteration limit.
4. **Field-Specific Validation & Provenance:**
   - PAN: Verify exact 10-character structure, and ensure optical substitutions log an uncertainty notice.
   - Aadhaar: Verify Verhoeff checksum calculation is never bypassed or weakened.
   - DL: Verify Sarathi formats (e.g., `DL-0120190012345` and `DL01 20190012345`) validate correctly.
5. **OCR Result Selection Logic:**
   - Verify that when full-image OCR fails on a field but crop OCR succeeds, the crop OCR value is selected.
   - Verify selection does not pick invalid format strings simply because of high raw confidence.
6. **Robustness & Edge Cases:**
   - Empty crops (0x0 pixels), solid black crops, low-contrast crops, blurry crops.
7. **PII Masking & Privacy:**
   - Verify that all newly extracted fields pass through `sanitize_screening_response` and are masked before reaching API callers.
8. **Regression Suite:**
   - All existing 197 tests must continue to pass without deletion, weakening, or alteration.

---

## 7. Expected Impact

| Area | Current Baseline | Expected Post-Improvement |
|---|---|---|
| **Small-Field OCR Recognition** | Fails or drops characters on text < 20px high due to tight cuts and no upscaling | Robust character extraction via padding + 2x/3x upscaling |
| **Low-Contrast Field Extraction** | High error rate on colored or patterned document backgrounds | Improved recovery via CLAHE and sharpening variants |
| **Crop Bounding Box Quality** | Characters touching YOLO borders frequently truncated | 5–10% margin padding preserves leading/trailing characters |
| **PAN Validation Transparency** | 10th character digit substitution is silent | Logged as optical disambiguation with uncertainty metadata |
| **Driving Licence Extraction** | Rigid regex rejects separated or Sarathi formats | Flexible yet strict regex handles canonical Indian formats |
| **RAM Footprint** | Stays within 512MB RAM | Remains within 512MB RAM (only small crops are transformed) |
| **API Response PII** | 100% masked | 100% masked (fully preserved) |
| **Backend Test Suite** | 197 / 197 passed | 197 existing + new focused tests passed |

---

## 8. No-Code-Change Recommendations

1. **Independent Evaluation Dataset Acquisition:**
   - Currently, real-world accuracy cannot be statistically proven across diverse populations because no large-scale, ethically sourced Indian ID evaluation dataset is available in the repository.
   - **Recommendation:** Curate a distinct, held-out test dataset (minimum 100 samples per document type: Aadhaar, PAN, DL, Passport, Voter ID) featuring real-world camera angles, glare, and compression. Keep this dataset strictly separated from training sets.

2. **Official Model Training Protocol:**
   - The DL field detector was trained on approximately 40 images, making it experimental.
   - **Recommendation:** Do not retrain or alter model weights until an explicitly approved training phase is initiated with at least 500+ annotated DL samples across all state formats.

3. **User Document Capture Guidance:**
   - Document capture quality directly limits OCR precision.
   - **Recommendation:** Present capture guidelines in client apps: avoid glare, align cards flat within frame borders, and maintain minimum resolution of 300 DPI for physical document scans.

---

## Conclusion & Next Step
This audit establishes clear, safe, and measurable opportunities to enhance OCR field extraction accuracy without modifying model weights, risking memory exhaustion, or weakening tests.

**Status:** Phase 1 complete. Awaiting explicit user approval before proceeding to Phase 2.
