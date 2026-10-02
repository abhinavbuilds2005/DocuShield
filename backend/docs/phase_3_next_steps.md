# Phase 3: Project Status & Next Steps

## 1. What Is Complete

1. **Phase 1: Read-Only System & Accuracy Audit**:
   - Identified 9 architectural and OCR bottlenecks (double downscaling, unpadded field bounding boxes, small crop degradation, naive concatenation, silently mutated characters).
   - Documented comprehensive baseline analysis in `backend/docs/accuracy_improvement_audit.md`.

2. **Phase 2: Safe OCR Accuracy Improvements**:
   - Unified image downscaling policies (`max_ocr_dim = 1280`, `canvas_size = 1280`) in `app.py` and `ocr_engine.py` to completely eliminate double downscaling.
   - Built `backend/vision/crop_utils.py` introducing padded field crop extraction (8% padding with strict boundary clamping), controlled 2x/3x aspect-preserving upscaling for small crops, and bounded CLAHE/unsharp preprocessing variants (max 3 variants per field with early-exit on high confidence).
   - Implemented composite candidate selection (`0.4 conf + 0.4 format + 0.2 length`) in `ocr_engine.py` and `fusion.py`.
   - Transformed `validate_pan_format` to track optical character disambiguation explicitly while maintaining 100% tuple unpacking compatibility.
   - Extended `validate_dl_format` to support modern 15-character Sarathi-4 DL structures.
   - Tagged approximate token bounding boxes (`is_estimated_box = True`) to distinguish interpolated word boxes from ground-truth lines.
   - Preserved API-boundary PII masking across all screening responses and crop metadata.
   - **Verification Passed**: 219/219 backend tests passed, 25/25 PII/security tests passed, frontend build passed cleanly in 2.39s, and all model SHA-256 hashes remained identical.

3. **Phase 3.1: Evaluation Architecture & Intake Specification**:
   - Designed the independent accuracy evaluation framework in `backend/docs/phase_3_accuracy_evaluation_plan.md`.
   - Formally documented the dataset audit and conceptual boundaries in `backend/docs/phase_3_accuracy_evaluation_report.md`.
   - Established the legal, privacy, and metadata requirements in `backend/docs/independent_dataset_intake_specification.md`.
   - Provided a privacy-preserving example schema in `backend/docs/examples/evaluation_sample_schema.json` using salted HMAC-SHA256 hashes.

---

## 2. What Is Currently Blocked

**Empirical Accuracy Benchmarking is Blocked**:
- Real-world extraction accuracy (Exact Match Ratio, Character Error Rate, Word Error Rate, and detection mAP50) **cannot be computed** because no labeled independent real-world ground-truth dataset exists in the workspace.
- In compliance with strict user constraints, the system has **not** downloaded any external datasets, has **not** generated automated synthetic documents, and has **not** used the internal 20-mock regression set as evidence of real-world authenticity.

---

## 3. What Must Be Approved Before Data Collection

Before any dataset is collected, imported, or ingested, the following formal approvals must be granted:

1. **Source Authorization**:
   - Explicit written consent forms (under DPDPA 2023) for private participant data, OR
   - Formal verification of an academic open-license agreement (e.g. CC-BY-4.0) for public benchmark images, OR
   - Approval of a controlled synthetic generator script utilizing fictional templates and disclaimed dummy identities.
2. **Privacy & Salt Management Sign-Off**:
   - Approval of the salted HMAC-SHA256 storage protocol to guarantee that no unmasked Aadhaar, PAN, or DL numbers are ever written to disk or source control.
3. **Approval Checklist Sign-Off**:
   - Completion of all 7 items in the [Dataset Intake Approval Checklist](file:///d:/Document%20detector/backend/docs/independent_dataset_intake_specification.md#8-dataset-intake-approval-checklist).

---

## 4. Execution Workflow After Approved Dataset Is Provided

Once an approved dataset satisfying the intake specification is deposited in the workspace:

```
[Approved Dataset Intake] 
          │
          ▼
1. Verify SHA-256 Checksums & Intake Checklist
          │
          ▼
2. Run Isolated Evaluation Harness (Phase 1 Baseline Pipeline vs. Phase 2 Pipeline)
          │
          ▼
3. Compute Objective Metrics:
   - Field Detection: Precision, Recall, mAP50
   - OCR Extraction: EMR, CER, Levenshtein Similarity
   - Rule Efficacy: Verhoeff validity, PAN entity logic, Sarathi-4 DL syntax
   - Screening Verdicts: False Positive Rate (FPR), False Negative Rate (FNR), Manual Review Rate (MRR)
   - System Performance: Average latency (ms) & peak RAM consumption
          │
          ▼
4. Generate Empirical Comparative Report (`backend/docs/phase_3_accuracy_evaluation_report.md`)
   - Document absolute delta improvements and confirm zero regressions
```
