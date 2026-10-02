# DocuShield AI — Phase 3.1 Pre-Implementation Review

**Date**: September 17, 2026  
**System**: DocuShield AI (Multimodal Identity Tampering Detection System)  
**Audit Scope**: Phase 3.1 Public Deployment Safety Fixes  
**Baseline Test Status**: 179 passed, 0 failed, 5 warnings (419.37s)

---

## 1. Current Implementation Findings

A thorough inspection of the repository was conducted across backend route definitions, the forensic pipeline, NLP field parsing, frontend UI presentation, Docker configuration, and test suites.

Key findings:
1. **CORS**:
   - `backend/app.py` reads `ALLOWED_ORIGINS` from `os.environ.get("ALLOWED_ORIGINS", _CORS_DEFAULT)`.
   - Default is `http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000`.
   - If `*` is present in `ALLOWED_ORIGINS`, `app.add_middleware` configures `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`, with a print warning.
   - In `render.yaml`, `ALLOWED_ORIGINS` is hardcoded to `"*"` on line 18, and `README.md` line 252 advertises `ALLOWED_ORIGINS: *`.
   - When deployed with `*`, any external web page can issue cross-origin requests to the screening API.

2. **PII Masking**:
   - `_mask_pii` exists as a local helper inside `screen_document()` in `backend/fusion.py` (lines 1063–1077).
   - Currently, `_mask_pii` only applies 3 regex substitutions:
     - 12-digit Aadhaar (`\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b` -> `XXXX XXXX \3`)
     - PAN (`\b([A-Z]{5})(\d{4})([A-Z])\b` -> `\1****\3`)
     - DL (`\b([A-Z]{2}\d{2})[\s-]?(\d{4,}[A-Z0-9]*)\b` -> `\1*****`)
   - It is only called on:
     - `structural_checks["field_details"][...]["value"]`
     - `qr_extracted_number`
     - `qr_analysis["extracted_number"]`
   - **Gaps identified**:
     - `signals["nlp_validation"]["extracted_full_text"]` returns raw OCR text directly.
     - `signals["nlp_validation"]["field_checks"]` contains unmasked field values and details.
     - `schema_fields` contains unmasked values for `name`, `father_name`, `dob`, `licence_number`, `pan_number`, `aadhaar_number`.
     - `qr_analysis["raw_text"]` returns unmasked raw QR strings.
     - Names (`name`, `father_name`) and Dates of Birth (`dob`, `date_of_birth`) are not masked at all in the backend response.
     - In `frontend/src/App.jsx`, `maskIdentifier()` was only applied to fields containing `number` or `id`, leaving name and DOB visible unless masked by backend.

3. **Benchmark UI Label**:
   - In `frontend/src/App.jsx` lines 1356–1440, Section 4 has:
     - Header badge: `20-Document Benchmark`
     - Title: `Performance & Validation`
     - Subtitle: `Evaluated on current 20-document dataset (Not a claim of universal real-world accuracy)`
     - Metric cards: `100% Accuracy (20-doc set)`, `100% Precision (20-doc set)`, `100% Recall (20-doc set)`, `100% F1 Score (20-doc set)`
     - Genuine vs Fake bars: `10 evaluated • 10 correctly identified (100%)`
   - In `BenchmarkResults` modal subcomponent: displays dynamic metrics without an explicit scoped benchmark disclaimer.
   - The wording requires tightening to eliminate any inference of real-world 100% accuracy or official verification.

4. **Verification Language**:
   - `backend/fusion.py` already includes `official_verification` with `status: "NOT_PERFORMED"`, `verified: False`, and disclaimer: *"Image-based screening cannot independently confirm official authenticity without UIDAI / issuing authority validation."*
   - `government_database_status` is `status: "NOT_PERFORMED"`.
   - `frontend/src/App.jsx` maps `AUTHENTIC` to display label `'STRUCTURALLY VALID (UNVERIFIED)'` with subtitle: *"Document structure, typography, and checksums match expected templates. Official issuing authority validation not performed."*
   - Statuses `STRUCTURALLY_VALID_UNVERIFIED`, `INCONCLUSIVE`, `MANUAL_REVIEW_REQUIRED`, `NOT_PERFORMED` are already used.
   - One minor inconsistency: in `backend/forensics/condition_analyzer.py` line 159, text states *"insufficient for reliable autonomous verification"*; this should be refined to *"autonomous screening"* to avoid conflating screening with official verification.

---

## 2. Files Inspected

| File Path | Component | Purpose / Current State |
|:---|:---|:---|
| `backend/app.py` | Backend API & App Entrypoint | CORS middleware, static mount, `/api/screen`, `/api/health`, `/api/samples`, `/api/benchmark` |
| `backend/fusion.py` | Evidence Fusion Pipeline | Multi-modal fusion, forensic rules, verdict assignment, response serialization, `_mask_pii` |
| `backend/nlp/field_validator.py` | Document & Field Validator | Regex checks, Verhoeff checksums, DL rules, schema field extraction |
| `backend/nlp/document_classifier.py` | Document Classifier | Keyword scoring & layout classification |
| `backend/nlp/ocr_engine.py` | EasyOCR / Tesseract Wrapper | OCR tokenization and bounding box extraction |
| `backend/forensics/condition_analyzer.py` | Image Quality & Reliability | Laplacian variance, exposure, glare analysis, `quality_explanation` |
| `backend/tests/test_api.py` | API Integration Tests | Endpoint smoke testing, sample and benchmark validation |
| `backend/tests/test_security.py` | Security Test Suite | Upload limits, path traversal, decompression bomb protection |
| `frontend/src/App.jsx` | React Frontend Application | UI state, canvas viewer, schema fields table, benchmark cards, modal |
| `render.yaml` | Render Blueprint Specification | Production container env vars (`ALLOWED_ORIGINS: "*"`) |
| `Dockerfile` | Multi-stage Docker Build | Multi-stage Node + Python container definition |
| `README.md` | Project Documentation | Architecture overview, local setup, deployment docs |

---

## 3. Existing CORS Behavior

In `backend/app.py`:
- `_CORS_DEFAULT = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"`
- Origins are parsed using `[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _CORS_DEFAULT).split(",") if o.strip()]`.
- If `"*"` is in `ALLOWED_ORIGINS`:
  - `allow_origins=["*"]`
  - `allow_methods=["*"]`
  - `allow_headers=["*"]`
  - Credentials are NOT enabled when wildcard is used (Starlette disallows wildcard + credentials).
- If specific origins are provided:
  - `allow_origins=ALLOWED_ORIGINS`
  - `allow_credentials=True`
  - `allow_methods=["GET", "POST", "OPTIONS"]`
  - `allow_headers=["Content-Type", "Accept", "Authorization"]`

### Identified Risk:
In `render.yaml`, `ALLOWED_ORIGINS` was explicitly set to `"*"` for deployment. If deployed to production with this setting, any site can access the API.
Furthermore, if an administrator attempts to pass an invalid or wildcard origin in production while expecting credentialed access, Starlette will either reject or open the origin unsafely.

---

## 4. Existing Masking Behavior

Currently in `backend/fusion.py`:
```python
def _mask_pii(val: Any) -> Any:
    if not isinstance(val, str):
        return val
    val = re.sub(r"\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b", r"XXXX XXXX \3", val)
    val = re.sub(r"\b([A-Z]{5})(\d{4})([A-Z])\b", r"\1****\3", val)
    val = re.sub(
        r"\b([A-Z]{2}\d{2})[\s-]?(\d{4,}[A-Z0-9]*)\b",
        lambda m: m.group(1) + "*" * len(m.group(2)),
        val
    )
    return val
```
### Observations & Gaps:
1. DL pattern handles only `([A-Z]{2}\d{2})` prefix, but DLs can be formatted with dashes or spaces (e.g. `DL-1420110012345` or `MH 12 2018 0001234`).
2. Does not mask names (`"Aarav Sharma"` remains unmasked).
3. Does not mask dates of birth (`"15/08/1990"` remains unmasked).
4. Does not mask `schema_fields` in the response:
   - `schema_fields["aadhaar_number"]["value"]`
   - `schema_fields["pan_number"]["value"]`
   - `schema_fields["licence_number"]["value"]`
   - `schema_fields["name"]["value"]`
   - `schema_fields["father_name"]["value"]`
   - `schema_fields["date_of_birth"]["value"]` / `["dob"]["value"]`
5. Does not mask raw OCR text in `signals["nlp_validation"]["extracted_full_text"]`.
6. Does not mask raw QR text in `qr_analysis["raw_text"]`.
7. Does not sanitize field values in `signals["nlp_validation"]["field_checks"]`.

---

## 5. Existing Response Schema

The public API response from `POST /api/screen` contains:
```json
{
  "status": "STRUCTURALLY_VALID_UNVERIFIED",
  "risk_level": "LOW",
  "screening_confidence": 0.94,
  "official_verification": {
    "status": "NOT_PERFORMED",
    "source": "NONE",
    "verified": false,
    "message": "..."
  },
  "structural_checks": {
    "format_valid": true,
    "checksum_valid": true,
    "all_required_fields_present": true,
    "field_details": [
      {"field": "Aadhaar Number Checksum", "status": "PASS", "value": "XXXX XXXX 3811"}
    ]
  },
  "tampering_signals": [...],
  "critical_mismatches": [...],
  "qr_mismatch_detected": false,
  "qr_extracted_number": "XXXX XXXX 3811",
  "qr_state": "VERIFIED_MATCH",
  "qr_analysis": {
    "status": "VERIFIED_MATCH",
    "raw_text": "...",
    "extracted_number": "XXXX XXXX 3811",
    "cross_field_match": true,
    "cryptographically_verified": false,
    "disclaimer": "..."
  },
  "detected_fields": [...],
  "field_detection_status": "SUCCESS",
  "field_detection_model": "...",
  "field_detection_provenance": {...},
  "recommendations": [...],
  "limitations": [...],
  "authenticity_score": 92.5,
  "risk_score": 7.5,
  "verdict": "AUTHENTIC",
  "legacy_verdict": "AUTHENTIC",
  "verdict_color": "emerald",
  "quality_tier": "GOOD",
  "reliability_score": 1.0,
  "quality_explanation": "...",
  "diagnostic_status": "CLEAN",
  "condition": {...},
  "detector_evidence": {...},
  "decision_threshold": 65,
  "why_this_verdict": {...},
  "ocr_extraction": {...},
  "condition_assessment": {...},
  "fusion_contributions": {...},
  "forensic_decision_debug": {...},
  "summary_explanation": "...",
  "critical_triggers": [...],
  "image_dimensions": {"width": 1200, "height": 800},
  "flagged_regions": [...],
  "evidence_strengths": {...},
  "ocr_engine_used": "easyocr",
  "visualizations": {...},
  "signals": {
    "nlp_validation": {
      "layer_name": "Text / NLP Field Validation",
      "score": 95.0,
      "evidence_strength": "STRONG",
      "document_type": "aadhaar",
      "extracted_full_text": "...",
      "field_checks": [...],
      "reasons": [...]
    },
    "ela_forensics": {...},
    "font_typography": {...},
    "copy_move": {...},
    "metadata_forensics": {...},
    "face_verification": {...}
  },
  "document_type": "aadhaar",
  "document_classification": {...},
  "schema_fields": {...},
  "validation_rules": [...],
  "face_verification": {...},
  "government_database_status": {...},
  "sih_evidence": {...},
  "filename": "...",
  "original_image_data_uri": "..."
}
```

### Critical Preservation Requirement:
The response schema must be preserved 100%. Only the sensitive content inside targeted leaf fields (`value`, `raw_text`, `extracted_full_text`, personal name/DOB strings) will be masked. No keys will be deleted, renamed, or restructured.

---

## 6. Existing Logging Risks

In `backend/app.py`:
- `print(f"[CORS] Allowed origins: {ALLOWED_ORIGINS}")` -> safe (origins only).
- `print(f"[RESIZE WARNING] Could not downscale uploaded image: {resize_err}")` -> safe (error only).
- `print(f"[SCREEN ERROR] Internal error occurred during document screening:\n{traceback.format_exc()}")` -> safe.
- `print(f"[BENCHMARK ERROR] {traceback.format_exc()}")` -> safe.

In `backend/vision/dl_field_detector.py` and `pan_field_detector.py`:
- Logs contain weight paths, initialization messages, device names, and error strings.
- Neither module logs raw OCR text, PAN strings, or DL numbers.

In `backend/fusion.py`:
- No prints or loggers output raw document numbers.

In `backend/scripts/run_pan_acceptance_audit.py`:
- Uses explicit `mask_pii_string()` to redact PAN and names before reporting.

**Conclusion**: The backend does not currently have active print/logger calls dumping raw PII to stdout. However, standardizing an explicit PII scrubber ensures that any future logging middleware or error diagnostic handlers cannot inadvertently leak sensitive data.

---

## 7. Existing Benchmark Wording

In `frontend/src/App.jsx`:
- Line 1367: `<span className="badge-tag badge-tag-primary">20-Document Benchmark</span>`
- Line 1364: `<div style={{ fontSize: '0.75rem', color: '#64748B', marginTop: '0.15rem' }}>Evaluated on current 20-document dataset (Not a claim of universal real-world accuracy)</div>`
- Lines 1378–1394:
  - `Accuracy (20-doc set)`: `100%`
  - `Precision (20-doc set)`: `100%`
  - `Recall (20-doc set)`: `100%`
  - `F1 Score (20-doc set)`: `100%`
- Lines 1409, 1419: `10 evaluated • 10 correctly identified (100%)`
- Lines 1704–1708: Benchmark modal results display `Accuracy`, `Precision`, `Recall`, `F1 Score`.

**Required adjustments**:
- Badge updated to: `"100% on Internal 20-Document Benchmark"` (or `"20-Document Benchmark Evaluation"`).
- Add explicit supporting notice:
  *“Measured on an internal 20-document benchmark; not equivalent to real-world authenticity accuracy or official verification.”*
- In `BenchmarkResults` modal, add a visible scoped disclaimer that evaluations reflect internal benchmark test conditions.

---

## 8. Existing Verification Wording

Current statuses in `backend/fusion.py` and `App.jsx`:
- `STRUCTURALLY_VALID_UNVERIFIED`: Used for documents whose layout, fonts, and checksums pass, but which lack official database verification.
- `INCONCLUSIVE`: Used when image degradation or conflicting evidence prevents confident automated assessment.
- `MANUAL_REVIEW_REQUIRED`: Prompted when anomalies or severe degradation are detected.
- `NOT_PERFORMED`: Explicitly attached to `official_verification` and `government_database_status`.

These statuses are already well-designed. They will be preserved with 100% compatibility.

---

## 9. Plan Items That Are Safe to Implement

1. **CORS Parsing & Production Guard**:
   - Parse comma-separated `ALLOWED_ORIGINS`.
   - Strip whitespace, trailing slashes, and empty entries.
   - If `ALLOWED_ORIGINS` is unset, default to localhost origins.
   - In production (`ENVIRONMENT=production` or `RENDER=true`), block wildcard `*` if credentials are requested or require an explicit domain.
   - Update `render.yaml` to replace `value: "*"` with documented placeholder/domain.
   - Update `README.md`.

2. **Dedicated PII Masking Module (`backend/nlp/pii_masking.py`)**:
   - `mask_aadhaar(val)`: Handles 12-digit Aadhaar with or without spaces/dashes. Idempotent.
   - `mask_pan(val)`: Handles 10-char PAN (`ABCDE1234F` -> `ABCDE****F`). Idempotent.
   - `mask_driving_license(val)`: Handles standard Indian DL formats (`BR0120190012345` -> `BR01***********`, `DL-1420110012345` -> `DL-14*********`). Idempotent.
   - `mask_name(val)`: Masks personal names (`"Aarav Sharma"` -> `"A**** S*****"`). Only applied to confirmed personal name fields.
   - `mask_dob(val)`: Masks dates of birth (`"15/08/1990"` -> `"**/**/1990"`). Only applied to confirmed personal DOB fields.
   - `mask_pii_text(text)`: Scrubs Aadhaar, PAN, and DL patterns from unstructured text (OCR full text, raw QR text).
   - `sanitize_screening_response(result)`: Applies field-aware, leaf-level masking to the final screening output dictionary without mutating internal forensic state or altering schema keys.

3. **Frontend Benchmark UI Scoping**:
   - Scope benchmark badge and labels to `"100% on Internal 20-Document Benchmark"` / `"20-Document Benchmark Evaluation"`.
   - Add supporting disclaimers in Section 4 and in `BenchmarkResults` modal.

4. **Verification Language Refinement**:
   - In `backend/forensics/condition_analyzer.py`, refine `"autonomous verification"` to `"autonomous screening"`.
   - Reiterate disclaimers on structural validity vs official verification.

5. **Security & Regression Testing**:
   - CORS origin verification tests (allowed origin, rejected origin, multi-origin parsing).
   - PII masking unit tests (Aadhaar, PAN, DL, names, DOB, idempotency, OCR text, nested response).
   - Run full pytest suite (must pass 179+ tests).
   - Frontend build test (`npm run build`).

---

## 10. Plan Items Requiring Adjustment

1. **Avoid Hardcoded Domain in `render.yaml`**:
   - Do NOT assume a specific domain like `https://docushield-ai.onrender.com` as final if the user might deploy under custom domain or HF Spaces.
   - Instead, provide a clear, documented default with explicit instructions:
     ```yaml
     - key: ALLOWED_ORIGINS
       # Comma-separated list of allowed frontend domains. Do NOT use '*' in production.
       value: "https://your-frontend-domain.com"
     ```
2. **Do Not Mask Model Metadata or Explanations Blindly**:
   - Never run blanket recursive string masking over the entire response dictionary, because words like `YOLOv8n`, `Aadhaar`, `Sarathi`, or reason strings like `"Photo swap detected at (50, 130)"` could be corrupted.
   - Masking must be strictly field-aware.
3. **Frontend Masking Fallback**:
   - Update frontend `maskIdentifier()` to recognise already-masked strings (`XXXX`, `****`, `•••`) so it does not double-mask or mangle strings returned by the backend.

---

## 11. Potential Compatibility Risks

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| Existing tests break due to changed response values | Low | Existing tests check `res["status"]`, `res["risk_score"]`, `res["verdict"]`. Verify whether any existing test asserts on raw unmasked Aadhaar in `schema_fields`. We verified that `test_aadhaar_tampering_scenarios.py` checks verdicts and checksum status, not raw PII strings. |
| Double-masking corrupts UI strings | Low | Build idempotency into all masking helpers (`if "XXXX" in val or "****" in val: return val`). |
| CORS middleware blocks legitimate localhost ports | Low | Preserve `_CORS_DEFAULT` with all standard dev ports (`5173`, `3000`, `127.0.0.1:5173`, `127.0.0.1:3000`). |
| Starlette CORS credentials error | Low | Ensure `allow_credentials=True` is only set when specific origins are configured, never when wildcard `*` is present. |

---

## 12. Exact Proposed File Changes

1. **`backend/nlp/pii_masking.py`** [NEW]:
   - Create deterministic, idempotent masking functions and field-aware response sanitizer.
2. **`backend/fusion.py`** [MODIFY]:
   - Import and apply `sanitize_screening_response(result)` at the very return statement of `screen_document()`, keeping all prior forensic calculations on original data.
3. **`backend/app.py`** [MODIFY]:
   - Refactor CORS configuration to support clean multi-origin parsing, strip whitespace/slashes, handle production restrictions, and disallow wildcard with credentials.
4. **`backend/forensics/condition_analyzer.py`** [MODIFY]:
   - Line 159: Change `"autonomous verification"` to `"autonomous screening"`.
5. **`frontend/src/App.jsx`** [MODIFY]:
   - Enhance `maskIdentifier()` for idempotency with backend-masked strings.
   - Update Section 4 benchmark badge to `"100% on Internal 20-Document Benchmark"`.
   - Add supporting disclaimers in Section 4 and the `BenchmarkResults` modal.
   - Ensure schema fields table masks names and dates of birth if unmasked.
6. **`render.yaml`** [MODIFY]:
   - Set `ALLOWED_ORIGINS` to documented placeholder `"https://your-frontend-domain.com"`.
7. **`README.md`** [MODIFY]:
   - Update CORS configuration instructions.
8. **`backend/tests/test_pii_masking.py`** [NEW]:
   - Unit and integration tests for all PII masking logic.
9. **`backend/tests/test_security.py`** [MODIFY]:
   - Add CORS allowed and rejected origin tests.
