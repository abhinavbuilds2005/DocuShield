# Independent Dataset Intake & Approval Specification

## 1. Scope and Objective

This specification establishes the mandatory governance, privacy, legal, and technical requirements for onboarding an independent evaluation dataset into the **DocuShield AI** project. 

Under Phase 3.1, no data collection, automated scraping, or data generation may occur without complying with this specification. Every proposed dataset must pass the approval checklist defined herein before ingestion into the benchmark harness.

---

## 2. Approved Dataset Sources

Only datasets originating from the following vetted channels may be considered for evaluation intake:

1. **Explicitly Consented Private Evaluation Data**:
   - Primary scans or mobile captures provided directly by participants who have signed an explicit Data Processing Consent Agreement under the Digital Personal Data Protection Act (DPDPA 2023).
   - The consent agreement must explicitly authorize non-commercial AI forensic and OCR evaluation.
2. **Legally Usable High-Fidelity Synthetic Data**:
   - Programmatically generated document templates rendered using graphics engines (e.g. SVG/HTML headless canvas) populated with purely fictional personae (e.g. `TEST USER`, `SAMPLE CITIZEN`, mathematically generated valid/invalid Verhoeff numbers from non-allocated series).
   - Generative adversarial or diffusion-based synthetic identity mockups clearly designated as non-real representations.
3. **Public Academic & Benchmark Datasets with Verified Open Licenses**:
   - Datasets published by academic institutions or research conferences under permissive licenses (e.g., CC-BY-4.0, MIT, Open Data Commons).
   - Must contain documented ethical clearance or IRB (Institutional Review Board) approval.
4. **Properly Redacted Real Documents**:
   - Documents where non-critical PII (such as full home address, parent name, or signature) has been permanently pixelated or redacted prior to ingestion, leaving only the target evaluation fields intact.

---

## 3. Disallowed Data (Strict Rejections)

Any dataset exhibiting any of the following characteristics will be **immediately and unconditionally rejected**:

- **Unconsented Personal Documents**: Scans or mobile photographs of real individuals collected without documented, verifiable opt-in consent.
- **Random Internet Scrapes**: Images sourced from search engines (Google Images, DuckDuckGo), social media platforms, public cloud buckets, or forums without unambiguous commercial/research licensing.
- **Data Dumps & Public Exposures**: Data originating from data breaches, dark web dumps, or leaked government repositories.
- **Unnecessary Raw PII**: Datasets that preserve full unmasked personal identifiers when evaluation can be satisfied via salted hashes or masked strings.
- **Unclear Provenance**: Datasets lacking identifiable authorship, acquisition methodology, or custodial history.

---

## 4. Required Metadata per Sample

Every ingested sample must be accompanied by a companion metadata record detailing its physical and digital acquisition properties:

| Metadata Field | Type | Description / Valid Values |
| :--- | :--- | :--- |
| `sample_id` | `string` | Anonymized unique identifier (e.g. `eval-pan-0014`, UUIDv4). |
| `document_type` | `string` | `aadhaar`, `pan`, `driving_license`, `passport`, `voter_id`, or `negative_control`. |
| `acquisition_category` | `string` | `flatbed_scan`, `mobile_camera`, `whatsapp_compressed`, `rotated`, `low_light`, `degraded_physical`, `screenshot`. |
| `image_dimensions` | `array[int]` | `[width, height]` in pixels. |
| `compression_status` | `string` | `lossless` (PNG/TIFF), `jpeg_high_q` ($Q \ge 85$), `jpeg_low_q` ($Q \le 50$). |
| `rotation_perspective` | `string` | `none`, `rotation_5deg`, `rotation_15deg`, `rotation_25deg`, `perspective_keystone`. |
| `ground_truth_status` | `string` | `dual_annotated_verified`, `single_annotated`, `unverified`. |
| `consent_license_id` | `string` | Reference ID to legal consent agreement or public license URI. |
| `dataset_split` | `string` | `test` (locked evaluation), `dev` (validation/calibration), `train` (if retraining approved in future). |

---

## 5. Ground-Truth Schema & Zero Raw PII Storage Policy

To prevent storing raw national identity numbers (Aadhaar, PAN, DL) in source control or disk storage, the evaluation schema utilizes **Salted One-Way Cryptographic Hashing**.

### A. Salt Management Guidance
1. **Per-Dataset Salt**: A 256-bit cryptographically secure random salt must be generated during dataset intake and stored in a secure local environment variable (`EVAL_DATASET_SALT`).
2. **Never Commit Salt to Git**: The salt must **never** be committed to version control, embedded in markdown docs, or written to public logs.
3. **Evaluation Verification Formula**:
   $$\text{Hash}_{\text{expected}} = \text{HMAC-SHA256}(\text{Salt}, \text{Normalize}(\text{RawID}))$$
   During evaluation, the predicted OCR string is normalized (whitespace/hyphens removed, uppercased) and hashed with the salt:
   $$\text{Hash}_{\text{predicted}} = \text{HMAC-SHA256}(\text{Salt}, \text{Normalize}(\text{PredictedOCR}))$$
   If $\text{Hash}_{\text{predicted}} == \text{Hash}_{\text{expected}}$, an **Exact Match (EM)** is recorded without ever exposing the unmasked plaintext ID.

### B. Schema Field Definitions
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "sample_id": "eval-doc-0001",
  "document_class": "pan",
  "is_negative_control": false,
  "annotation_confidence": 1.0,
  "tampering_label": {
    "is_tampered": false,
    "tampering_types": [],
    "tampered_regions": []
  },
  "fields": {
    "primary_id": {
      "field_name": "pan_number",
      "masked_display": "ABCP•••••F",
      "salted_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
      "normalized_length": 10,
      "format_pattern": "^[A-Z]{5}[0-9]{4}[A-Z]$",
      "bbox": [150, 320, 280, 45],
      "is_mandatory": true
    },
    "cardholder_name": {
      "field_name": "name",
      "masked_display": "J••• D••",
      "salted_hash": "c6a1e9...",
      "normalized_length": 8,
      "bbox": [150, 140, 310, 35],
      "is_mandatory": true
    }
  }
}
```

---

## 6. Dataset Split Rules & Leakage Prevention

If the dataset is utilized across development and testing phases, strict isolation boundaries must be observed:

1. **Independent Test Set Isolation**:
   - The test partition must comprise at least **30% of the total dataset** (minimum 250 documents for comprehensive evaluation).
   - **No Hyperparameter Tuning on Test Set**: The test set must remain completely unread and untouched during pipeline refinement. It is executed only for the final evaluation run.
2. **Zero Duplicate or Near-Duplicate Leakage**:
   - Perceptual hashing (pHash) and feature vector similarity must be run across splits to verify that no duplicate images exist across `train`, `dev`, or `test`.
3. **No Variant Leakage**:
   - If an image is captured under multiple conditions (e.g. flatbed scan and mobile photo of the same card), **all variants of that specific card must reside in the same split**. Spanning the same card across train and test is strictly forbidden.

---

## 7. Quantitative Evaluation Protocol

The independent evaluation script must execute and report the following standardized metrics:

### A. Field Localization (Computer Vision)
- **Bounding Box IoU**: Overlap between predicted detector bounding box and ground-truth annotation.
- **Field Precision & Recall**: Measured at $\text{IoU} \ge 0.50$ and $\text{IoU} \ge 0.75$.
- **Mean Average Precision (mAP@50)**: Evaluated per class and across the full document schema.

### B. Field Extraction (OCR & NLP)
- **Exact Match Ratio (EMR)**:
  $$\text{EMR} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Hash}_{\text{pred}} == \text{Hash}_{\text{gt}})}{N}$$
- **Character Error Rate (CER)**:
  $$\text{CER} = \frac{\text{Levenshtein}(T_{\text{pred}}, T_{\text{gt}})}{\text{Length}(T_{\text{gt}})}$$
  *(Evaluated on non-sensitive fields or inside memory-isolated sandboxes with instantaneous variable clearing)*.
- **Normalized Levenshtein Similarity**: String similarity scaled between $0.0$ and $1.0$.

### C. Forensic Screening Efficacy
- **False Positive Rate (FPR)**: Proportion of genuine documents flagged as `FLAGGED / TAMPERED`. Target: $\le 3.0\%$.
- **False Negative Rate (FNR)**: Proportion of tampered documents flagged as clean. Target: $\le 2.0\%$.
- **Manual Review Rate (MRR)**: Proportion of ambiguous or degraded cards escalated to human review. Target: $\le 10.0\%$.

### D. Computational Benchmarks
- **Average Screening Latency**: Wall-clock milliseconds per document (OCR + YOLO + Fusion).
- **Peak RAM / VRAM Allocation**: Measured continuously to guarantee $< 512\,\text{MB}$ compliance.

---

## 8. Dataset Intake Approval Checklist

Before any external dataset file is committed, copied, or processed by the evaluation framework, the Lead Engineer and Compliance Reviewer must sign off on the following checklist:

| # | Check Item | Status | Verifier Notes |
| :---: | :--- | :---: | :--- |
| **1** | **License or Consent Verified** | [ ] PENDING | Verified opt-in consent forms or open academic license URI archived. |
| **2** | **Privacy Review Completed** | [ ] PENDING | Zero unconsented PII present. Verified compliance with DPDPA 2023. |
| **3** | **Zero Raw PII Storage Approved** | [ ] PENDING | Sensitive fields hashed with HMAC-SHA256. Salt stored outside version control. |
| **4** | **Annotation Quality Checked** | [ ] PENDING | Dual-annotator verification on $\ge 15\%$ sample subset; bounding boxes tight. |
| **5** | **Deduplication Scan Completed** | [ ] PENDING | Perceptual hash scan confirmed zero duplicate cards across splits. |
| **6** | **Dataset Split Finalized** | [ ] PENDING | Train, dev, and test sets strictly separated with zero variant leakage. |
| **7** | **Evaluation Set Locked** | [ ] PENDING | Test directory checksum frozen via SHA-256 manifest. |
