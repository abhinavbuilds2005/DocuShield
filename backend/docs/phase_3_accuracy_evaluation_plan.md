# Phase 3: Independent Accuracy Evaluation Plan

## Executive Notice & Dataset Audit Findings

> [!WARNING]
> **Real Accuracy Cannot Yet Be Established**:
> An exhaustive audit of the project repository reveals that **no independent, labeled OCR ground-truth evaluation dataset currently exists** for the five supported document categories (Aadhaar, PAN, Driving Licence, Passport, Voter ID).
>
> - The existing training/validation splits in `data/aadhaar_field_detection/`, `data/processed/pan_field_detection/`, and `data/processed/dl_field_detection/` provide **only YOLO bounding-box coordinates** (`class_id x_center y_center w h`) for object detection. They contain **zero ground-truth text strings** (no true names, numbers, dates, or addresses).
> - The files in `backend/data/ground_truth.json` comprise an **internal 20-document mock/synthetic benchmark** used exclusively for functional integration tests. Per Rule 7, this internal synthetic set **cannot and must not be used as proof of real-world OCR or authenticity accuracy**.
> - In accordance with the Phase 3 strict execution instructions, **no external datasets have been downloaded, and no synthetic datasets have been generated**. Real-world accuracy and error rates can only be measured once an approved, independent evaluation dataset is curated.

---

## 1. Evaluation Objectives

The objective of this evaluation plan is to establish a rigorous, mathematically sound, and privacy-compliant benchmark framework to measure:
1. Field detection localization accuracy (bounding boxes vs. ground truth).
2. Field text extraction accuracy (OCR string exact-match, Character Error Rate, Word Error Rate).
3. Deterministic algorithmic validation efficacy (Verhoeff checksum, PAN entity logic, Sarathi-4 DL structure).
4. Screening verdict behavior (false-positive rate on genuine documents, false-negative rate on tampered documents, manual-review escalation rate).
5. Computational and operational performance (latency and memory profile).

---

## 2. Dataset Requirements & Target Composition

To obtain statistically meaningful results with a 95% confidence interval ($\pm 3\%$ margin of error), the evaluation corpus must be partitioned across all five supported document classes and cover varied real-world acquisition scenarios.

### A. Document Type Target Quotas
A minimum of **500 independent, real-world test documents** is required:

| Document Type | Minimum Samples | Genuine / Normal | Distorted / Mobile / Degraded | Tampered / Adversarial |
| :--- | :--- | :--- | :--- | :--- |
| **Aadhaar** | 120 | 60 | 40 | 20 |
| **PAN Card** | 120 | 60 | 40 | 20 |
| **Driving Licence** | 120 | 50 | 50 | 20 |
| **Voter ID (EPIC)** | 80 | 40 | 30 | 10 |
| **Passport** | 60 | 30 | 20 | 10 |
| **Total** | **500** | **240 (48%)** | **180 (36%)** | **80 (16%)** |

### B. Acquisition & Environmental Conditions
Test samples must be distributed across the following operational categories:
1. **Category A: High-Quality Flatbed Scans** (300 DPI, unskewed, uniform lighting).
2. **Category B: Mobile Camera Scans** (Variable angles, ambient room light, slight glare).
3. **Category C: WhatsApp / Social Media Compression** (Downscaled to ~800–1280px, high JPEG compression artifacts, $Q \le 50$).
4. **Category D: Geometric Distortions** (Rotations of $\pm 5^\circ$ to $\pm 25^\circ$, perspective keystoning).
5. **Category E: Degraded Physical Cards** (Faded thermal printing, scratches, laminating film reflections).
6. **Category F: Digital Screenshots** (Mobile display screenshots, PDF rasterizations).
7. **Category G: Negative Controls** (Non-identity documents, blank pages, foreign certificates).

---

## 3. Required Annotations Schema

Each test sample must have a cryptographically linked JSON metadata file (e.g., `<hash_id>.eval.json`). To avoid PII exposure, all document files are referenced by anonymous UUIDs.

```json
{
  "document_id": "eval-aadhaar-0042",
  "document_type": "aadhaar",
  "acquisition_category": "mobile_camera_compressed",
  "ground_truth": {
    "is_authentic": true,
    "tampering_type": null,
    "fields": {
      "aadhaar_number": {
        "text": "XXXX XXXX 1234",
        "raw_text_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "bbox": [210, 360, 540, 65],
        "is_mandatory": true
      },
      "name": {
        "text": "ANON USER",
        "bbox": [210, 140, 320, 40],
        "is_mandatory": true
      },
      "dob": {
        "text": "15/08/1990",
        "bbox": [210, 190, 180, 35],
        "is_mandatory": true
      },
      "gender": {
        "text": "MALE",
        "bbox": [210, 230, 100, 30],
        "is_mandatory": true
      }
    }
  }
}
```

> [!IMPORTANT]
> **Zero Raw PII Storage Policy**:
> Evaluation ground-truth datasets must either use explicitly consented synthetic/anonymized documents or store identity strings in salted one-way hashes (`SHA-256(salt + raw_id)`) so that exact-match verification can be performed mathematically without storing unmasked national identity numbers on disk.

---

## 4. Privacy, Consent & Compliance Standards

To comply with the Digital Personal Data Protection Act (DPDPA 2023), Aadhaar Regulations, and enterprise security policies:
1. **Explicit Written Consent**: All real document samples collected must possess verifiable opt-in consent for algorithmic evaluation.
2. **Local Machine Containment**: Evaluation data must remain within the local workspace; transmission over external network connections is prohibited.
3. **No Unmasked PII in Artifacts**: Reports, logs, terminal traces, and error summaries must display masked strings only (`XXXX XXXX 1234`, `ABCP•••••F`).
4. **Permanent Purge Capability**: When benchmarking completes, raw evaluation images must be shreddable via a single script.

---

## 5. Quantitative Evaluation Metrics

### A. Computer Vision (Field Detection)
- **Intersection over Union (IoU)**: Evaluated at thresholds $\text{IoU} \ge 0.50$ (PASCAL VOC) and $\text{IoU} \ge 0.75$.
- **Precision ($P$)**: $\frac{TP}{TP + FP}$
- **Recall ($R$)**: $\frac{TP}{TP + FN}$
- **Mean Average Precision (mAP@50)**: Across all 7 Aadhaar classes, 5 PAN classes, and 6 DL classes.

### B. NLP & OCR Extraction Accuracy
- **Exact Match Ratio (EMR)**: Percentage of fields extracted with 100% string identity to ground truth.
- **Character Error Rate (CER)**:
  $$\text{CER} = \frac{S + D + I}{N}$$
  Where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is total ground-truth characters.
- **Normalized Levenshtein Similarity**:
  $$1 - \frac{\text{Levenshtein}(T_{\text{pred}}, T_{\text{gt}})}{\max(|T_{\text{pred}}|, |T_{\text{gt}}|)}$$
- **Algorithmic Validation Pass Rate**:
  - Verhoeff validity on extracted 12-digit Aadhaar strings.
  - Entity-code check validity on extracted 10-character PAN strings.
  - Sarathi-4 syntax and State/UT code validity on extracted DL numbers.

### C. Forensic Screening Performance
- **False Positive Rate (FPR)**: Percentage of genuine documents incorrectly flagged as `FLAGGED / TAMPERED`.
- **False Negative Rate (FNR)**: Percentage of tampered/counterfeit documents incorrectly marked as `MANUAL_REVIEW_REQUIRED` or clean.
- **Manual Review Rate (MRR)**: Percentage of documents escalated to human oversight due to low OCR confidence or ambiguous edge cases.

### D. Computational Benchmarks
- **Mean Latency per Document**: Measured in milliseconds (EasyOCR pass + YOLO detector + fusion pipeline).
- **Peak RAM / VRAM Consumption**: Monitored via `psutil` or `torch.cuda` memory tracking to verify $< 512\,\text{MB}$ compliance.

---

## 6. Baseline Comparison Method (Phase 1 vs. Phase 2)

Once the independent dataset is approved and loaded, the comparison framework will execute two benchmark passes on identical hardware:

1. **Baseline Configuration (Phase 1 Pipeline)**:
   - Full-image double downscaling ($1280\text{px} \to 960\text{px}$).
   - Unpadded bounding-box field crops ($\text{pad\_ratio} = 0.0$).
   - No crop upscaling.
   - Single OCR attempt per crop (no CLAHE/unsharp variants).
   - Naive text concatenation without composite candidate scoring.
   - Standard PAN validation without optical disambiguation tracking.

2. **Improved Configuration (Phase 2 Pipeline)**:
   - Single-pass resolution policy ($1280\text{px}$ canvas).
   - 8% padded field crops with boundary clamping.
   - Controlled 2x/3x bicubic upscaling on small crops.
   - Bounded preprocessing variants (max 3) with fast-exit.
   - Multi-factor candidate selection ($0.4 \text{conf} + 0.4 \text{format} + 0.2 \text{length}$).
   - Transparent PAN disambiguation tracking.
   - Modern Sarathi-4 DL parsing.

3. **Delta Reporting**:
   - $\Delta \text{EMR} = \text{EMR}_{\text{Phase2}} - \text{EMR}_{\text{Phase1}}$
   - $\Delta \text{CER} = \text{CER}_{\text{Phase1}} - \text{CER}_{\text{Phase2}}$
   - $\Delta \text{Latency} = \text{Time}_{\text{Phase2}} - \text{Time}_{\text{Phase1}}$

---

## 7. Acceptance Criteria for Real-World Deployment

To consider the field-extraction pipeline ready for production deployment, the following minimum performance criteria must be met on the independent evaluation dataset:

1. **Aadhaar Number EMR**: $\ge 92.0\%$ on mobile/compressed images; $\ge 98.0\%$ on clean scans.
2. **PAN Number EMR**: $\ge 94.0\%$ on mobile/compressed images; $\ge 99.0\%$ on clean scans.
3. **Driving Licence EMR**: $\ge 88.0\%$ on standard Sarathi-4 layouts.
4. **False Positive Rate**: $\le 3.0\%$ on genuine documents.
5. **No Regressions**: Zero instances where Phase 2 corrupts a field accurately extracted by Phase 1.
6. **Execution Budget**: Total screening time $\le 1.8\,\text{seconds}$ per document on CPU; $\le 0.6\,\text{seconds}$ on GPU.

---

## 8. Limitations & Edge Cases

1. **Physical Card Damage**: Extensive water damage, severed cards, or deliberate scratching of numbers cannot be resolved by OCR alone and will trigger `MANUAL_REVIEW_REQUIRED`.
2. **Camera Motion Blur**: Fast movement causing point spread blur exceeding 5 pixels will result in character misrecognition.
3. **Regional Non-Standard DL Layouts**: Booklets, hand-written licences, or pre-2000 state cards without standard digital serial numbers will require human verification.
4. **No Issuer Verification**: Even a 100% character-accurate extraction does not verify whether the document identity is active or genuine in government registries. Official verification remains out of scope.
