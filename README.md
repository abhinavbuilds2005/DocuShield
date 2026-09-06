---
title: DocuShield AI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DocuShield AI: AI-Based Identity & Document Forensic Screening System
**Smart India Hackathon (SIH 2026) — Problem Statement SIH26188**

A multi-layered AI, computer vision, and document forensics system designed to screen, detect, and explain tampering indicators in identity documents (Aadhaar-style, PAN-style, Driving License, and voter cards) using a multimodal fusion of **Text/NLP Field Validation** and **Image Forensics Tampering Detection**.

> **Important Forensic Notice**: This system performs **forensic risk assessment and tampering indicator detection**. It provides explainable supporting evidence for human reviewers; it does **not** claim to provide infallible or guaranteed fraud detection.

---

## Architecture Overview

```
                                  Uploaded Document Image
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
             [Layer 1: NLP & Field Logic]             [Layer 2: Image Forensics]
          • Real OCR (EasyOCR / Tesseract)         • Error Level Analysis (ELA)
          • Normalized BBox Coordinates            • Copy-Move Keypoint Matching + RANSAC
          • Verhoeff Checksum Algorithm (12-digit) • Typography & Laplacian Variance
          • PAN Tax Entity Alphanumeric Regex      • EXIF & Software Signatures
          • Chronological Date Logic & Formats     • Pixel-Level Gradient Maps
                         │                                       │
                         └───────────────────┬───────────────────┘
                                             ▼
                              [Multimodal Fusion Engine]
                           • Calibrated Authenticity Score (0-100%)
                           • 3-Tier Categorical Verdict (AUTHENTIC, SUSPICIOUS, FLAGGED)
                           • Evidence Strength Classification per Signal
                           • Multi-Layer Corroboration Scoring (No Single-Veto Blind Spots)
                           • Unified Flagged Bounding Box Overlays
                                             │
                                             ▼
                              [Modern React UI Dashboard]
                           • Interactive Forensic Canvas with SVG Bounding Boxes
                           • Toggleable ELA Heatmaps & Edge Gradient Maps
                           • Quick Test Carousel (20 Synthetic Paired Cards)
                           • Sidecar-Free Real-OCR Benchmark Evaluation
```

---

## Key Detection Layers

### 1. Text / NLP Field Validation
- **Real OCR Processing**: Runs EasyOCR as primary neural text recognizer, producing token-level and line-level text with both absolute pixel coordinates and normalized coordinates `[0, 1]`.
- **Strict Mode Separation**: Real document screening (`POST /api/screen`) **never** reads sidecar `.ocr.json` or `ground_truth.json` files. If OCR fails or is unavailable, a structured `OCR_UNAVAILABLE` error is returned; fake placeholder OCR is never returned.
- **Verhoeff Checksum Verification**: Validates 12-digit national identity numbers against the D5 dihedral group permutation tables. Altered or fabricated digits are flagged as strong supporting evidence.
- **Tax ID / PAN Alphanumeric Pattern**: Enforces 10-character structure (`ABCDE1234F`), checking that the 4th character matches official entity types (e.g., `P` for Person) and the 5th character matches the surname initial.
- **Chronological Date Consistency**: Validates calendar plausibility and chronological relationships (e.g., driving license expiry must follow issue date).

### 2. Image Forensics Tampering Detection
- **Error Level Analysis (ELA)**: Recompresses the image at 90% JPEG quality to measure localized quantization error. Regions altered or pasted with differing compression histories exhibit elevated error hotspots. Includes format-awareness notes for PNG vs. JPEG inputs.
- **Typography & Rendering Forensics**: Evaluates stroke sharpness via local Laplacian variance, edge density via Canny gradients, and relative OCR token confidence. Flags words that exhibit statistically significant sharpness and edge disparity relative to the line baseline.
- **Copy-Move Forgery Detection**: Employs ORB feature descriptors with self-match elimination, Lowe's ratio test, spatial displacement clustering, and RANSAC geometric affine verification to detect cloned graphics, stamps, or duplicated background patches.
- **Metadata & EXIF Inspection**: Inspects embedded image headers and EXIF fields, checking for editing software signatures (Adobe Photoshop, Canva, GIMP) and anomalies.

### 3. Multimodal Evidence Fusion
- Calculates a weighted authenticity score (NLP 30%, ELA 25%, Typography 20%, Copy-Move 15%, Metadata 10%).
- Implements **multi-layer corroboration**: a single anomaly reduces confidence or shifts verdict to `SUSPICIOUS`, while multiple concurring forensic signals lower the score into `FLAGGED / TAMPERED`.

---

## Security & Robustness Measures

- **File Upload Limits**: Enforces a strict 10 MB upload ceiling.
- **Magic Byte Inspection**: Validates true binary signatures for JPEG (`FF D8 FF`), PNG (`89 50 4E 47`), and WebP (`RIFF...WEBP`). Rejects mismatched Content-Types.
- **Decompression Bomb Protection**: Guarded against pixel floods with dimension limits (maximum 8192×8192 pixels) and `PIL.Image.MAX_IMAGE_PIXELS` capping.
- **Path Traversal Prevention**: User-provided `sample_id` and `/api/image/{filename}` parameters are strictly checked against a dataset whitelist and validated with `os.path.realpath` containment checks.
- **CORS Hardening**: Configurable allowed origins via the `ALLOWED_ORIGINS` environment variable (defaults to local Vite dev server).
- **Structured Error Responses**: Clean JSON errors (`FILE_TOO_LARGE`, `UNSUPPORTED_FORMAT`, `CORRUPT_IMAGE`, `OCR_UNAVAILABLE`, `SAMPLE_NOT_FOUND`) without leaking raw Python stack traces.

---

## Ethical Synthetic Data Strategy

1. **No Real PII**: No real government identity cards or citizen data are collected, scraped, or stored.
2. **Procedural Generation**: All templates are programmatically generated using Pillow with stylized avatar silhouettes and fictitious identities (e.g., "Aarav Sharma", "Priya Verma").
3. **Mandatory Watermarking**: Every generated card (both baseline and tampered) carries explicit, prominent watermarks:
   - Header: `"SAMPLE / MOCK — NOT A REAL GOVERNMENT DOCUMENT — SIH 2026 DEMO"`
   - Diagonal Banner: `"MOCK / SPECIMEN — NOT REAL ID"`
   - Footer: `"SYNTHETIC FICTIONAL TEST DATASET — ETHICAL AI RESEARCH ONLY"`
4. **20-Document Synthetic Test Suite**: 9 genuine baseline cards + 11 tampered variants with specific attack vectors (photo swap, checksum corruption, date splicing, copy-move cloning, font splicing, metadata tampering).

---

## Benchmark Evaluation Results

The system supports two distinct evaluation modes:

### 1. Sidecar-Free Real-OCR Benchmark (`--real-ocr`)
Evaluates the full end-to-end pipeline without reading any sidecar files. All text is extracted dynamically by the EasyOCR engine.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Overall Accuracy** | **80.0%** | Tested across 20 synthetic cards |
| **Precision** | **88.89%** | High precision when flagging tampered cards |
| **Recall** | **72.73%** | 8 of 11 tampered documents detected |
| **F1 Score** | **80.0%** | Balanced harmonic mean |
| **True Positives (TP)** | **8 / 11** | Correctly identified tampering |
| **True Negatives (TN)** | **8 / 9** | Correctly verified genuine cards |
| **False Positives (FP)** | **1 / 9** | Genuine card flagged as suspicious |
| **False Negatives (FN)** | **3 / 11** | Tampered cards with subtle field changes missed by OCR |

### 2. Controlled Synthetic Benchmark (Sidecar Mode)
Used for baseline regression testing against pre-computed character-level alignments:

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **90.0%** |
| **Precision** | **84.62%** |
| **Recall** | **100.0%** |
| **F1 Score** | **91.67%** |
| **Confusion Matrix** | **TP=11, TN=7, FP=2, FN=0** |

> **Important Note on Benchmark Claims**: The 90-100% scores achieved on controlled synthetic test suites represent performance on procedural test templates under synthetic attack vectors. Real-world physical document verification involves lighting variance, print-scan degradation, and camera perspective distortions that will produce different performance profiles.

---

## Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested with Python 3.13)
- Node.js 18+ and npm

### 1. Backend Setup

```powershell
# From project root (d:\Document detector):
pip install -r requirements.txt

# Start the FastAPI service:
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8008
```

> **First-Run Note for EasyOCR**: On first execution, EasyOCR will automatically download CRAFT text detection and recognition model weights (~100 MB) to `~/.EasyOCR/model/`. Subsequent runs use the cached weights.

- API Docs: `http://127.0.0.1:8008/docs`
- Health Endpoint: `http://127.0.0.1:8008/api/health`

### 2. Frontend Setup

```powershell
# From frontend directory:
cd frontend
npm install
npm run dev
```

- Open browser at: `http://localhost:5173/`

### 3. Running Automated Tests

```powershell
# Run the complete test suite (API, Security, OCR Isolation, Forensics):
python -m pytest backend/tests -v
```

### 4. Running the Benchmark

```powershell
# Real OCR Benchmark (sidecar-free, live inference):
python -m backend.scripts.run_benchmark --real-ocr

# Controlled Synthetic Benchmark (sidecar mode):
python -m backend.scripts.run_benchmark
```

---

## API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Health status and OCR engine availability |
| `/api/screen` | `POST` | Screen document via multipart upload or `sample_id` (Real Screening Mode) |
| `/api/samples` | `GET` | Catalog of pre-generated mock documents with thumbnails |
| `/api/benchmark` | `GET` | Executes benchmark evaluation (`?real_ocr=true` supported) |
| `/api/image/{filename}` | `GET` | Serves dataset images (protected by filename whitelist) |

---

## Known Limitations

1. **OCR on Heavily Watermarked Synthetic Cards**: Programmatic diagonal watermarks across synthetic cards occasionally interfere with character segmentation for tiny font sizes.
2. **Physical Print-and-Scan Attacks**: ELA and copy-move forensics operate on digital compression and pixel artifacts; re-photographed physical documents require specialized sensor-noise (PRNU) analysis.
3. **Template Coverage**: Built-in field validation currently targets Indian identity formats (Aadhaar, PAN, Driving License). Additional document specifications can be added via modular validator extensions.
