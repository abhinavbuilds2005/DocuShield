---
title: DocuShield AI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DocuShield AI: AI-Based Fake Identity & Document Screening System
**Smart India Hackathon (SIH 2026) — Problem Statement SIH26188**

DocuShield AI is an enterprise-grade, multimodal forensic screening platform built to verify document authenticity, extract key identity attributes, detect digital tampering, and validate biometric credentials across five core identity document categories.

> **Honesty & Forensic Disclaimer**:
> 1. **Rule-Based Validation**: All field, pattern, and checksum evaluations are performed locally using deterministic rule sets. **No external government database verification is performed or simulated.**
> 2. **Transparent Model Architecture**: Document classification currently relies on layout heuristics, ICAO MRZ signatures, and keyword matching. Future ML model training pipelines are scaffolded in `training/` with reproducible training scripts ready for real-world datasets.
> 3. **Non-Infallible Decision Support**: This system assesses forensic risk and presents explainable corroborating evidence to empower human border/immigration and verification officers; it does not claim autonomous infallibility.

---

## 1. Supported Document Categories

DocuShield provides dedicated schema parsing, structural validation, and classification for:

1. **Passport**:
   - ICAO Doc 9303 TD3 (2×44) and TD1/TD2 Machine Readable Zone (MRZ) parser.
   - 7-3-1 weighted modulus 10 check digit verification for Document Number, Date of Birth, Date of Expiry, and Composite check digits.
   - Cross-consistency verification between MRZ and visible OCR text.
2. **Visa**:
   - Extraction and validation of Visa Number, Holder Name, Nationality, Date of Birth, Visa Type, Issue Date, Expiry Date, Entries, Duration of Stay, and Issuing Authority.
3. **National Identity Card**:
   - Indian 12-digit Aadhaar / UID with strict Verhoeff checksum algorithm (D5 dihedral group).
   - Tax ID / PAN alphanumeric structure validation (entity type and surname initial consistency).
   - Citizen ID generic layout parsing with age range plausibility checks.
4. **Driving Licence**:
   - Jurisdictional state/union code verification (e.g., `DL`, `MH`, `KA`, `TN`, `UP`).
   - Vehicle class authorization extraction (LMV, MCWG, HMV).
   - Plausible issue-to-expiry validity span verification.
5. **Permit / Travel Authorization**:
   - Permit identifier validation, nationality, travel validity windows, permit purpose, and issuing authority.

---

## 2. Core Architectural Modules

```
                           Uploaded Document (+ Optional Selfie)
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      ▼                                      ▼                                      ▼
[Module 1: OCR Extraction]      [Module 3: Tampering Detection]       [Module 4: Face Verification]
• Document Classifier (Auto/Manual) • Error Level Analysis (ELA)          • Document Face Detection (Haar)
• EasyOCR / Tesseract Fallback      • Copy-Move (ORB + RANSAC)            • Live Person Face Detection
• Normalized & Bounding Boxes       • Typography & Laplacian Variance     • HSV/Gradient Similarity Score
• Document-Specific JSON Schemas    • EXIF Software Signatures (Adobe)    • Graceful "Not Performed" Handling
      │                             • Condition Analyzer (Blur/Quality)                     │
      ▼                                      │                                             │
[Module 2: Document Validation]              │                                             │
• ICAO Doc 9303 MRZ Check Digits             │                                             │
• Date Chronology (DOB < Issue < Expiry)     │                                             │
• Algorithmic Checksum (Verhoeff for UID)    │                                             │
• Rule-Based Disclaimer Notice               │                                             │
      │                                      │                                             │
      └──────────────────────────────────────┼─────────────────────────────────────────────┘
                                             ▼
                             [Multimodal Evidence Fusion Engine]
                              • 5-Level Hierarchical Evidence Fusion
                              • Multi-Family Corroboration (Text, Image, Identity)
                              • Unified Risk Score [0 - 100%] & Calibrated Verdict:
                                [AUTHENTIC | SUSPICIOUS | FLAGGED / TAMPERED]
                              • Explainable Evidence Item List for Human Reviewers
                                             │
                                             ▼
                             [Interactive Forensic UI Dashboard]
                              • React 18 + Vite Canvas Inspector
                              • Document Category Dropdown (Auto-Detect / Manual)
                              • Live Person Selfie Uploader & Biometric Card
                              • Document-Specific Field Extraction Table
                              • Interactive Forensic Canvas with SVG Overlays
```

### Module 1 — OCR Extraction (`backend/nlp/`)
- **Document Classifier** (`document_classifier.py`): Automatically classifies inputs into `passport`, `visa`, `national_id`, `driving_license`, or `permit` using MRZ signatures, structural keywords, or user selection overrides.
- **OCR Engine** (`ocr_engine.py`): Singleton running EasyOCR with Tesseract fallback. Extracts raw text, word tokens, confidences, and normalized bounding boxes `[0, 1]`.
- **Field Schemas**: Standardized `{ "value": "...", "confidence": float, "status": "valid|invalid|unknown" }` schema mappings.

### Module 2 — Document Validation (`backend/nlp/`)
- **MRZ Parser** (`mrz_parser.py`): Fully compliant ICAO 9303 TD3 parser. Computes and checks:
  $$\text{Check Digit} = \left( \sum_{i} \text{char\_val}(c_i) \times w_i \right) \pmod{10}, \quad w \in \{7, 3, 1\}$$
- **Date Chronology**: Verifies calendar syntax, plausible age ranges (1920–present), and ensures `DOB < Issue Date < Expiry Date`.
- **Selective Checksum Policy**: Verhoeff checksum algorithm is strictly isolated to 12-digit Indian National ID cards; it is never erroneously applied to passports, visas, or driving licences.

### Module 3 — Tampering Detection (`backend/forensics/`)
- **Error Level Analysis (ELA)** (`ela.py`): Measures localized JPEG recompression error at 90% quality to detect cut-and-paste seams, font splicing, and compression discrepancies.
- **Copy-Move Forgery Detection** (`copy_move.py`): Extracts ORB descriptors, eliminates self-matches, clusters spatial displacement vectors, and performs RANSAC homography estimation to detect cloned stamps, graphics, or text blocks.
- **Typography & Font Alignment** (`font_alignment.py`): Analyzes stroke sharpness via local Laplacian variance, edge gradients, and token confidence disparities.
- **Metadata Forensics** (`metadata_checker.py`): Parses EXIF markers and detects traces of image manipulation software (Adobe Photoshop, GIMP, Canva).
- **Document Condition Analyzer** (`condition_analyzer.py`): Quantifies image resolution, blur (Laplacian variance), and JPEG quality metrics.

### Module 4 — Face Verification (`backend/forensics/face_verifier.py`)
- Detects the portrait in the document using OpenCV cascade models and extracts the cropped face thumbnail.
- If a live person / selfie image is supplied, detects and aligns the person's face.
- Computes structural similarity, multi-channel HSV color histogram correlation, and gradient orientation similarity.
- Returns match confidence, similarity percentage [0–100%], and side-by-side cropped face previews.
- **Graceful Handling**: If no live person photo is provided, reports:
  `"Face verification not performed — person image not provided."` (never assumes or claims verification).

---

## 3. Explainable Evidence Fusion

DocuShield organizes all signals into a structured 5-level hierarchy across three independent evidence families:
1. **Compression Family**: ELA error levels, DCT grid uniformity, JPEG quality index.
2. **Structural / Visual Family**: Cut seams, copy-move feature clusters, typography variance.
3. **Content & Identity Family**: ICAO MRZ check digits, Verhoeff checksum, date chronology, OCR confidence disparities, and facial portrait similarity.

### Standardized Verdict Thresholds:
- **`AUTHENTIC`** (Authenticity Score $\ge 80\%$ / Risk Score $\le 20\%$): All check digits valid, uniform compression, authentic typography, no tamper indicators.
- **`SUSPICIOUS`** (Authenticity Score $50\% - 79\%$ / Risk Score $21\% - 50\%$): Isolated anomaly detected (e.g., low-resolution artifact, isolated font variance, expired document). Recommended for secondary manual review.
- **`FLAGGED / TAMPERED`** (Authenticity Score $< 50\%$ / Risk Score $> 50\%$): Multi-family corroborated tampering, verified cut seam, corrupted check digits, or face mismatch.

---

## 4. Dataset Management & Synthetic Testbed

DocuShield enforces strict privacy ethics: **no real government identity cards or citizen PII are collected or stored**.

### Directory Structure (`datasets/`)
```
datasets/
├── README.md               # Privacy ethics, citations & data governance
├── authentic/              # Watermarked synthetic authentic cards
├── tampered/               # Watermarked synthetic tampered cards
├── passport/               # Sample passport test templates
├── visa/                   # Sample visa test templates
├── national_id/            # Sample national identity card templates
├── driving_license/        # Sample driving licence test templates
├── permit/                 # Sample travel authorization permit templates
├── raw/                    # Raw templates
├── processed/              # Preprocessed normalized assets
└── annotations/            # Ground truth bounding boxes & metadata
```

### Generating the Controlled Synthetic Testbed
Run the automated testbed generator to produce watermarked synthetic test samples for all 5 document categories:
```powershell
python backend/scripts/generate_synthetic_testbed.py
```
Every generated image carries explicit, mandatory watermarks:
- `"SAMPLE / MOCK — NOT A REAL GOVERNMENT DOCUMENT"`
- `"MOCK / SPECIMEN — NOT REAL ID"`

---

## 5. Future ML Training Framework (`training/`)

For future scalability with real-world research datasets, a modular training pipeline is scaffolded in `training/`:
- `dataset_loader.py`: Document dataset loader supporting PyTorch and NumPy tensors with train/val/test splits.
- `preprocessing.py`: Multi-resolution resizing, CLAHE contrast enhancement, and noise normalization.
- `augmentation.py`: Realistic document augmentations (perspective tilt, blur, JPEG recompression, shadow gradients).
- `train.py`: Training engine with learning rate scheduling, early stopping, and checkpoint saving.
- `evaluate.py`: Evaluation pipeline tracking Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix.
- `inference.py`: Standalone inference wrapper for deployment.

---

## 6. API Reference

| Endpoint | Method | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | — | System health, OCR engine availability, and dataset status |
| `/api/document-types` | `GET` | — | Supported document categories and their required schema fields |
| `/api/screen` | `POST` | `file`, `document_type`, `person_image`, `sample_id` | Full multimodal screening pipeline execution |
| `/api/samples` | `GET` | — | Catalog of pre-generated mock documents with thumbnails |
| `/api/benchmark` | `GET` | `real_ocr: bool` | Automated benchmark evaluation across testbed |
| `/api/image/{filename}` | `GET` | `filename: str` | Serves dataset images with path-traversal protection |

---

## 7. Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ and npm

### 1. Backend Service
```powershell
# From project root:
pip install -r requirements.txt

# Start FastAPI server on port 8000:
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
- Interactive API Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/health`

### 2. Frontend Dashboard
```powershell
# From frontend directory:
cd frontend
npm install
npm run dev
```
- Dashboard runs at: `http://localhost:5173/`

### 3. Production Build
```powershell
cd frontend
npm run build
```
Builds optimized production assets into `frontend/dist/`.

### 4. Running Automated Tests
```powershell
# Run the complete test suite (90 passing tests):
python -m pytest backend/tests -v
```

---

## 8. Deploying to Render

DocuShield AI includes ready-to-deploy configuration for [Render](https://render.com) using Docker:

### Option A: One-Click Blueprint Deployment (Recommended)
1. Push your repository to **GitHub** or **GitLab**.
2. Go to your [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Blueprint**.
3. Connect your repository. Render will automatically detect [`render.yaml`](file:///d:/Document%20detector/render.yaml) and configure:
   - **Service Type**: Web Service
   - **Environment**: Docker (multi-stage build with CPU PyTorch and Node 20)
   - **Port**: Bound dynamically to `$PORT` (default 10000)
   - **Health Check**: `/api/health`
4. Click **Apply**. Render will build and deploy the container.

### Option B: Manual Web Service Setup
1. On the Render Dashboard, click **New +** → **Web Service**.
2. Select your repository.
3. Configure the following settings:
   - **Name**: `docushield-ai` (or your choice)
   - **Region**: Oregon (or nearest)
   - **Branch**: `main`
   - **Runtime**: **Docker**
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Context**: `.`
   - **Instance Type**: Free (or Starter/Standard)
4. Under **Advanced Settings**:
   - **Health Check Path**: `/api/health`
   - **Environment Variables**:
     - `PORT`: `10000`
     - `ALLOWED_ORIGINS`: `*`
5. Click **Create Web Service**.

> [!TIP]
> **Free Tier Cold Starts**: On Render's Free tier, the service spins down after 15 minutes of inactivity and wakes up upon incoming requests (usually takes ~30-50 seconds to spin up). The Dockerfile pre-caches EasyOCR weights at build time so screening documents has zero download delay once online.

---

## 9. Real-World Limitations

1. **Physical Print-and-Scan Limitations**: ELA and copy-move forensics inspect digital compression and pixel-level gradients. High-resolution re-photographed physical printouts may require physical security feature analysis (UV luminescence, hologram reflection, guilloche pattern integrity).
2. **Camera Lighting & Glare**: Heavy glare or extreme shadows on plastic laminate cards can degrade OCR token confidence or trigger localized false positives in Laplacian sharpness maps.
3. **No Database Verification**: DocuShield verifies mathematical, grammatical, and forensic structural authenticity. A fabricated identity card containing completely plausible but unregistered information cannot be invalidated without authoritative national registry access.

