"""
FastAPI Backend Service for SIH 2026 Problem Statement SIH26188
AI-Based Fake Identity & Document Screening System

Exposes:
- POST /api/screen   (multipart upload or sample ID selection — REAL SCREENING MODE)
- GET  /api/samples   (lists all synthetic genuine & tampered cards with previews)
- GET  /api/benchmark  (runs live evaluation on the 20-card test dataset — BENCHMARK MODE)
- GET  /api/health     (OCR engine status and system health)
- GET  /api/image/{fn} (serves dataset images with path traversal protection)

SECURITY:
- Upload size limit: 10 MB
- Magic byte validation (JPEG, PNG, WebP)
- Decompression bomb protection
- Path traversal protection on all user-supplied paths
- Configurable CORS origins
- Structured error responses (never expose stack traces)
"""

import os
import io
import json
import base64
import tempfile
import traceback
from typing import Optional

import cv2
from PIL import Image as PILImage
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.fusion import DocumentScreeningPipeline
from backend.nlp.ocr_engine import OCRUnavailableError, get_ocr_status
from backend.scripts.run_benchmark import run_benchmark

# ---------------------------------------------------------------------------
# Decompression bomb protection
# ---------------------------------------------------------------------------
PILImage.MAX_IMAGE_PIXELS = 67_000_000  # ~8192x8192

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION = 8192  # Max width or height in pixels
ALLOWED_MIME_MAGIC = {
    b'\xff\xd8\xff': 'image/jpeg',        # JPEG
    b'\x89PNG\r\n\x1a\n': 'image/png',    # PNG
}
WEBP_RIFF_HEADER = b'RIFF'
WEBP_MARKER = b'WEBP'

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fake Identity & Document Screening System",
    description="Multimodal NLP & Image Forensics Tampering Detection API (SIH 2026 SIH26188)",
    version="2.0.0"
)

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

if "*" in ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ---------------------------------------------------------------------------
# Data paths & Ground truth whitelist
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GT_PATH = os.path.join(DATA_DIR, "ground_truth.json")

# Load allowed sample IDs at startup for path traversal protection
_ALLOWED_SAMPLE_IDS = set()
if os.path.exists(GT_PATH):
    try:
        with open(GT_PATH, "r", encoding="utf-8") as _f:
            _ALLOWED_SAMPLE_IDS = set(json.load(_f).keys())
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Pipeline (initialized once)
# ---------------------------------------------------------------------------
pipeline = DocumentScreeningPipeline()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_sample_id(sample_id: str) -> str:
    """Validates sample_id against whitelist and returns safe file path."""
    # Must be in the known ground truth keys
    if sample_id not in _ALLOWED_SAMPLE_IDS:
        raise HTTPException(
            status_code=404,
            detail={"error": "SAMPLE_NOT_FOUND", "message": f"Sample '{sample_id}' not found in dataset"}
        )
    # Defense-in-depth: resolve and verify containment
    target = os.path.realpath(os.path.join(DATA_DIR, sample_id))
    data_real = os.path.realpath(DATA_DIR)
    if not target.startswith(data_real):
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_PATH", "message": "Invalid sample path"}
        )
    if not os.path.exists(target):
        raise HTTPException(
            status_code=404,
            detail={"error": "SAMPLE_NOT_FOUND", "message": f"Sample file '{sample_id}' does not exist on disk"}
        )
    return target


def _detect_image_format(content: bytes) -> Optional[str]:
    """Detect image format from magic bytes. Returns MIME type or None."""
    for magic, mime in ALLOWED_MIME_MAGIC.items():
        if content[:len(magic)] == magic:
            return mime
    # Check WebP: RIFF....WEBP
    if (len(content) >= 12
            and content[:4] == WEBP_RIFF_HEADER
            and content[8:12] == WEBP_MARKER):
        return 'image/webp'
    return None


def _validate_uploaded_image(content: bytes, filename: str) -> str:
    """
    Validates uploaded image content.
    Returns the detected MIME type on success.
    Raises HTTPException on failure.
    """
    # 1. Size check
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "FILE_TOO_LARGE",
                "message": f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit ({len(content)} bytes)"
            }
        )

    # 2. Magic byte check
    detected_mime = _detect_image_format(content)
    if detected_mime is None:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "UNSUPPORTED_FORMAT",
                "message": "Only JPEG, PNG, and WebP images are accepted. "
                           "File does not match any supported image format."
            }
        )

    # 3. Try opening with PIL to catch corrupt images & decompression bombs
    try:
        img = PILImage.open(io.BytesIO(content))
        img.verify()  # Verify without fully loading
        # Re-open to get dimensions (verify() closes the image)
        img = PILImage.open(io.BytesIO(content))
        width, height = img.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "IMAGE_TOO_LARGE",
                    "message": f"Image dimensions {width}x{height} exceed maximum {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "CORRUPT_IMAGE",
                "message": f"Image file is corrupt or cannot be processed: {str(e)}"
            }
        )

    return detected_mime


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Frontend Static Files Setup
# ---------------------------------------------------------------------------
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
FRONTEND_INDEX = os.path.join(FRONTEND_DIST, "index.html")
FRONTEND_ASSETS = os.path.join(FRONTEND_DIST, "assets")

if os.path.exists(FRONTEND_ASSETS):
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="assets")


@app.get("/")
def root(request: Request):
    # When accessed from a web browser, serve the interactive React frontend
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header and os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {
        "system": "AI-Based Fake Identity & Document Screening System",
        "problem_statement": "SIH 2026 SIH26188",
        "status": "online",
        "endpoints": ["/api/screen", "/api/samples", "/api/benchmark", "/api/health"]
    }


@app.get("/api/health")
def health_check():
    """Returns system health and OCR engine availability."""
    ocr_status = get_ocr_status()
    return {
        "status": "healthy",
        "ocr": ocr_status,
        "dataset_loaded": len(_ALLOWED_SAMPLE_IDS) > 0,
        "dataset_size": len(_ALLOWED_SAMPLE_IDS)
    }


@app.get("/api/samples")
def get_samples():
    """Returns catalog of pre-generated mock documents (genuine and tampered) with thumbnails."""
    if not os.path.exists(GT_PATH):
        raise HTTPException(status_code=404, detail="Ground truth dataset not found")

    with open(GT_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    samples = []
    for filename, meta in ground_truth.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            continue

        # Generate small thumbnail data URI
        img = cv2.imread(filepath)
        if img is not None:
            thumb = cv2.resize(img, (240, 150))
            _, buf = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
        else:
            thumb_b64 = None

        samples.append({
            "id": filename,
            "filename": filename,
            "label": meta.get("label"),
            "doc_type": meta.get("doc_type"),
            "name": meta.get("name"),
            "tampering_types": meta.get("tampering_types", []),
            "thumbnail_b64": thumb_b64
        })

    return {"samples": samples, "count": len(samples)}


@app.post("/api/screen")
async def screen_document(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None)
):
    """
    Screens an uploaded identity document or pre-selected synthetic sample.
    
    REAL SCREENING MODE: benchmark_mode is ALWAYS False here.
    The pipeline will NEVER use sidecar OCR or ground truth data.
    
    Returns:
    - Authenticity score (0-100%)
    - Categorical verdict (AUTHENTIC, SUSPICIOUS, FLAGGED / TAMPERED)
    - Unified flagged bounding box coordinates
    - Forensic layer breakdown (NLP, ELA, Font Analysis, Copy-Move, Metadata)
    """
    temp_path = None
    try:
        if file and file.filename:
            # Read and validate uploaded file
            content = await file.read()
            detected_mime = _validate_uploaded_image(content, file.filename)

            # Determine safe file extension from detected MIME
            ext_map = {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/webp': '.webp',
            }
            suffix = ext_map.get(detected_mime, '.png')

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                temp_path = tmp.name
            target_path = temp_path
            original_filename = file.filename

        elif sample_id:
            # Validate sample_id against whitelist (path traversal protection)
            target_path = _validate_sample_id(sample_id)
            original_filename = sample_id
        else:
            raise HTTPException(
                status_code=400,
                detail={"error": "MISSING_INPUT", "message": "Must provide either 'file' or 'sample_id'"}
            )

        # Execute screening pipeline — ALWAYS in real screening mode
        # benchmark_mode=False means OCR sidecar/ground truth is NEVER accessed
        result = pipeline.screen_document(target_path, benchmark_mode=False)
        result["filename"] = original_filename

        # Also provide base64 of original document for the canvas viewer
        with open(target_path, "rb") as f_img:
            b64_orig = base64.b64encode(f_img.read()).decode('utf-8')
            mime = "image/jpeg" if target_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            result["original_image_data_uri"] = f"data:{mime};base64,{b64_orig}"

        return JSONResponse(content=result)

    except OCRUnavailableError as e:
        return JSONResponse(
            status_code=503,
            content={
                "error": "OCR_UNAVAILABLE",
                "message": str(e)
            }
        )
    except HTTPException:
        raise
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_IMAGE",
                "message": str(e)
            }
        )
    except Exception as e:
        # Never expose raw stack traces to the client
        print(f"[SCREEN ERROR] {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "SCREENING_FAILED",
                "message": "An internal error occurred during document screening. Please try again."
            }
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.get("/api/benchmark")
def get_benchmark(real_ocr: bool = Query(default=False)):
    """
    Executes dataset benchmark evaluation and returns precision, recall, and matrix.
    
    Args:
        real_ocr: If True, uses real OCR only (no sidecar). If False (default),
                  allows sidecar OCR for controlled benchmark testing.
    """
    try:
        return run_benchmark(use_real_ocr=real_ocr)
    except Exception as e:
        print(f"[BENCHMARK ERROR] {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "BENCHMARK_FAILED",
                "message": f"Benchmark evaluation failed: {str(e)}"
            }
        )


@app.get("/api/benchmark/batch")
@app.post("/api/benchmark/batch")
def get_batch_benchmark(rerun: bool = Query(default=False)):
    r"""
    Executes or returns cached results of the automated batch test on user datasets
    (D:\original test data and D:\fake data).
    """
    report_file = os.path.join("reports", "batch_test", "batch_results.json")
    if not rerun and os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        from backend.scripts.run_batch_test import run_batch_test
        return run_batch_test(real_ocr=True)
    except Exception as e:
        print(f"[BATCH BENCHMARK ERROR] {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "BATCH_BENCHMARK_FAILED",
                "message": f"Batch benchmark failed: {str(e)}"
            }
        )


@app.get("/api/image/{filename}")
def get_image(filename: str):
    """Serves a dataset image file directly. Protected against path traversal."""
    # Whitelist validation
    if filename not in _ALLOWED_SAMPLE_IDS:
        raise HTTPException(
            status_code=404,
            detail={"error": "SAMPLE_NOT_FOUND", "message": "Image not found"}
        )
    # Defense-in-depth containment check
    path = os.path.realpath(os.path.join(DATA_DIR, filename))
    data_real = os.path.realpath(DATA_DIR)
    if not path.startswith(data_real) or not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail={"error": "SAMPLE_NOT_FOUND", "message": "Image not found"}
        )
    return FileResponse(path)
