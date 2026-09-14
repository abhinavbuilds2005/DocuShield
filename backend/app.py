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
        "*"
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


@app.get("/api/document-types")
def get_document_types():
    """Returns supported document types and their human-readable labels."""
    from backend.nlp.document_classifier import DOCUMENT_TYPES, DOCUMENT_TYPE_LABELS
    return {
        "supported_types": [
            {
                "id": dt,
                "label": DOCUMENT_TYPE_LABELS.get(dt, dt),
                "is_mrz_capable": dt in ["passport", "visa"]
            }
            for dt in DOCUMENT_TYPES
        ],
        "default": "auto"
    }


@app.post("/api/screen")
async def screen_document(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    document_type: Optional[str] = Form(None),
    person_image: Optional[UploadFile] = File(None)
):
    """
    Screens an uploaded identity document or pre-selected synthetic sample.
    
    Supports:
    - Document categories: passport, visa, national_id, driving_license, permit (or auto-detect)
    - Optional biometric face verification when person_image is supplied
    - Full explainable evidence fusion across NLP, ELA, Typography, Copy-Move, Metadata, Face
    
    Returns:
    - Authenticity score (0-100%)
    - Risk score & verdict (AUTHENTIC, SUSPICIOUS, FLAGGED / TAMPERED)
    - Document classification & structured OCR schema fields
    - Unified flagged bounding box coordinates
    - Face verification report (or 'not performed' notice)
    """
    temp_path = None
    person_temp_path = None
    try:
        # 1. Process document image
        if file and file.filename:
            content = await file.read()
            detected_mime = _validate_uploaded_image(content, file.filename)

            ext_map = {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/webp': '.webp',
            }
            suffix = ext_map.get(detected_mime, '.png')

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                temp_path = tmp.name

            # Safe downscaling to prevent memory exhaustion in 512MB RAM
            try:
                with PILImage.open(temp_path) as p_img:
                    cur_w, cur_h = p_img.size
                    max_dim = max(cur_w, cur_h)
                    if max_dim > 1600:
                        ratio = 1600.0 / max_dim
                        new_size = (max(1, int(cur_w * ratio)), max(1, int(cur_h * ratio)))
                        resample_filter = getattr(PILImage, "Resampling", PILImage).LANCZOS
                        resized_img = p_img.resize(new_size, resample=resample_filter)
                        if p_img.mode in ("RGBA", "P") and suffix.lower() in (".jpg", ".jpeg"):
                            resized_img = resized_img.convert("RGB")
                        resized_img.save(temp_path)
            except Exception as resize_err:
                print(f"[RESIZE WARNING] Could not downscale uploaded image: {resize_err}")

            target_path = temp_path
            original_filename = file.filename

        elif sample_id:
            target_path = _validate_sample_id(sample_id)
            original_filename = sample_id
        else:
            raise HTTPException(
                status_code=400,
                detail={"error": "MISSING_INPUT", "message": "Must provide either 'file' or 'sample_id'"}
            )

        # 2. Process optional live person/selfie image
        if person_image and person_image.filename:
            p_content = await person_image.read()
            p_mime = _validate_uploaded_image(p_content, person_image.filename)
            p_ext = {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/webp': '.webp',
            }.get(p_mime, '.png')

            with tempfile.NamedTemporaryFile(delete=False, suffix=p_ext) as p_tmp:
                p_tmp.write(p_content)
                person_temp_path = p_tmp.name

            try:
                with PILImage.open(person_temp_path) as p_img:
                    cur_w, cur_h = p_img.size
                    max_dim = max(cur_w, cur_h)
                    if max_dim > 800:
                        ratio = 800.0 / max_dim
                        new_size = (max(1, int(cur_w * ratio)), max(1, int(cur_h * ratio)))
                        resample_filter = getattr(PILImage, "Resampling", PILImage).LANCZOS
                        resized_img = p_img.resize(new_size, resample=resample_filter)
                        if p_img.mode in ("RGBA", "P") and p_ext.lower() in (".jpg", ".jpeg"):
                            resized_img = resized_img.convert("RGB")
                        resized_img.save(person_temp_path)
            except Exception as resize_err:
                print(f"[RESIZE WARNING] Could not downscale person image: {resize_err}")

        # Clean document_type
        clean_doc_type = document_type.strip().lower() if document_type and document_type.strip().lower() != "auto" else None

        # Execute screening pipeline
        result = pipeline.screen_document(
            target_path,
            benchmark_mode=False,
            document_type=clean_doc_type,
            person_image_path=person_temp_path
        )
        result["filename"] = original_filename

        # Provide base64 of original document for canvas viewer
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
        print(f"[SCREEN ERROR] Internal error occurred during document screening:\n{traceback.format_exc()}")
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
        if person_temp_path and os.path.exists(person_temp_path):
            try:
                os.remove(person_temp_path)
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


@app.get("/api/download/pptx")
@app.get("/download/pptx")
def download_pptx():
    """Download the final SIH 2026 presentation in PowerPoint (.pptx) format."""
    file_path = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "DocuShield_AI_SIH2026_Final_Presentation.pptx"))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Presentation file not found.")
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="DocuShield_AI_SIH2026_Final_Presentation.pptx"
    )


@app.get("/api/download/pdf")
@app.get("/download/pdf")
def download_pdf():
    """Download the final SIH 2026 presentation in PDF format."""
    file_path = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "DocuShield_AI_SIH2026_Final_Presentation.pdf"))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Presentation PDF not found.")
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename="DocuShield_AI_SIH2026_Final_Presentation.pdf"
    )


@app.get("/api/download/reference-template")
@app.get("/download/reference-template")
def download_reference_template():
    """Download the uploaded original reference SIH presentation template."""
    file_path = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploaded_reference_sih_template.pdf"))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Reference template file not found.")
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename="uploaded_reference_sih_template.pdf"
    )


@app.get("/{full_path:path}")
def serve_spa_catchall(full_path: str, request: Request):
    """Fallback handler to serve static frontend files or index.html for client-side routing."""
    # Never intercept API or documentation routes
    if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json"):
        raise HTTPException(status_code=404, detail="Endpoint not found")

    # Serve static assets from FRONTEND_DIST if safe and existent
    static_file = os.path.realpath(os.path.join(FRONTEND_DIST, full_path))
    dist_real = os.path.realpath(FRONTEND_DIST)
    if static_file.startswith(dist_real) and os.path.isfile(static_file):
        return FileResponse(static_file)

    # Only return index.html for browser page navigation
    accept = request.headers.get("accept", "")
    if "text/html" in accept and os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)

    raise HTTPException(status_code=404, detail="Not found")




