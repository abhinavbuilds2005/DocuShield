# ==============================================================================
# DocuShield AI — Production Multi-Stage Dockerfile (Render & HF Spaces Compatible)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build React Frontend
# ------------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Python Runtime & Production Server
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=10000

# Install system dependencies required for OpenCV, PyTorch, and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security and platform compatibility (HF Spaces UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Pre-install CPU-only PyTorch first to save ~1.5GB of container image size & build time
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download EasyOCR neural network weights (~100MB) during build time
# Eliminates startup latency and ensures offline reliability on cloud hosts
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy built frontend static bundle
COPY --chown=user:user --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy backend application source code and dataset
COPY --chown=user:user backend/ ./backend/

# Copy benchmark reports and presentation materials
COPY --chown=user:user reports/ ./reports/
COPY --chown=user:user DocuShield_AI_SIH2026_Final_Presentation.* ./
COPY --chown=user:user uploaded_reference_sih_template.pdf ./

# Render web services listen on 10000 by default; HF Spaces on 7860
EXPOSE 10000 7860

# Start FastAPI application dynamically binding to Render's $PORT (or fallback to 10000)
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-10000}"]

