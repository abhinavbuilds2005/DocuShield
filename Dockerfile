# ==============================================================================
# DocuShield AI — Hugging Face Spaces Multi-Stage Dockerfile
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
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for OpenCV, PyTorch, and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download EasyOCR neural network weights (~100MB) during build time
# This eliminates startup latency and ensures offline reliability on Hugging Face
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy built frontend static bundle
COPY --chown=user:user --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy backend application source code and mock dataset
COPY --chown=user:user backend/ ./backend/

# Hugging Face Spaces listens on port 7860 by default
EXPOSE 7860

# Start FastAPI application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
