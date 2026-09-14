"""
Robust OCR Engine Module — Hardened for Real Screening
Provides an interface for text extraction from identity cards.

CRITICAL DESIGN:
- REAL SCREENING MODE (benchmark_mode=False):
    Uses ONLY genuine OCR engines (EasyOCR primary, Tesseract fallback).
    NEVER reads .ocr.json sidecars, ground_truth.json, or any pre-baked text.
    If no OCR engine is available, raises OCRUnavailableError.
    
- BENCHMARK MODE (benchmark_mode=True):
    May use sidecar .ocr.json files for controlled synthetic testing.
    Falls back to real OCR if sidecars are not found.

Features:
1. Singleton EasyOCR reader (loaded once, reused)
2. Bounding box and per-token confidence extraction
3. Normalized bounding box coordinates (x, y, width, height in [0,1])
4. Explicit OCR health status reporting
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Custom exception — raised when no OCR engine is available in real mode
# ---------------------------------------------------------------------------
class OCRUnavailableError(Exception):
    """Raised when no OCR engine is available for real document screening."""
    pass


# ---------------------------------------------------------------------------
# Singleton EasyOCR Reader (lazy loaded on first use to stay within 512MB RAM)
# ---------------------------------------------------------------------------
_EASYOCR_READER = None
_EASYOCR_AVAILABLE = None
_EASYOCR_INIT_ATTEMPTED = False

_TESSERACT_AVAILABLE = False
_TESSERACT_INIT_ATTEMPTED = False


def _can_import_easyocr() -> bool:
    """Check if easyocr library is installed without loading model weights."""
    try:
        import easyocr
        return True
    except Exception:
        return False


def _init_easyocr():
    """Attempt to initialize EasyOCR reader on demand."""
    global _EASYOCR_READER, _EASYOCR_AVAILABLE, _EASYOCR_INIT_ATTEMPTED
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    _EASYOCR_INIT_ATTEMPTED = True

    # Memory optimization for 512MB container environments
    try:
        import torch
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        import easyocr
        user_home = os.path.expanduser("~")
        model_dir = os.path.join(user_home, ".EasyOCR", "model")
        craft_file = os.path.join(model_dir, "craft_mlt_25k.pth")
        has_weights = (
            os.path.exists(model_dir)
            and os.path.exists(craft_file)
            and os.path.getsize(craft_file) > 1_000_000
        )
        _EASYOCR_READER = easyocr.Reader(
            ['en'],
            gpu=False,
            verbose=False,
            download_enabled=not has_weights
        )
        _EASYOCR_AVAILABLE = True
        print("[OCR] EasyOCR initialized successfully (lazy loaded)")
        return _EASYOCR_READER
    except Exception as e:
        print(f"[OCR] EasyOCR initialization failed: {e}")
        _EASYOCR_READER = None
        _EASYOCR_AVAILABLE = False
        return None


def _init_tesseract():
    """Attempt to verify Tesseract availability exactly once."""
    global _TESSERACT_AVAILABLE, _TESSERACT_INIT_ATTEMPTED
    if _TESSERACT_INIT_ATTEMPTED:
        return
    _TESSERACT_INIT_ATTEMPTED = True
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
        print("[OCR] Tesseract fallback available")
    except Exception as e:
        _TESSERACT_AVAILABLE = False


# Lightweight initialization at module load (Tesseract only; EasyOCR is lazy-loaded)
_init_tesseract()


# ---------------------------------------------------------------------------
# Public health check
# ---------------------------------------------------------------------------
def get_ocr_status() -> Dict[str, Any]:
    """Returns availability status without forcing heavy neural weights into RAM."""
    easyocr_ok = (_EASYOCR_AVAILABLE if _EASYOCR_AVAILABLE is not None else _can_import_easyocr())
    tesseract_ok = _TESSERACT_AVAILABLE
    pref = os.environ.get("OCR_ENGINE", "auto").strip().lower()
    use_tess_primary = (pref == "tesseract") or (pref == "auto" and (os.environ.get("RENDER") == "true" or tesseract_ok))
    primary = "tesseract" if (use_tess_primary and tesseract_ok) else ("easyocr" if easyocr_ok else ("tesseract" if tesseract_ok else None))
    return {
        "ocr_available": easyocr_ok or tesseract_ok,
        "easyocr_available": easyocr_ok,
        "tesseract_available": tesseract_ok,
        "primary_engine": primary,
    }


# ---------------------------------------------------------------------------
# OCR Engine
# ---------------------------------------------------------------------------
class OCREngine:
    """Extracts text tokens, confidence, and bounding boxes from ID documents.
    
    IMPORTANT: In real screening mode (benchmark_mode=False), this engine
    ONLY uses genuine OCR. It NEVER reads .ocr.json sidecars or ground truth.
    """

    def __init__(self):
        self.tesseract_available = _TESSERACT_AVAILABLE

    @property
    def easyocr_reader(self):
        global _EASYOCR_READER
        if _EASYOCR_READER is None:
            return _init_easyocr()
        return _EASYOCR_READER

    @property
    def easyocr_available(self) -> bool:
        global _EASYOCR_AVAILABLE
        if _EASYOCR_AVAILABLE is None:
            return _can_import_easyocr()
        return _EASYOCR_AVAILABLE

    def process_image(self, image_path_or_array, benchmark_mode: bool = False) -> Dict[str, Any]:
        """
        Extracts text, bounding boxes, and confidence from a document image.

        Args:
            image_path_or_array: File path (str) or BGR numpy array
            benchmark_mode: If False (DEFAULT), ONLY uses real OCR engines.
                           If True, may use sidecar OCR for benchmark testing.

        Returns:
            Dict with keys: full_text, tokens, lines, image_dims

        Raises:
            OCRUnavailableError: In real mode when no OCR engine is available
            ValueError: When image cannot be loaded
        """
        image_path = None
        if isinstance(image_path_or_array, str):
            image_path = image_path_or_array
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image at {image_path}")
        else:
            img = image_path_or_array

        h, w = img.shape[:2]

        # ─── BENCHMARK MODE: Allow sidecar OCR ───────────────────────
        if benchmark_mode and image_path:
            sidecar_result = self._try_sidecar_ocr(image_path, w, h)
            if sidecar_result is not None:
                return sidecar_result

        # ─── REAL OCR (used in both modes) ────────────────────────────
        pref = os.environ.get("OCR_ENGINE", "auto").strip().lower()
        use_tesseract_first = (pref == "tesseract") or (pref == "auto" and (os.environ.get("RENDER") == "true" or self.tesseract_available))

        # Try Tesseract first if preferred or in memory-constrained cloud environments (~25MB RAM)
        if use_tesseract_first and self.tesseract_available:
            result = self._run_tesseract(img, w, h)
            if result is not None:
                return result

        # Try EasyOCR
        if self.easyocr_available and self.easyocr_reader is not None:
            result = self._run_easyocr(img, w, h)
            if result is not None:
                return result

        # Fallback to Tesseract if not already tried
        if not use_tesseract_first and self.tesseract_available:
            result = self._run_tesseract(img, w, h)
            if result is not None:
                return result

        # ─── NO OCR AVAILABLE ─────────────────────────────────────────
        if not benchmark_mode:
            # REAL SCREENING: Hard fail — never return fake data
            raise OCRUnavailableError(
                "No OCR engine is available. Install EasyOCR (pip install easyocr) "
                "or Tesseract (pip install pytesseract + system binary). "
                "Real document screening requires a genuine OCR engine."
            )
        else:
            # BENCHMARK MODE: Return empty result (benchmark will record failure)
            return {
                "full_text": "",
                "tokens": [],
                "lines": [],
                "image_dims": [w, h],
                "ocr_engine": "none",
                "ocr_warning": "No OCR engine available; benchmark result may be degraded"
            }

    def _run_easyocr(self, img: np.ndarray, w: int, h: int) -> Optional[Dict[str, Any]]:
        """Run EasyOCR on the image. Returns None on failure."""
        try:
            if len(img.shape) == 3 and img.shape[2] == 3:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = img
            results = self.easyocr_reader.readtext(
                rgb_img,
                batch_size=1,
                workers=0,
                canvas_size=1600,
                mag_ratio=1.0
            )
            lines = []
            tokens = []
            full_parts = []

            for bbox, text, conf in results:
                t_str = str(text).strip()
                if not t_str:
                    continue

                pts = np.array(bbox, dtype=np.int32)
                bx = int(np.min(pts[:, 0]))
                by = int(np.min(pts[:, 1]))
                bw = int(np.max(pts[:, 0]) - bx)
                bh = int(np.max(pts[:, 1]) - by)

                # Pixel-coordinate box (backward compatible)
                pixel_box = [bx, by, bw, bh]
                # Normalized coordinates
                norm_box = {
                    "x": round(bx / w, 4),
                    "y": round(by / h, 4),
                    "width": round(bw / w, 4),
                    "height": round(bh / h, 4)
                }

                lines.append({
                    "text": t_str,
                    "confidence": round(float(conf), 3),
                    "box": pixel_box,
                    "bbox_normalized": norm_box
                })
                full_parts.append(t_str)

                # Split into word-level tokens
                words = t_str.split()
                if words:
                    avg_w = max(1, bw // len(words))
                    for idx, word in enumerate(words):
                        tok_bx = bx + (idx * avg_w)
                        tokens.append({
                            "text": word,
                            "confidence": round(float(conf), 3),
                            "box": [tok_bx, by, avg_w, bh],
                            "bbox_normalized": {
                                "x": round(tok_bx / w, 4),
                                "y": round(by / h, 4),
                                "width": round(avg_w / w, 4),
                                "height": round(bh / h, 4)
                            }
                        })

            return {
                "full_text": " ".join(full_parts),
                "tokens": tokens,
                "lines": lines,
                "image_dims": [w, h],
                "ocr_engine": "easyocr"
            }
        except Exception as e:
            print(f"[OCR] EasyOCR processing error: {e}")
            return None
        finally:
            try:
                import gc
                gc.collect()
            except Exception:
                pass

    def _run_tesseract(self, img: np.ndarray, w: int, h: int) -> Optional[Dict[str, Any]]:
        """Run Tesseract OCR on the image. Returns None on failure."""
        try:
            import pytesseract
            # Get detailed word-level data
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            lines_dict = {}  # group by block+line
            tokens = []
            full_parts = []

            n_items = len(data['text'])
            for i in range(n_items):
                text = str(data['text'][i]).strip()
                conf = float(data['conf'][i])
                if not text or conf < 0:
                    continue

                bx = int(data['left'][i])
                by = int(data['top'][i])
                bw = int(data['width'][i])
                bh = int(data['height'][i])
                conf_norm = conf / 100.0

                pixel_box = [bx, by, bw, bh]
                norm_box = {
                    "x": round(bx / w, 4),
                    "y": round(by / h, 4),
                    "width": round(bw / w, 4),
                    "height": round(bh / h, 4)
                }

                tokens.append({
                    "text": text,
                    "confidence": round(conf_norm, 3),
                    "box": pixel_box,
                    "bbox_normalized": norm_box
                })
                full_parts.append(text)

                # Group into lines
                line_key = (data['block_num'][i], data['line_num'][i])
                if line_key not in lines_dict:
                    lines_dict[line_key] = {
                        "parts": [], "confs": [],
                        "min_x": bx, "min_y": by,
                        "max_x": bx + bw, "max_y": by + bh
                    }
                ld = lines_dict[line_key]
                ld["parts"].append(text)
                ld["confs"].append(conf_norm)
                ld["min_x"] = min(ld["min_x"], bx)
                ld["min_y"] = min(ld["min_y"], by)
                ld["max_x"] = max(ld["max_x"], bx + bw)
                ld["max_y"] = max(ld["max_y"], by + bh)

            lines = []
            for lk, ld in lines_dict.items():
                line_text = " ".join(ld["parts"])
                avg_conf = sum(ld["confs"]) / len(ld["confs"]) if ld["confs"] else 0
                lx, ly = ld["min_x"], ld["min_y"]
                lw = ld["max_x"] - lx
                lh = ld["max_y"] - ly
                lines.append({
                    "text": line_text,
                    "confidence": round(avg_conf, 3),
                    "box": [lx, ly, lw, lh],
                    "bbox_normalized": {
                        "x": round(lx / w, 4),
                        "y": round(ly / h, 4),
                        "width": round(lw / w, 4),
                        "height": round(lh / h, 4)
                    }
                })

            if tokens:
                return {
                    "full_text": " ".join(full_parts),
                    "tokens": tokens,
                    "lines": lines,
                    "image_dims": [w, h],
                    "ocr_engine": "tesseract"
                }
            return None
        except Exception as e:
            print(f"[OCR] Tesseract processing error: {e}")
            return None

    def _try_sidecar_ocr(self, image_path: str, w: int, h: int) -> Optional[Dict[str, Any]]:
        """
        BENCHMARK ONLY: Try to load pre-computed OCR from sidecar .ocr.json.
        This method must NEVER be called in real screening mode.
        """
        # Try .ocr.json sidecar file
        sidecar_path = os.path.splitext(image_path)[0] + ".ocr.json"
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["image_dims"] = [w, h]
                    data["ocr_engine"] = "sidecar"
                    return data
            except Exception:
                pass

        # Try ground_truth.json embedded OCR data
        base_fn = os.path.basename(image_path)
        data_dir = os.path.dirname(image_path)
        gt_path = os.path.join(data_dir, "ground_truth.json")
        if os.path.exists(gt_path):
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt = json.load(f)
                    if base_fn in gt and "ocr_data" in gt[base_fn]:
                        data = gt[base_fn]["ocr_data"]
                        data["image_dims"] = [w, h]
                        data["ocr_engine"] = "ground_truth"
                        return data
            except Exception:
                pass

        return None
