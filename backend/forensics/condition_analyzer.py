"""
Document Condition Analyzer Module
Analyzes physical and digital capture conditions of document images:
- JPEG compression level & DCT blocking artifact intensity
- Compression uniformity (global vs localized)
- Focus / blur estimation (Laplacian variance)
- Effective resolution & scale
- Capture type estimation (scanner vs camera photo vs web thumbnail)

CRITICAL FORENSIC PRINCIPLE:
Condition analysis provides CONTEXT, never a direct tampering verdict.
Example: 'Heavy JPEG compression detected' -> 'Lower sensitivity of ELA detector'
NOT: 'Document is suspicious.'
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class DocumentConditionAnalyzer:
    """Estimates capture condition and compression characteristics of an image."""

    def __init__(self):
        pass

    def analyze(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Extracts condition metrics from a BGR image array.
        Returns a structured dictionary of normalized condition parameters.
        """
        if img_bgr is None or img_bgr.size == 0:
            return {
                "resolution": {"width": 0, "height": 0, "megapixels": 0.0},
                "quality_tier": "VERY_LOW",
                "reliability_score": 0.0,
                "quality_explanation": "Image array is empty or unreadable.",
                "condition_profile": "SEVERELY_DEGRADED",
                "is_low_res": True,
                "blur_score": 0.0,
                "is_blurry": True,
                "compression_uniformity": 1.0,
                "estimated_jpeg_quality": 85,
                "is_heavily_compressed": False,
                "is_compressed": False,
                "is_camera_photo": False,
                "is_screenshot": False,
                "is_print_and_scan": False,
                "recommended_sensitivity": "standard"
            }

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mp = (w * h) / 1_000_000.0

        # 1. Blur / Sharpness estimation
        # Laplacian variance: < 100 indicates blur/out-of-focus; > 500 is sharp; > 2000 is crisp digital render
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = lap_var < 120.0

        # 2. Estimate 8x8 DCT Block Discontinuity (JPEG grid artifacts)
        # In JPEGs, pixel differences across 8-pixel boundaries (x % 8 == 7 vs x % 8 == 0)
        # are systematically higher than within-block differences (x % 8 == 3 vs x % 8 == 4).
        block_step, internal_step = self._measure_blocking_artifacts(gray)
        blockiness_ratio = block_step / (internal_step + 1e-5)
        
        # High blockiness ratio (> 1.25) indicates prominent JPEG quantization (quality <= 75)
        is_heavily_compressed = blockiness_ratio > 1.20 or (internal_step < 3.0 and blockiness_ratio > 1.12)

        # 3. Spatial Compression Uniformity
        # Divide image into a 4x4 grid and measure high-frequency residual standard deviation
        uniformity_score = self._measure_compression_uniformity(gray)
        # High uniformity (> 0.70) means the whole document was compressed uniformly (e.g. WhatsApp/web re-save)

        # 4. Camera Photo vs Flatbed / Digital Render
        # Camera captures typically exhibit gradual illumination falloff or slight perspective curvature
        is_camera_photo = (w > 1200 or h > 1200) and lap_var < 1500.0 and blockiness_ratio < 1.35

        # 5. Orientation / Rotation Estimation
        orientation_angle, orientation_conf = self.estimate_orientation(gray)

        # 6. Global Illumination & Contrast Metrics
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # High-frequency noise estimation via median absolute deviation of Laplacian
        noise_level = float(np.median(np.abs(cv2.Laplacian(gray, cv2.CV_64F))))

        # 7. Screenshot and Print-and-Scan Heuristics
        # Common screen aspect ratios: 16:9 (1.778), 16:10 (1.60), 4:3 (1.333), 19.5:9 (2.167)
        aspect_ratio = float(w) / max(1.0, float(h))
        is_screen_ratio = (
            abs(aspect_ratio - 1.778) < 0.05 or
            abs(aspect_ratio - 1.600) < 0.05 or
            abs(aspect_ratio - 2.167) < 0.06 or
            (w in (1920, 1366, 1280, 1440, 2560, 3840) and h in (1080, 768, 720, 900, 1440, 2160))
        )
        # Screenshots typically have very low noise level and zero perspective/optical curvature
        is_screenshot = bool(is_screen_ratio and noise_level < 3.5 and blockiness_ratio < 1.15 and mp >= 0.50)

        # Print-and-scan indicators: presence of scanner noise / paper texture,
        # combined with absence of clean DCT grids and moderate high-frequency noise
        is_print_and_scan = bool(
            not is_screenshot and
            noise_level > 6.0 and
            blockiness_ratio < 1.10 and
            uniformity_score > 0.65 and
            lap_var < 500.0
        )

        # 8. Forensic Reliability Score & Quality Tier Computation
        # Quantifies "How reliable is the document image for automated forensic screening?"
        # NOT "How fake is the document?"
        rel_score = 1.0

        # Penalize resolution limitations
        if mp < 0.08:
            rel_score -= 0.45
        elif mp < 0.20:
            rel_score -= 0.22
        elif mp < 0.45:
            rel_score -= 0.08

        # Penalize blur / defocus
        if lap_var < 45.0:
            rel_score -= 0.38
        elif lap_var < 100.0:
            rel_score -= 0.20
        elif lap_var < 220.0:
            rel_score -= 0.08

        # Penalize heavy quantization
        if is_heavily_compressed:
            rel_score -= 0.15
        elif blockiness_ratio > 1.14:
            rel_score -= 0.06

        # Penalize low contrast
        if contrast < 24.0:
            rel_score -= 0.18
        elif contrast < 35.0:
            rel_score -= 0.08

        # Bound reliability score
        reliability_score = round(max(0.08, min(1.0, rel_score)), 3)

        # 4-Tier Taxonomy
        if reliability_score >= 0.82 and mp >= 0.40 and lap_var >= 180.0:
            quality_tier = "GOOD"
            quality_explanation = "High image clarity with sufficient resolution and sharpness for full automated forensic analysis."
        elif reliability_score >= 0.60:
            quality_tier = "ACCEPTABLE"
            quality_explanation = "Acceptable capture clarity suitable for OCR parsing, mathematical checksums, and standard forensic inspection."
        elif reliability_score >= 0.40:
            quality_tier = "LOW"
            quality_explanation = "Degraded capture quality (blur, compression, or low resolution). Digital forensics attenuated to prevent false alarms."
        else:
            quality_tier = "VERY_LOW"
            quality_explanation = "Image quality is severely degraded and insufficient for reliable autonomous verification. Manual human inspection required."

        # Map to legacy condition profile
        if quality_tier == "VERY_LOW":
            condition_profile = "SEVERELY_DEGRADED"
        elif quality_tier == "LOW":
            condition_profile = "DEGRADED"
        elif quality_tier == "ACCEPTABLE":
            condition_profile = "GOOD"
        else:
            condition_profile = "EXCELLENT"

        # Downstream forensic attenuation factors
        if quality_tier == "VERY_LOW":
            typography_attenuation = 0.20
            ela_attenuation = 0.35
            copy_move_attenuation = 0.30
        elif quality_tier == "LOW":
            typography_attenuation = 0.50
            ela_attenuation = 0.65
            copy_move_attenuation = 0.60
        else:
            typography_attenuation = 1.0
            ela_attenuation = 1.0
            copy_move_attenuation = 1.0

        # Structured Condition Flags & Levels
        condition_flags = []
        if is_heavily_compressed:
            condition_flags.append("heavy_jpeg_compression")
        if uniformity_score > 0.70:
            condition_flags.append("uniform_recompression_artifact")
        if is_blurry:
            condition_flags.append("mild_blur_defocus")
        if is_camera_photo:
            condition_flags.append("camera_capture_lighting")
        if is_screenshot:
            condition_flags.append("screenshot_capture")
        if is_print_and_scan:
            condition_flags.append("print_and_scan_texture")
        if orientation_angle != 0:
            condition_flags.append(f"rotation_{orientation_angle}deg")
        if mp < 0.35:
            condition_flags.append("low_resolution")
        if quality_tier in ["LOW", "VERY_LOW"]:
            condition_flags.append(f"quality_tier_{quality_tier.lower()}")

        blur_level = "HIGH" if lap_var < 100.0 else ("NORMAL" if lap_var < 400.0 else "LOW")
        compression_level = "HEAVY" if is_heavily_compressed else ("MODERATE" if blockiness_ratio > 1.08 else "LOW")

        # 9. Determine Sensitivity Adaptations
        if is_heavily_compressed or uniformity_score > 0.75:
            recommended_sensitivity = "tolerant_compression"
        elif is_blurry:
            recommended_sensitivity = "tolerant_blur"
        elif mp < 0.25: # < 250k pixels
            recommended_sensitivity = "tolerant_low_res"
        else:
            recommended_sensitivity = "standard"

        return {
            "resolution": {
                "width": w,
                "height": h,
                "megapixels": round(mp, 3)
            },
            "quality_tier": quality_tier,
            "reliability_score": reliability_score,
            "quality_explanation": quality_explanation,
            "condition_profile": condition_profile,
            "typography_attenuation": typography_attenuation,
            "ela_attenuation": ela_attenuation,
            "copy_move_attenuation": copy_move_attenuation,
            "blur_level": blur_level,
            "blur_score": round(lap_var, 1),
            "is_blurry": is_blurry,
            "compression_level": compression_level,
            "blockiness_ratio": round(blockiness_ratio, 3),
            "is_heavily_compressed": is_heavily_compressed,
            "is_compressed": is_heavily_compressed,
            "compression_uniformity": round(uniformity_score, 3),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "noise_level": round(noise_level, 2),
            "orientation": orientation_angle,
            "orientation_confidence": round(orientation_conf, 2),
            "is_camera_photo": is_camera_photo,
            "is_screenshot": is_screenshot,
            "is_print_and_scan": is_print_and_scan,
            "is_low_res": mp < 0.35,
            "condition_flags": condition_flags,
            "recommended_sensitivity": recommended_sensitivity
        }

    def estimate_orientation(self, gray: np.ndarray) -> Tuple[int, float]:
        """
        Estimates document rotation angle (0, 90, 180, 270) using card aspect ratio
        and line projection profiles. Returns (angle_deg, confidence).
        Only confident non-zero orientations are recommended for rotation.
        """
        h, w = gray.shape
        # Standard ID cards are landscape (w / h ~ 1.58).
        # If image is prominently taller than wide (h / w > 1.25), document is rotated 90 or 270 deg.
        if h > int(w * 1.25):
            # Check projection profile variance along rows vs cols
            # Rotate 90 deg clockwise to test if text becomes horizontal
            rot90 = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            var_orig_rows = float(np.var(np.mean(gray, axis=1)))
            var_rot_rows = float(np.var(np.mean(rot90, axis=1)))
            if var_rot_rows > var_orig_rows * 1.15:
                return 90, 0.85
            else:
                return 270, 0.80

        return 0, 0.95

    @staticmethod
    def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
        """Rotates image by 0, 90, 180, or 270 degrees clockwise."""
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    @staticmethod
    def remap_box_to_original(box: List[int], angle: int, orig_w: int, orig_h: int) -> List[int]:
        """
        Converts a bounding box [x, y, w, h] from a rotated image coordinate space
        back to the original image coordinate space.
        """
        if angle == 0 or len(box) < 4:
            return box

        bx, by, bw, bh = box[:4]
        if angle == 90:
            # Rotated 90 CW: new_w = orig_h, new_h = orig_w
            # x_orig = y_rot, y_orig = orig_h - (x_rot + w_rot)
            return [by, orig_h - (bx + bw), bh, bw]
        elif angle == 180:
            return [orig_w - (bx + bw), orig_h - (by + bh), bw, bh]
        elif angle == 270:
            # Rotated 270 CW (90 CCW):
            # x_orig = orig_w - (y_rot + h_rot), y_orig = x_rot
            return [orig_w - (by + bh), bx, bh, bw]
        return box

    def _measure_blocking_artifacts(self, gray: np.ndarray) -> Tuple[float, float]:
        """Measures gradient strength at 8x8 block boundaries vs intra-block columns/rows."""
        h, w = gray.shape
        if h < 64 or w < 64:
            return 1.0, 1.0

        # Subsample to central 80% to avoid border artifacts
        y1, y2 = int(h * 0.1), int(h * 0.9)
        x1, x2 = int(w * 0.1), int(w * 0.9)
        crop = gray[y1:y2, x1:x2].astype(np.float32)

        # Differences across columns
        diff_x = np.abs(crop[:, 1:] - crop[:, :-1])
        # Grid boundaries: col index 7, 15, 23...
        cols = np.arange(diff_x.shape[1])
        boundary_mask_x = (cols % 8 == 7)
        internal_mask_x = (cols % 8 == 3)

        b_diff_x = float(np.mean(diff_x[:, boundary_mask_x])) if np.any(boundary_mask_x) else 1.0
        i_diff_x = float(np.mean(diff_x[:, internal_mask_x])) if np.any(internal_mask_x) else 1.0

        return b_diff_x, max(0.5, i_diff_x)

    def _measure_compression_uniformity(self, gray: np.ndarray) -> float:
        """Measures whether compression/noise behavior is uniform across spatial quadrants."""
        h, w = gray.shape
        grid_rows, grid_cols = 4, 4
        rh, cw = h // grid_rows, w // grid_cols
        if rh < 16 or cw < 16:
            return 1.0

        quad_stds = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                patch = gray[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
                # High-pass filter via Laplacian
                lap = cv2.Laplacian(patch, cv2.CV_32F)
                quad_stds.append(float(np.std(lap)))

        if not quad_stds:
            return 1.0

        mean_std = float(np.mean(quad_stds))
        std_of_stds = float(np.std(quad_stds))
        # Coefficient of variation (lower variation = higher uniformity)
        cov = std_of_stds / (mean_std + 1e-5)
        # Normalized uniformity index between 0.0 and 1.0
        uniformity = max(0.0, min(1.0, 1.0 - (cov / 2.0)))
        return uniformity


_analyzer_instance = DocumentConditionAnalyzer()

def analyze_document_condition(img_bgr: np.ndarray) -> Dict[str, Any]:
    """Convenience helper function to run DocumentConditionAnalyzer."""
    return _analyzer_instance.analyze(img_bgr)

