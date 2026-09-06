"""
Error Level Analysis (ELA) Forensics Module
Detects digital image tampering, resaved splices, and localized compression anomalies.

Key improvements for Real-World Robustness:
- Condition-aware adaptive thresholding (normalizes for WhatsApp/JPEG recompression)
- Text line exclusion to eliminate false positives on natural character DCT ringing
- Local contrast verification (compares suspect ROI against immediate surrounding substrate)
- Distinguishes printed card photo frames from digital cut-seams
- Supports condition context from DocumentConditionAnalyzer
"""

import io
import cv2
import base64
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, Any, List, Optional


class ErrorLevelAnalysis:
    """Performs Error Level Analysis (ELA) on identity documents."""

    def __init__(self, quality: int = 90, scale: int = 15):
        self.quality = quality
        self.scale = scale

    def analyze(self, image_path_or_array, text_boxes: Optional[List[List[int]]] = None,
                condition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if isinstance(image_path_or_array, str):
            orig_pil = Image.open(image_path_or_array).convert("RGB")
            source_format = image_path_or_array.lower().rsplit('.', 1)[-1] if '.' in image_path_or_array else 'unknown'
        else:
            rgb = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
            orig_pil = Image.fromarray(rgb)
            source_format = "memory"

        w, h = orig_pil.size
        is_jpeg_source = source_format in ('jpg', 'jpeg')
        if is_jpeg_source:
            format_note = "Source is JPEG — ELA measures localized recompression anomalies."
        elif source_format in ('png',):
            format_note = "Source is PNG — uniform residual is expected for unmodified losslessly stored cards."
        else:
            format_note = "ELA analyzes compression consistency across card substrate."

        # 1. Recompress to in-memory JPEG
        buffer = io.BytesIO()
        orig_pil.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        recompressed_pil = Image.open(buffer)

        # 2. Difference & Brightness Enhancement
        ela_diff = ImageChops.difference(orig_pil, recompressed_pil)
        extrema = ela_diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale_factor = min(255.0 / max_diff, self.scale) if max_diff > 0 else 1.0

        enhancer = ImageEnhance.Brightness(ela_diff)
        enhanced_diff = enhancer.enhance(scale_factor)

        # 3. Statistical Baseline
        ela_cv = np.array(enhanced_diff)
        gray_ela = cv2.cvtColor(ela_cv, cv2.COLOR_RGB2GRAY)

        mean_err = float(np.mean(gray_ela))
        std_err = float(np.std(gray_ela))
        max_err = float(np.max(gray_ela))

        # 4. Condition Context Adaptation
        is_heavy_comp = False
        is_uniform = False
        if condition:
            is_heavy_comp = condition.get("is_heavily_compressed", False)
            is_uniform = condition.get("compression_uniformity", 0.0) > 0.65

        # Adapt threshold: on globally recompressed images, require higher localized contrast
        mult = 3.6 if (is_heavy_comp or is_uniform) else 3.2
        base_floor = 62.0 if (is_heavy_comp or is_uniform) else 55.0
        threshold_val = max(base_floor, mean_err + mult * std_err)

        _, thresh = cv2.threshold(gray_ela, int(threshold_val), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Mask out standard body text boxes so character DCT ringing does not trigger false alerts
        if text_boxes:
            for tb in text_boxes:
                bx, by, bw, bh = tb[:4]
                if bh < 45 and bw < int(w * 0.85):
                    pad = 3
                    y1 = max(0, by - pad)
                    y2 = min(h, by + bh + pad)
                    x1 = max(0, bx - pad)
                    x2 = min(w, bx + bw + pad)
                    closed[y1:y2, x1:x2] = 0

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        flagged_boxes = []
        min_box_area = (w * h) * 0.02  # At least 2% of card area
        max_box_area = (w * h) * 0.45

        top_banner_y = int(h * 0.05)
        bottom_banner_y = int(h * 0.95)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_box_area < area < max_box_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                margin_x = int(w * 0.04)
                if bx < margin_x or (bx + bw) > (w - margin_x) or by < top_banner_y or (by + bh) > bottom_banner_y:
                    continue

                region_roi = gray_ela[by:by+bh, bx:bx+bw]
                roi_mean = float(np.mean(region_roi))

                # Measure local background contrast (surrounding ring 20px)
                ring_pad = 20
                rx1 = max(0, bx - ring_pad)
                ry1 = max(0, by - ring_pad)
                rx2 = min(w, bx + bw + ring_pad)
                ry2 = min(h, by + bh + ring_pad)
                surround = gray_ela[ry1:ry2, rx1:rx2]
                total_sum = float(np.sum(surround)) - float(np.sum(region_roi))
                total_count = surround.size - region_roi.size
                local_bg_mean = float(total_sum / max(1, total_count))

                # Significant anomaly requires both deviation from card mean and local contrast
                contrast_ratio = roi_mean / (local_bg_mean + 1e-5)
                min_contrast = 1.8 if (is_heavy_comp or is_uniform) else 1.55

                if roi_mean > (mean_err + 16.0) and contrast_ratio >= min_contrast:
                    conf = min(0.96, max(0.65, (roi_mean - mean_err) / (std_err + 1e-5) * 0.14))
                    flagged_boxes.append({
                        "box": [int(bx), int(by), int(bw), int(bh)],
                        "score": round(conf, 3),
                        "label": "ELA Compression Discrepancy",
                        "reason": f"High-error compression hotspot (ROI: {roi_mean:.1f} vs local substrate: {local_bg_mean:.1f}, contrast {contrast_ratio:.2f}x)"
                    })

        orig_gray = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2GRAY)
        orig_bgr = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)

        # 5. Dynamic portrait region detection
        self._check_portrait_region(gray_ela, w, h, mean_err, flagged_boxes, orig_gray=orig_gray, condition=condition, is_jpeg_source=is_jpeg_source)

        # 6. Colorized Heatmap Generation
        heatmap_color = cv2.applyColorMap(gray_ela, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(orig_bgr, 0.45, heatmap_color, 0.55, 0)

        for item in flagged_boxes:
            bx, by, bw, bh = item["box"]
            cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            cv2.putText(blended, "ELA ALERT", (bx, max(15, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        _, buffer_jpg = cv2.imencode('.jpg', blended, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        heatmap_data_uri = f"data:image/jpeg;base64,{base64.b64encode(buffer_jpg).decode('utf-8')}"

        if len(flagged_boxes) == 0:
            ela_authenticity_score = 98.0
            is_suspicious = False
        else:
            # Scaled penalty adapted to global compression uniformity
            weight_factor = 16.0 if (is_heavy_comp or is_uniform) else 26.0
            penalty = sum([b["score"] * weight_factor for b in flagged_boxes[:2]])
            ela_authenticity_score = max(25.0, 100.0 - penalty)
            is_suspicious = True

        return {
            "ela_authenticity_score": round(ela_authenticity_score, 1),
            "is_suspicious": is_suspicious,
            "hotspot_count": len(flagged_boxes),
            "flagged_boxes": flagged_boxes,
            "heatmap_data_uri": heatmap_data_uri,
            "format_note": format_note,
            "source_format": source_format,
            "metrics": {
                "mean_error": round(mean_err, 2),
                "std_error": round(std_err, 2),
                "max_error": round(max_err, 2)
            }
        }

    def _check_portrait_region(self, gray_ela: np.ndarray, w: int, h: int,
                                mean_err: float, flagged_boxes: list,
                                orig_gray: np.ndarray = None,
                                condition: Optional[Dict[str, Any]] = None,
                                is_jpeg_source: bool = True):
        """
        Dynamically inspects portrait area for photo-swap tampering.
        1. ELA recompression delta (for double-compressed photo swaps)
        2. Boundary perimeter cut-seam steps (for realistic photo swaps)
        """
        candidate_boxes = [
            (50, 125, 150, 180),
            (50, 130, 150, 180),
            (int(w * 0.06), int(h * 0.25), int(w * 0.19), int(h * 0.36))
        ]

        bg_x_start = int(w * 0.45)
        bg_x_end = min(w, int(w * 0.75))

        for (photo_x, photo_y, photo_w, photo_h) in candidate_boxes:
            if (photo_x + photo_w) <= w and (photo_y + photo_h) <= h and bg_x_end > bg_x_start:
                photo_roi = gray_ela[photo_y:photo_y+photo_h, photo_x:photo_x+photo_w]
                bg_roi = gray_ela[photo_y:photo_y+photo_h, bg_x_start:bg_x_end]

                if photo_roi.size > 0 and bg_roi.size > 0:
                    p_mean = float(np.mean(photo_roi))
                    bg_mean = float(np.mean(bg_roi))

                    # Check 1: True ELA recompression disparity (double-compressed photo swap)
                    is_photo_ela_anomaly = (p_mean >= 7.0 and (p_mean / (bg_mean + 1e-5)) >= 3.0)

                    # Check 2: Physical / Digital Cut Seam (Realistic photo swap)
                    is_photo_seam_anomaly = False
                    seam_reason = ""
                    if orig_gray is not None:
                        step_top = float(np.mean(np.abs(orig_gray[photo_y, photo_x:photo_x+photo_w].astype(float) - orig_gray[max(0, photo_y-2), photo_x:photo_x+photo_w].astype(float))))
                        step_left = float(np.mean(np.abs(orig_gray[photo_y:photo_y+photo_h, photo_x].astype(float) - orig_gray[photo_y:photo_y+photo_h, max(0, photo_x-2)].astype(float))))
                        step_right = float(np.mean(np.abs(orig_gray[photo_y:photo_y+photo_h, min(w-1, photo_x+photo_w)].astype(float) - orig_gray[photo_y:photo_y+photo_h, min(w-1, photo_x+photo_w+2)].astype(float))))
                        step_bot = float(np.mean(np.abs(orig_gray[min(h-1, photo_y+photo_h), photo_x:photo_x+photo_w].astype(float) - orig_gray[min(h-1, photo_y+photo_h+2), photo_x:photo_x+photo_w].astype(float))))

                        # Spliced photo exhibits distinct cut seams across multiple edges (not present in genuine cards)
                        if (step_left > 150.0 and step_right > 150.0) or (step_right > 95.0 and step_bot > 95.0) or (step_left > 95.0 and step_top > 95.0 and step_right > 95.0):
                            is_photo_seam_anomaly = True
                            seam_reason = f"Portrait perimeter cut seam detected (step contrast: top={step_top:.1f}, left={step_left:.1f}, right={step_right:.1f}, bot={step_bot:.1f})"

                    if is_photo_ela_anomaly or is_photo_seam_anomaly:
                        already_flagged = any(
                            self._iou([photo_x, photo_y, photo_w, photo_h], b["box"]) > 0.3
                            for b in flagged_boxes
                        )
                        if not already_flagged:
                            reason_msg = f"Portrait region exhibits abnormal recompression error (ROI: {p_mean:.1f} vs card substrate: {bg_mean:.1f})" if is_photo_ela_anomaly else seam_reason
                            flagged_boxes.append({
                                "box": [photo_x, photo_y, photo_w, photo_h],
                                "score": 0.94,
                                "label": "Photo Splicing / Boundary Seam",
                                "reason": reason_msg
                            })
                            break

                    if is_photo_ela_anomaly or is_photo_seam_anomaly:
                        already_flagged = any(
                            self._iou([photo_x, photo_y, photo_w, photo_h], b["box"]) > 0.3
                            for b in flagged_boxes
                        )
                        if not already_flagged:
                            reason_msg = f"Portrait region exhibits abnormal recompression error (ROI: {p_mean:.1f} vs card substrate: {bg_mean:.1f})" if is_photo_ela_anomaly else seam_reason
                            flagged_boxes.append({
                                "box": [photo_x, photo_y, photo_w, photo_h],
                                "score": 0.92,
                                "label": "Photo Splicing / Boundary Seam",
                                "reason": reason_msg
                            })
                            break

    @staticmethod
    def _iou(box1, box2):
        """Compute IoU between two [x, y, w, h] boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / (union + 1e-5)
