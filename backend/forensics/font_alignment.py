"""
Typography / Rendering Anomaly Forensics Module
Analyzes typography and edge continuity across document lines.
Compares tokens on the same horizontal line or belonging to the same text field
to detect spliced names, altered dates, and mismatched anti-aliasing.

Improvements for Real-World Robustness:
- Rejects continuous printed layout / table divider lines from being flagged as cut seams
- Height-aware token grouping (avoids comparing small labels against bold headers)
- Multi-evidence requirement for typography disparity (sharpness + edge density or OCR drop)
- Condition-aware perspective and blur compensation
"""

import cv2
import numpy as np
import base64
from typing import Dict, Any, List, Optional


class FontAlignmentForensics:
    """Detects spliced typography, font mismatches, and pixel-boundary discontinuities."""

    def __init__(self):
        pass

    def analyze(self, image_path_or_array, ocr_tokens: List[Dict[str, Any]],
                condition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
        else:
            img = image_path_or_array.copy()

        if img is None:
            return {
                "font_consistency_score": 100.0,
                "anomalies_detected": False,
                "flagged_boxes": [],
                "edge_map_base64": None,
                "details": "Image unavailable"
            }

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Edge gradient map for visual inspection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        abs_lap = np.absolute(laplacian)
        norm_lap = np.uint8(np.clip(abs_lap, 0, 255))
        edge_colored = cv2.applyColorMap(norm_lap, cv2.COLORMAP_MAGMA)
        _, edge_buffer = cv2.imencode('.jpg', edge_colored)
        edge_map_base64 = f"data:image/jpeg;base64,{base64.b64encode(edge_buffer).decode('utf-8')}"

        flagged_boxes = []
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

        # Condition flags
        is_camera = condition.get("is_camera_photo", False) if condition else False
        is_blurry = condition.get("is_blurry", False) if condition else False

        # 2. Line-wise Typographical Consistency & Boundary Seam Inspection
        if ocr_tokens and len(ocr_tokens) >= 3:
            top_exclude = int(h * 0.20)
            bottom_exclude = max(int(h * 0.18), 75)

            lines_dict = {}
            for t in ocr_tokens:
                bx, by, bw, bh = t.get("box", [0, 0, 0, 0])
                tok_text = t.get("text", "").strip()

                if bw >= 10 and bh >= 10 and (bx + bw) <= w and (by + bh) <= h:
                    if by < top_exclude or (by + bh) > (h - bottom_exclude):
                        continue

                    # Check 1: Pixel-boundary cut seam along token boundary
                    # A digital box splice corresponds to an inserted name/field, not single punctuation or 1-2 char noise
                    if len(tok_text) >= 3 and bw >= 35 and bh >= 10:
                        top_y1, top_y2 = max(0, by - 5), max(0, by - 1)
                        bot_y1, bot_y2 = min(h, by + bh + 1), min(h, by + bh + 5)
                        top_seam = float(np.mean(sobel_y[top_y1:top_y2, bx:bx+bw])) if top_y2 > top_y1 else 0.0
                        bot_seam = float(np.mean(sobel_y[bot_y1:bot_y2, bx:bx+bw])) if bot_y2 > bot_y1 else 0.0
                        max_seam = max(top_seam, bot_seam)

                        # Verify that this is not a continuous horizontal printed card divider line
                        is_layout_divider = False
                        seam_threshold = 245.0 if (is_camera or is_blurry) else 200.0
                        if max_seam > 175.0:
                            sy1, sy2 = (top_y1, top_y2) if top_seam >= bot_seam else (bot_y1, bot_y2)
                            left_ext = float(np.mean(sobel_y[sy1:sy2, max(0, bx-60):bx])) if bx > 60 else 0.0
                            right_ext = float(np.mean(sobel_y[sy1:sy2, bx+bw:min(w, bx+bw+60)])) if (bx+bw+60) < w else 0.0
                            if left_ext > 130.0 and right_ext > 130.0:
                                is_layout_divider = True

                        # Boundary cut seam indicator (digital paste / box splice)
                        if max_seam > seam_threshold and not is_layout_divider:
                            flagged_boxes.append({
                                "box": [bx, by, bw, bh],
                                "score": min(0.92, 0.72 + 0.10 * (max_seam / 240.0)),
                                "label": "Typography / Rendering Anomaly",
                                "reason": f"Pixel-boundary discontinuity / cut seam detected along '{tok_text}' (gradient: {max_seam:.1f})"
                            })
                            continue

                    roi = gray[by:by+bh, bx:bx+bw]
                    lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())

                    # Filter faint background watermarks from typography consistency analysis
                    if lap_var < 5000:
                        continue

                    y_center = by + bh // 2
                    matched_line = None
                    for line_y in lines_dict.keys():
                        if abs(y_center - line_y) <= 16:
                            matched_line = line_y
                            break
                    if matched_line is None:
                        matched_line = y_center
                        lines_dict[matched_line] = []

                    edges = cv2.Canny(roi, 50, 150)
                    edge_density = float(np.mean(edges > 0))
                    conf = float(t.get("confidence", 1.0))
                    lines_dict[matched_line].append((t, lap_var, conf, edge_density, bh))

            all_confs = [float(t.get("confidence", 1.0)) for t in ocr_tokens if "confidence" in t]
            doc_baseline_conf = float(np.median(all_confs)) if all_confs else 0.95

            # Known static template header tokens to ignore from typography anomaly flagging
            IGNORED_TEMPLATE_WORDS = {
                "GOVERNMENT", "INDIA", "INCOME", "TAX", "DEPARTMENT",
                "AADHAAR", "MERA", "PEHCHAN", "BHARAT", "SARKAR", "UNION",
                "DRIVING", "LICENCE", "LICENSE", "TRANSPORT", "AUTHORITY"
            }

            # Inspect each line for typography disparity
            for line_y, line_tokens in lines_dict.items():
                if len(line_tokens) < 2:
                    continue

                # Filter tokens of comparable character height to avoid label vs value mismatches
                heights = [item[4] for item in line_tokens]
                med_h = float(np.median(heights))
                comparable_tokens = [
                    item for item in line_tokens
                    if abs(item[4] - med_h) <= (0.35 * med_h)
                    and 12 <= item[4] <= 60
                    and item[0].get("text", "").upper().strip() not in IGNORED_TEMPLATE_WORDS
                ]

                if len(comparable_tokens) < 2:
                    continue

                lap_values = [item[1] for item in comparable_tokens]
                min_lap = min(lap_values)
                max_lap = max(lap_values)

                # Camera blur / perspective / anti-aliasing tolerance
                lap_ratio_thresh = 4.5 if (is_camera or is_blurry) else 3.8
                abs_lap_diff_thresh = 48000 if (is_camera or is_blurry) else 40000

                # Intra-line sharpness disparity: must be corroborated by confidence drop or edge difference
                if min_lap > 10000 and (max_lap / (min_lap + 1e-5)) > lap_ratio_thresh and (max_lap - min_lap) > abs_lap_diff_thresh:
                    for item in comparable_tokens:
                        t, v_val, c_val, e_val, _ = item
                        if v_val == max_lap:
                            bx, by, bw, bh = t.get("box")
                            conf_drop = (doc_baseline_conf - c_val) >= 0.28
                            edge_values = [it[3] for it in comparable_tokens]
                            line_edge_median = float(np.median(edge_values))
                            edge_diff = abs(e_val - line_edge_median) / (line_edge_median + 1e-5) > 0.50

                            # Must have actual secondary corroboration (not simply being non-camera)
                            if conf_drop or edge_diff:
                                if not any(f["box"] == [bx, by, bw, bh] for f in flagged_boxes):
                                    flagged_boxes.append({
                                        "box": [bx, by, bw, bh],
                                        "score": 0.86,
                                        "label": "Typography / Rendering Anomaly",
                                        "reason": f"Intra-line stroke sharpness disparity on '{t.get('text')}' ({v_val:.1f} vs line min {min_lap:.1f})"
                                    })

                # Compute line-level edge density baseline
                edge_values = [item[3] for item in comparable_tokens]
                line_edge_median = float(np.median(edge_values))

                for item in comparable_tokens:
                    t, v_val, c_val, e_val, _ = item
                    anomaly_reasons = []

                    # Check 1: Significant OCR confidence drop relative to document baseline
                    conf_drop = (doc_baseline_conf - c_val) >= 0.35
                    if conf_drop:
                        anomaly_reasons.append(
                            f"OCR confidence drop ({c_val:.2f} vs baseline {doc_baseline_conf:.2f})"
                        )

                    # Check 2: Edge density anomaly with confidence drop
                    edge_mismatch = False
                    if line_edge_median > 0:
                        edge_ratio = abs(e_val - line_edge_median) / (line_edge_median + 1e-5)
                        if edge_ratio > 0.65 and len(comparable_tokens) >= 3:
                            edge_mismatch = True
                            anomaly_reasons.append(
                                f"Edge density anomaly ({e_val:.3f} vs line median {line_edge_median:.3f})"
                            )

                    if len(anomaly_reasons) >= 2 or (conf_drop and edge_mismatch):
                        bx, by, bw, bh = t.get("box")
                        if not any(f["box"] == [bx, by, bw, bh] for f in flagged_boxes):
                            reason_text = "; ".join(anomaly_reasons)
                            flagged_boxes.append({
                                "box": [bx, by, bw, bh],
                                "score": min(0.92, 0.70 + 0.10 * len(anomaly_reasons)),
                                "label": "Typography / Rendering Anomaly",
                                "reason": f"Typographical mismatch on '{t.get('text')}': {reason_text}"
                            })

        # Multi-token requirement: a single isolated token variation does not constitute verified forgery unless it is a verified cut seam on a clean scan
        has_cut_seam = any("cut seam" in f.get("reason", "").lower() for f in flagged_boxes)
        if len(flagged_boxes) >= 2 or (len(flagged_boxes) == 1 and has_cut_seam and not (is_camera or is_blurry)):
            is_suspicious = True
            score = max(35.0, 100.0 - (len(flagged_boxes) * 25.0))
            details = f"Detected {len(flagged_boxes)} fields with abnormal typographical rendering or edge cut seams."
        elif len(flagged_boxes) == 1:
            is_suspicious = False  # Isolated token variation, treated as weak anomaly
            score = 88.0
            details = f"Minor isolated typographical variation on 1 token ('{flagged_boxes[0].get('reason')}'), likely optical focus or formatting difference."
        else:
            is_suspicious = False
            score = 100.0
            details = "Typography rendering and stroke gradients are uniformly consistent across document fields."

        return {
            "font_consistency_score": round(score, 1),
            "anomalies_detected": is_suspicious,
            "flagged_boxes": flagged_boxes,
            "edge_map_base64": edge_map_base64,
            "details": details
        }
