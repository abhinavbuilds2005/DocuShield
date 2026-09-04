"""
Typography / Rendering Anomaly Forensics Module
Analyzes typography and edge continuity across document lines.
Compares tokens on the same horizontal line or belonging to the same text field
to detect spliced names, altered dates, and mismatched anti-aliasing.

Detection methods:
1. Laplacian variance comparison (stroke sharpness consistency)
2. OCR confidence anomaly detection (spliced text recognition degradation)
3. Edge density comparison across same-line tokens
"""

import cv2
import numpy as np
import base64
from typing import Dict, Any, List


class FontAlignmentForensics:
    """Detects spliced typography, font mismatches, and pixel-boundary discontinuities."""

    def __init__(self):
        pass

    def analyze(self, image_path_or_array, ocr_tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
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

        # 2. Line-wise Typographical Consistency & Boundary Seam Inspection
        if ocr_tokens and len(ocr_tokens) >= 3:
            # Dynamic banner exclusion (proportional)
            top_exclude = int(h * 0.20)
            bottom_exclude = int(h * 0.06)

            # Group tokens into horizontal lines (tokens with similar y center within 16px)
            lines_dict = {}
            for t in ocr_tokens:
                bx, by, bw, bh = t.get("box", [0, 0, 0, 0])
                if bw > 10 and bh > 10 and (bx + bw) <= w and (by + bh) <= h:
                    # Ignore header banner and bottom safety banner (proportional)
                    if by < top_exclude or (by + bh) > (h - bottom_exclude):
                        continue

                    # Check 1: Pixel-boundary discontinuity / cut seam along token boundary
                    top_y1, top_y2 = max(0, by - 5), max(0, by - 1)
                    bot_y1, bot_y2 = min(h, by + bh + 1), min(h, by + bh + 5)
                    top_seam = float(np.mean(sobel_y[top_y1:top_y2, bx:bx+bw])) if top_y2 > top_y1 else 0.0
                    bot_seam = float(np.mean(sobel_y[bot_y1:bot_y2, bx:bx+bw])) if bot_y2 > bot_y1 else 0.0
                    max_seam = max(top_seam, bot_seam)

                    roi = gray[by:by+bh, bx:bx+bw]
                    lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())

                    # Boundary cut seam indicator (strong evidence of digital paste / splice)
                    if max_seam > 180.0:
                        flagged_boxes.append({
                            "box": [bx, by, bw, bh],
                            "score": min(0.95, 0.75 + 0.10 * (max_seam / 200.0)),
                            "label": "Typography / Rendering Anomaly",
                            "reason": f"Pixel-boundary discontinuity / cut seam detected along '{t.get('text')}' (gradient: {max_seam:.1f})"
                        })
                        continue

                    # Filter faint background watermarks from typography consistency analysis
                    if lap_var < 5000:
                        continue
                    
                    y_center = by + bh // 2
                    # Find matching line bin
                    matched_line = None
                    for line_y in lines_dict.keys():
                        if abs(y_center - line_y) <= 16:
                            matched_line = line_y
                            break
                    if matched_line is None:
                        matched_line = y_center
                        lines_dict[matched_line] = []
                    
                    # Edge density: proportion of strong edges in the ROI
                    edges = cv2.Canny(roi, 50, 150)
                    edge_density = float(np.mean(edges > 0))
                    
                    conf = float(t.get("confidence", 1.0))
                    lines_dict[matched_line].append((t, lap_var, conf, edge_density))

            # Document baseline confidence across all extracted tokens
            all_confs = [float(t.get("confidence", 1.0)) for t in ocr_tokens if "confidence" in t]
            doc_baseline_conf = float(np.median(all_confs)) if all_confs else 0.95

            # Inspect each line for typography & sharpness disparity
            for line_y, line_tokens in lines_dict.items():
                if len(line_tokens) < 2:
                    continue

                # Compute line-level Laplacian variance baseline
                lap_values = [item[1] for item in line_tokens]
                min_lap = min(lap_values)
                max_lap = max(lap_values)
                line_lap_median = float(np.median(lap_values))

                # Intra-line sharpness disparity: spliced text typically exhibits starkly higher sharpness
                if min_lap > 10000 and (max_lap / (min_lap + 1e-5)) > 3.2 and (max_lap - min_lap) > 35000:
                    for item in line_tokens:
                        t, v_val, c_val, e_val = item
                        if v_val == max_lap:
                            bx, by, bw, bh = t.get("box")
                            # Avoid duplicate flags
                            if not any(f["box"] == [bx, by, bw, bh] for f in flagged_boxes):
                                flagged_boxes.append({
                                    "box": [bx, by, bw, bh],
                                    "score": 0.88,
                                    "label": "Typography / Rendering Anomaly",
                                    "reason": f"Intra-line stroke sharpness disparity on '{t.get('text')}' ({v_val:.1f} vs line min {min_lap:.1f})"
                                })

                # Compute line-level edge density baseline
                edge_values = [item[3] for item in line_tokens]
                line_edge_median = float(np.median(edge_values))

                for item in line_tokens:
                    t, v_val, c_val, e_val = item
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
                        if edge_ratio > 0.6 and len(line_tokens) >= 3:
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
                                "score": min(0.95, 0.70 + 0.12 * len(anomaly_reasons)),
                                "label": "Typography / Rendering Anomaly",
                                "reason": f"Typographical mismatch on '{t.get('text')}': {reason_text}"
                            })

        is_suspicious = len(flagged_boxes) > 0
        if is_suspicious:
            score = max(30.0, 100.0 - (len(flagged_boxes) * 35.0))
            details = f"Detected {len(flagged_boxes)} fields with abnormal typographical rendering or edge gradients."
        else:
            score = 100.0
            details = "Typography rendering and stroke gradients are uniformly consistent across document fields."

        return {
            "font_consistency_score": round(score, 1),
            "anomalies_detected": is_suspicious,
            "flagged_boxes": flagged_boxes,
            "edge_map_base64": edge_map_base64,
            "details": details
        }
