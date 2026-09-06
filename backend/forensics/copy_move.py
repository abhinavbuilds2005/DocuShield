"""
Copy-Move Forgery Detection Module
Identifies duplicated or cloned image regions using keypoint spatial clustering.
Masks standard text form lines to avoid false-positive matches on recurring typographical characters.

Improvements for Real-World Robustness:
- Stronger internal repetitive pattern rejection (QR code finder patterns & emblem self-matches)
- Center distance separation enforcement to distinguish foreign clones from localized textures
- Robust RANSAC affine transformation geometric verification
- Minimum physical patch area filtering
"""

import cv2
import numpy as np
import base64
from typing import Dict, Any, List, Tuple, Optional


class CopyMoveDetector:
    """Detects duplicated image regions using keypoint spatial displacement analysis."""

    def __init__(self, min_matches: int = 14, min_distance_px: float = 75.0):
        self.min_matches = min_matches
        self.min_distance_px = min_distance_px
        self.orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)

    def analyze(self, image_path_or_array, text_boxes: Optional[List[List[int]]] = None,
                condition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
        else:
            img = image_path_or_array.copy()

        if img is None:
            return {
                "copy_move_detected": False,
                "score": 100.0,
                "matched_pair_count": 0,
                "flagged_boxes": [],
                "details": "Image unavailable"
            }

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Dynamic mask: exclude top/bottom safety banners
        mask = np.ones((h, w), dtype=np.uint8) * 255
        top_banner = max(1, int(h * 0.20))
        bottom_banner = max(1, int(h * 0.05))
        mask[0:top_banner, :] = 0
        mask[h-bottom_banner:h, :] = 0

        # Mask standard form body text boxes to prevent recurring character false positives
        if text_boxes:
            for b in text_boxes:
                bx, by, bw, bh = b[:4]
                if bh < 45 and bw < int(w * 0.85):
                    pad = 3
                    y1 = max(0, by - pad)
                    y2 = min(h, by + bh + pad)
                    x1 = max(0, bx - pad)
                    x2 = min(w, bx + bw + pad)
                    mask[y1:y2, x1:x2] = 0

        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)

        if descriptors is None or len(keypoints) < 20:
            return {
                "copy_move_detected": False,
                "score": 100.0,
                "matched_pair_count": 0,
                "flagged_boxes": [],
                "details": "No non-text graphic keypoints found for copy-move comparison"
            }

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            k = min(4, len(descriptors))
            if k < 2:
                return {
                    "copy_move_detected": False,
                    "score": 100.0,
                    "matched_pair_count": 0,
                    "flagged_boxes": [],
                    "details": "Insufficient descriptors for matching"
                }
            matches = bf.knnMatch(descriptors, descriptors, k=k)
        except Exception:
            return {
                "copy_move_detected": False,
                "score": 100.0,
                "matched_pair_count": 0,
                "flagged_boxes": [],
                "details": "Matching failed"
            }

        valid_pairs: List[Tuple[cv2.KeyPoint, cv2.KeyPoint, float, float]] = []

        for m_list in matches:
            non_self = [m for m in m_list if m.queryIdx < m.trainIdx]
            if len(non_self) < 2:
                continue

            best = non_self[0]
            second_best = non_self[1]

            # Lowe's ratio test
            if best.distance > 0.74 * second_best.distance:
                continue

            kp1 = keypoints[best.queryIdx]
            kp2 = keypoints[best.trainIdx]
            dist_px = np.hypot(kp1.pt[0] - kp2.pt[0], kp1.pt[1] - kp2.pt[1])

            if dist_px > self.min_distance_px and best.distance < 36:
                dx = kp2.pt[0] - kp1.pt[0]
                dy = kp2.pt[1] - kp1.pt[1]
                valid_pairs.append((kp1, kp2, dx, dy))

        flagged_boxes = []
        is_tampered = False
        vis_base64 = None

        if len(valid_pairs) >= self.min_matches:
            displacements = np.array([(p[2], p[3]) for p in valid_pairs])
            bins = {}
            for idx, (dx, dy) in enumerate(displacements):
                bin_key = (int(round(dx / 30.0)), int(round(dy / 30.0)))
                bins.setdefault(bin_key, []).append(idx)

            # Periodicity check:
            # 1. Document-wide periodic guilloche security background produces >= 4 distinct displacement bins with many matches
            # 2. Multi-directional displacement clusters (angle difference between 15 deg and 165 deg) indicate 2D geometric lattices
            # 3. High total dispersal: if matches are spread across many bins, it's repetitive texture, not localized cloning
            large_cluster_keys = [k for k, v in bins.items() if len(v) >= max(8, self.min_matches // 2)]
            is_multi_directional = False
            if len(large_cluster_keys) >= 2:
                angles = [np.arctan2(k[1], k[0]) for k in large_cluster_keys]
                for i in range(len(angles)):
                    for j in range(i + 1, len(angles)):
                        diff = abs(angles[i] - angles[j])
                        diff = min(diff, 2 * np.pi - diff)
                        # Exclude 180-degree opposite vectors (query/train index order inversions)
                        if np.deg2rad(15) < diff < np.deg2rad(165):
                            is_multi_directional = True
                            break
                    if is_multi_directional:
                        break

            # Dispersal check: if total valid pairs are spread across 5+ distinct bins, it's repetitive texture
            populated_bins = [k for k, v in bins.items() if len(v) >= 3]
            is_high_dispersal = len(populated_bins) >= 5 and len(valid_pairs) >= 20

            is_global_periodic_pattern = (len(large_cluster_keys) >= 4) or is_multi_directional or is_high_dispersal

            max_cluster_key = max(bins.keys(), key=lambda k: len(bins[k]))
            max_cluster = bins[max_cluster_key]

            # In diffuse periodic patterns (guilloche lines), all bins have small comparable sizes.
            # In genuine copy-move forgery, a single displacement vector has a dense, dominant cluster.
            # Only discard as global periodic pattern if there is NO dominant dense cluster.
            is_dominant_cluster = (len(max_cluster) >= 22) and (len(max_cluster) >= 0.25 * len(valid_pairs))

            if is_global_periodic_pattern and not is_dominant_cluster:
                max_cluster = []  # Discard as periodic security background

            if len(max_cluster) >= self.min_matches:
                pts_src = np.array([valid_pairs[i][0].pt for i in max_cluster])
                pts_dst = np.array([valid_pairs[i][1].pt for i in max_cluster])

                is_geometrically_valid = False
                if len(pts_src) >= 4:
                    try:
                        H, inlier_mask = cv2.estimateAffinePartial2D(
                            pts_src.reshape(-1, 1, 2),
                            pts_dst.reshape(-1, 1, 2),
                            method=cv2.RANSAC,
                            ransacReprojThreshold=5.0
                        )
                        if inlier_mask is not None:
                            inlier_count = int(np.sum(inlier_mask))
                            inlier_ratio = inlier_count / len(pts_src)
                            if (inlier_count >= 15 and inlier_ratio >= 0.35) or (inlier_count >= 8 and inlier_ratio >= 0.45):
                                is_geometrically_valid = True
                                inlier_indices = [max_cluster[i] for i in range(len(max_cluster))
                                                 if inlier_mask[i]]
                                pts_src = np.array([valid_pairs[i][0].pt for i in inlier_indices])
                                pts_dst = np.array([valid_pairs[i][1].pt for i in inlier_indices])
                                max_cluster = inlier_indices
                    except Exception:
                        pass

                if is_geometrically_valid and len(max_cluster) >= 8:
                    x_min1, y_min1 = np.min(pts_src, axis=0)
                    x_max1, y_max1 = np.max(pts_src, axis=0)
                    x_min2, y_min2 = np.min(pts_dst, axis=0)
                    x_max2, y_max2 = np.max(pts_dst, axis=0)

                    pad = 10
                    b1 = [max(0, int(x_min1 - pad)), max(0, int(y_min1 - pad)),
                          min(w, int(x_max1 - x_min1 + 2 * pad)), min(h, int(y_max1 - y_min1 + 2 * pad))]
                    b2 = [max(0, int(x_min2 - pad)), max(0, int(y_min2 - pad)),
                          min(w, int(x_max2 - x_min2 + 2 * pad)), min(h, int(y_max2 - y_min2 + 2 * pad))]

                    # 1. Reject internal repetitive patterns within a single compact graphic (e.g. QR code finder squares)
                    overall_w = max(b1[0] + b1[2], b2[0] + b2[2]) - min(b1[0], b2[0])
                    overall_h = max(b1[1] + b1[3], b2[1] + b2[3]) - min(b1[1], b2[1])
                    cx1, cy1 = b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0
                    cx2, cy2 = b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0
                    center_dist = np.hypot(cx1 - cx2, cy1 - cy2)

                    # 2. Scale-adaptive patch size and area thresholds
                    min_patch_w = max(35, int(w * 0.045))
                    min_patch_h = max(35, int(h * 0.045))
                    min_patch_area = max(1800, int(w * h * 0.0030))
                    is_sub_patch = (b1[2] < min_patch_w or b1[3] < min_patch_h or
                                    b2[2] < min_patch_w or b2[3] < min_patch_h or
                                    (b1[2] * b1[3]) < min_patch_area or
                                    (b2[2] * b2[3]) < min_patch_area)

                    # 3. Margin / Border template rejection (card perimeter decorative patterns)
                    margin_x = max(int(w * 0.08), 25)
                    margin_y = max(int(h * 0.08), 25)
                    is_border_match = (
                        (b1[0] < margin_x or (b1[0] + b1[2]) > (w - margin_x) or b1[1] < margin_y or (b1[1] + b1[3]) > (h - margin_y)) and
                        (b2[0] < margin_x or (b2[0] + b2[2]) > (w - margin_x) or b2[1] < margin_y or (b2[1] + b2[3]) > (h - margin_y))
                    )

                    # 4. Aspect ratio check: reject long thin divider / decorative lines
                    aspect1 = max(b1[2] / max(1, b1[3]), b1[3] / max(1, b1[2]))
                    aspect2 = max(b2[2] / max(1, b2[3]), b2[3] / max(1, b2[2]))
                    is_thin_stripe = (aspect1 > 3.5 or aspect2 > 3.5)

                    # 5. Patch texture variance: reject flat uniform background patches and low-detail regions
                    p1 = gray[b1[1]:b1[1]+b1[3], b1[0]:b1[0]+b1[2]]
                    p2 = gray[b2[1]:b2[1]+b2[3], b2[0]:b2[0]+b2[2]]
                    var1 = float(np.var(p1)) if p1.size > 0 else 0.0
                    var2 = float(np.var(p2)) if p2.size > 0 else 0.0
                    # Also check Laplacian (edge) texture — uniform security background has low Laplacian variance
                    lap_var1 = float(cv2.Laplacian(p1, cv2.CV_64F).var()) if p1.size > 0 else 0.0
                    lap_var2 = float(cv2.Laplacian(p2, cv2.CV_64F).var()) if p2.size > 0 else 0.0
                    is_low_texture = (var1 < 100.0 or var2 < 100.0 or lap_var1 < 150.0 or lap_var2 < 150.0)

                    # 6. Keypoint 2D spatial distribution within patch (genuine cloned graphics are 2D, not 1D lines)
                    spread_x1 = float(np.std(pts_src[:, 0])) if len(pts_src) > 1 else 0.0
                    spread_y1 = float(np.std(pts_src[:, 1])) if len(pts_src) > 1 else 0.0
                    is_1d_collinear = (spread_x1 < 8.0 or spread_y1 < 8.0)

                    # 7. Periodic 1D displacement check (security guilloche wave columns)
                    dx_abs = abs(cx1 - cx2)
                    dy_abs = abs(cy1 - cy2)
                    is_periodic_1d = (dx_abs < max(15, int(w * 0.015)) and dy_abs > 100) or \
                                     (dy_abs < max(15, int(h * 0.015)) and dx_abs > 100)

                    # 8. Dual QR Code / Official Barcode Rejection:
                    # Multi-section credentials (e.g., e-Aadhaar letters, dual-section tax filings/certificates)
                    # legitimately print identical machine-readable 2D QR codes in both the notification and wallet sections.
                    is_dual_qr = False
                    ar1 = b1[2] / max(1, b1[3])
                    ar2 = b2[2] / max(1, b2[3])
                    if (0.75 <= ar1 <= 1.33) and (0.75 <= ar2 <= 1.33):
                        try:
                            qr_detector = cv2.QRCodeDetector()
                            is_qr1 = bool(qr_detector.detect(p1)[0])
                            is_qr2 = bool(qr_detector.detect(p2)[0])
                            if is_qr1 or is_qr2:
                                edges1 = cv2.Canny(p1, 50, 150)
                                edges2 = cv2.Canny(p2, 50, 150)
                                d1 = float(np.mean(edges1 > 0)) if edges1.size > 0 else 0.0
                                d2 = float(np.mean(edges2 > 0)) if edges2.size > 0 else 0.0
                                if d1 > 0.15 and d2 > 0.15:
                                    is_dual_qr = True
                        except Exception:
                            pass

                    min_required_dist = max(110, int(np.hypot(w, h) * 0.08))
                    # Require more inliers for blurry/compressed images where noise creates spurious matches
                    is_degraded = condition and (condition.get("is_blurry", False) or condition.get("is_heavily_compressed", False))
                    min_inliers = 14 if is_degraded else 10

                    # Requires meaningful spatial displacement, reasonable patch size, and non-template context
                    if overall_w < 150 and overall_h < 180:
                        is_tampered = False
                    elif center_dist < min_required_dist:
                        is_tampered = False
                    elif is_sub_patch:
                        is_tampered = False
                    elif is_border_match or is_thin_stripe or is_low_texture:
                        is_tampered = False
                    elif is_1d_collinear or is_periodic_1d:
                        is_tampered = False
                    elif is_dual_qr:
                        is_tampered = False
                    elif len(max_cluster) < min_inliers:
                        is_tampered = False
                    else:
                        is_tampered = True
                        flagged_boxes.append({
                            "box": b1,
                            "score": 0.92,
                            "label": "Copy-Move Source Region",
                            "reason": f"Cloned graphic patch verified with {len(max_cluster)} spatially coherent keypoints (RANSAC verified)"
                        })
                        flagged_boxes.append({
                            "box": b2,
                            "score": 0.95,
                            "label": "Copy-Move Cloned Region",
                            "reason": "Target duplicate patch inserted over background"
                        })

                        vis_img = img.copy()
                        for idx in max_cluster:
                            kp1, kp2, _, _ = valid_pairs[idx]
                            pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
                            pt2 = (int(kp2.pt[0]), int(kp2.pt[1]))
                            cv2.line(vis_img, pt1, pt2, (0, 165, 255), 2)
                            cv2.circle(vis_img, pt1, 4, (0, 0, 255), -1)
                            cv2.circle(vis_img, pt2, 4, (255, 0, 0), -1)

                        _, buffer_jpg = cv2.imencode('.jpg', vis_img)
                        vis_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_jpg).decode('utf-8')}"

        if is_tampered:
            score = max(20.0, 100.0 - (len(flagged_boxes) * 35.0))
            details = f"Detected {len(flagged_boxes)} duplicated image regions with high spatial keypoint coherence"
        else:
            score = 100.0
            details = "No cloned texture or copy-move forgery patterns identified"

        return {
            "copy_move_detected": is_tampered,
            "score": round(score, 1),
            "matched_pair_count": len(valid_pairs),
            "flagged_boxes": flagged_boxes,
            "visualization_base64": vis_base64,
            "details": details
        }
