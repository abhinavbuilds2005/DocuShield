"""
Copy-Move Forgery Detection Module
Identifies duplicated or cloned image regions using keypoint spatial clustering.
Masks standard text form lines to avoid false-positive matches on recurring typographical characters.

Improvements over v1:
- Self-match elimination (descriptor cannot match itself)
- Lowe's ratio test for match quality
- RANSAC geometric verification
- Dynamic banner masking (proportional, not hardcoded)
"""

import cv2
import numpy as np
import base64
from typing import Dict, Any, List, Tuple


class CopyMoveDetector:
    """Detects duplicated image regions using keypoint spatial displacement analysis."""

    def __init__(self, min_matches: int = 15, min_distance_px: float = 70.0):
        self.min_matches = min_matches
        self.min_distance_px = min_distance_px
        self.orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)

    def analyze(self, image_path_or_array, text_boxes: List[List[int]] = None) -> Dict[str, Any]:
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

        # Dynamic mask: exclude top/bottom proportional regions (banners)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        top_banner = max(1, int(h * 0.20))     # Top 20% (header banner)
        bottom_banner = max(1, int(h * 0.05))   # Bottom 5% (footer)
        mask[0:top_banner, :] = 0
        mask[h-bottom_banner:h, :] = 0

        # Mask standard form body text boxes to prevent recurring letter false positives
        if text_boxes:
            for b in text_boxes:
                bx, by, bw, bh = b[:4]
                # Mask small/medium body text lines, leave graphic patches (QR, stamps, photos)
                if bh < 40 and bw < (w * 0.85):
                    pad = 2
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

        # Match descriptors against themselves, but with self-match elimination
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            # Request k=4 to have room after removing self-matches
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
            # Filter out self-matches (where queryIdx == trainIdx)
            non_self = [m for m in m_list if m.queryIdx != m.trainIdx]
            if len(non_self) < 2:
                continue

            best = non_self[0]
            second_best = non_self[1]

            # Lowe's ratio test — best match must be significantly better than second best
            if best.distance > 0.75 * second_best.distance:
                continue

            kp1 = keypoints[best.queryIdx]
            kp2 = keypoints[best.trainIdx]
            dist_px = np.hypot(kp1.pt[0] - kp2.pt[0], kp1.pt[1] - kp2.pt[1])

            # Must be separated by at least min_distance_px
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

            max_cluster_key = max(bins.keys(), key=lambda k: len(bins[k]))
            max_cluster = bins[max_cluster_key]

            if len(max_cluster) >= self.min_matches:
                pts_src = np.array([valid_pairs[i][0].pt for i in max_cluster])
                pts_dst = np.array([valid_pairs[i][1].pt for i in max_cluster])

                # RANSAC geometric verification
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
                            if inlier_count >= 8 and inlier_ratio > 0.5:
                                is_geometrically_valid = True
                                # Use only inliers
                                inlier_indices = [max_cluster[i] for i in range(len(max_cluster))
                                                 if inlier_mask[i]]
                                pts_src = np.array([valid_pairs[i][0].pt for i in inlier_indices])
                                pts_dst = np.array([valid_pairs[i][1].pt for i in inlier_indices])
                                max_cluster = inlier_indices
                    except Exception:
                        pass

                # Fallback: if RANSAC didn't run (< 4 points), use spread check
                if not is_geometrically_valid and len(pts_src) >= self.min_matches:
                    spread_x1 = np.ptp(pts_src[:, 0])
                    spread_y1 = np.ptp(pts_src[:, 1])
                    spread_x2 = np.ptp(pts_dst[:, 0])
                    spread_y2 = np.ptp(pts_dst[:, 1])
                    # Requires genuine 2D area (at least 30px in both X and Y)
                    if (spread_x1 > 30 and spread_y1 > 30) and (spread_x2 > 30 and spread_y2 > 30):
                        is_geometrically_valid = True

                if is_geometrically_valid:
                    is_tampered = True
                    x_min1, y_min1 = np.min(pts_src, axis=0)
                    x_max1, y_max1 = np.max(pts_src, axis=0)
                    x_min2, y_min2 = np.min(pts_dst, axis=0)
                    x_max2, y_max2 = np.max(pts_dst, axis=0)

                    pad = 10
                    b1 = [max(0, int(x_min1 - pad)), max(0, int(y_min1 - pad)),
                          min(w, int(x_max1 - x_min1 + 2 * pad)), min(h, int(y_max1 - y_min1 + 2 * pad))]
                    b2 = [max(0, int(x_min2 - pad)), max(0, int(y_min2 - pad)),
                          min(w, int(x_max2 - x_min2 + 2 * pad)), min(h, int(y_max2 - y_min2 + 2 * pad))]

                    # Reject internal repetitive patterns within a single compact graphic (e.g. QR code finder squares)
                    overall_w = max(b1[0] + b1[2], b2[0] + b2[2]) - min(b1[0], b2[0])
                    overall_h = max(b1[1] + b1[3], b2[1] + b2[3]) - min(b1[1], b2[1])
                    if overall_w < 120 and overall_h < 160:
                        is_tampered = False
                    else:
                        flagged_boxes.append({
                            "box": b1,
                            "score": 0.92,
                            "label": "Copy-Move Source Region",
                            "reason": f"Cloned graphic patch verified with {len(max_cluster)} spatially coherent keypoints"
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
