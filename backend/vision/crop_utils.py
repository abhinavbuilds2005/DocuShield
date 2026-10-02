"""
DocuShield AI — Crop Extraction, Geometry, and Preprocessing Utilities
Provides safe, modular utilities for:
1. Bounded padded field crop extraction with coordinate clamping.
2. Controlled aspect-ratio-preserving upscaling for low-resolution text crops.
3. Bounded image preprocessing variants (original, CLAHE, sharpening) for OCR assistance.

Strict safety constraints:
- Clamps coordinates to image boundaries [0, W] and [0, H].
- Rejects invalid, inverted, or empty bounding boxes without crashing.
- Never mutates the original input image.
- Upscales small crops only; never upscales full document images.
- Enforces hard limits on variants (max 3) and dimensions (max 800px).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np


def extract_padded_crop(
    img: np.ndarray,
    bbox: Union[Dict[str, Any], List[Union[int, float]], Tuple[Union[int, float], ...]],
    pad_ratio: float = 0.08,
    min_pad_px: int = 4,
    max_pad_px: int = 24
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Extracts a padded field crop from an image array while safely clamping
    coordinates to image dimensions.

    Args:
        img: Input image as numpy array (H, W, C) or (H, W).
        bbox: Bounding box in one of the following formats:
              - Dict: {"x1": int, "y1": int, "x2": int, "y2": int}
              - Dict: {"x": float, "y": float, "width": float, "height": float} (pixel or norm)
              - List/Tuple of 4 elements: [x1, y1, x2, y2] or [x, y, w, h]
        pad_ratio: Fraction of box dimension to add as margin on each side (default: 8%).
        min_pad_px: Minimum padding in pixels (default: 4px).
        max_pad_px: Maximum padding in pixels to avoid over-expansion (default: 24px).

    Returns:
        Tuple of (cropped_image_array, metadata_dict).
        If bbox or image is invalid, returns (None, {"error": reason}).
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0 or len(img.shape) < 2:
        return None, {"error": "INVALID_IMAGE", "message": "Image array is empty or invalid"}

    img_h, img_w = img.shape[:2]
    if img_h <= 0 or img_w <= 0:
        return None, {"error": "INVALID_IMAGE_DIMENSIONS", "message": f"Invalid image size {img_w}x{img_h}"}

    # 1. Parse bounding box coordinates to (x1, y1, x2, y2) in pixels
    x1, y1, x2, y2 = None, None, None, None

    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        elif all(k in bbox for k in ("x", "y", "width", "height")):
            bx, by, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            # Detect normalized coordinates [0.0, 1.0]
            if 0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0 and bw <= 1.0 and bh <= 1.0 and (img_w > 1 and img_h > 1):
                x1 = int(bx * img_w)
                y1 = int(by * img_h)
                x2 = int((bx + bw) * img_w)
                y2 = int((by + bh) * img_h)
            else:
                x1, y1, x2, y2 = int(bx), int(by), int(bx + bw), int(by + bh)

    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        b0, b1, b2, b3 = bbox
        # Disambiguate [x, y, w, h] vs [x1, y1, x2, y2]
        if b2 > b0 and b3 > b1:
            x1, y1, x2, y2 = int(b0), int(b1), int(b2), int(b3)
        else:
            x1, y1, x2, y2 = int(b0), int(b1), int(b0 + b2), int(b1 + b3)

    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None, {"error": "INVALID_BBOX_FORMAT", "message": f"Unsupported bbox format: {type(bbox)}"}

    # 2. Validate geometric sanity
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return None, {"error": "INVERTED_OR_ZERO_AREA_BOX", "message": f"Box has non-positive area: [{x1}, {y1}, {x2}, {y2}]"}

    # Check that box overlaps at least partially with the image
    if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
        return None, {"error": "BOX_OUT_OF_BOUNDS", "message": f"Box [{x1}, {y1}, {x2}, {y2}] is entirely outside image {img_w}x{img_h}"}

    raw_w = x2 - x1
    raw_h = y2 - y1

    # 3. Calculate bounded padding
    pad_x = max(min_pad_px, min(max_pad_px, int(raw_w * pad_ratio)))
    pad_y = max(min_pad_px, min(max_pad_px, int(raw_h * pad_ratio)))

    # Apply padding and clamp strictly to [0, W] and [0, H]
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(img_w, x2 + pad_x)
    py2 = min(img_h, y2 + pad_y)

    if px2 <= px1 or py2 <= py1:
        return None, {"error": "CLAMPED_BOX_EMPTY", "message": "Padded box collapsed during boundary clamping"}

    # 4. Extract safe copy of crop (do not mutate original image)
    crop = img[py1:py2, px1:px2].copy()

    metadata = {
        "original_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": raw_w, "height": raw_h},
        "padded_bbox": {"x1": px1, "y1": py1, "x2": px2, "y2": py2, "width": px2 - px1, "height": py2 - py1},
        "padding_applied": {"pad_x": pad_x, "pad_y": pad_y},
        "crop_shape": list(crop.shape),
        "image_dims": [img_w, img_h]
    }

    return crop, metadata


def upscale_crop_if_needed(
    crop_img: np.ndarray,
    min_height: int = 64,
    max_scale: float = 3.0,
    max_dimension: int = 800
) -> Tuple[np.ndarray, float]:
    """
    Conditionally upscales small field crops (e.g. height < 64px) to improve OCR
    character recognition while preserving aspect ratio and preventing excessive RAM usage.

    Args:
        crop_img: Input crop array.
        min_height: Target minimum height for text legibility (default: 64px).
        max_scale: Upper bound on scaling factor (default: 3.0x).
        max_dimension: Maximum allowed width or height after upscaling (default: 800px).

    Returns:
        Tuple of (upscaled_image, scale_factor).
        If no upscaling was required or crop is invalid, returns (crop_img, 1.0).
    """
    if crop_img is None or not isinstance(crop_img, np.ndarray) or crop_img.size == 0:
        return crop_img, 1.0

    ch, cw = crop_img.shape[:2]
    if ch <= 0 or cw <= 0:
        return crop_img, 1.0

    # If already sufficiently large, return without modification
    if ch >= min_height:
        return crop_img, 1.0

    # Determine desired scale factor to bring height close to min_height
    scale = float(min_height) / float(ch)

    # Standardize to 2.0x or 3.0x bounded by max_scale
    if scale > 2.2:
        scale = min(max_scale, 3.0)
    else:
        scale = min(max_scale, 2.0)

    # Ensure max dimension guard is respected
    if int(cw * scale) > max_dimension or int(ch * scale) > max_dimension:
        scale = min(float(max_dimension) / float(cw), float(max_dimension) / float(ch))

    if scale <= 1.05:
        return crop_img, 1.0

    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    upscaled = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return upscaled, round(scale, 2)


def get_crop_preprocessing_variants(
    crop_img: np.ndarray,
    max_variants: int = 3
) -> List[Tuple[str, np.ndarray]]:
    """
    Generates a bounded set of image preprocessing variants for low-confidence field crops:
    1. 'original': Raw crop (or upscaled crop).
    2. 'clahe_contrast': Local contrast enhancement via CLAHE to handle patterned/colored backgrounds.
    3. 'sharpened': Unsharp mask sharpening to enhance faint character strokes.

    Guarantees:
    - Never generates more than `max_variants` (default: 3).
    - Always returns valid BGR or Grayscale uint8 numpy arrays.
    - No unbounded processing loops.
    """
    if crop_img is None or not isinstance(crop_img, np.ndarray) or crop_img.size == 0:
        return []

    variants: List[Tuple[str, np.ndarray]] = []

    # Variant 1: Original
    variants.append(("original", crop_img))
    if max_variants <= 1:
        return variants

    # Variant 2: Contrast-enhanced via CLAHE
    try:
        if len(crop_img.shape) == 3 and crop_img.shape[2] == 3:
            # Convert to LAB color space, apply CLAHE on L-channel, convert back to BGR
            lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            cl = clahe.apply(l_chan)
            limg = cv2.merge((cl, a_chan, b_chan))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(crop_img)
        variants.append(("clahe_contrast", enhanced))
    except Exception:
        pass

    if len(variants) >= max_variants:
        return variants

    # Variant 3: Mild unsharp mask sharpening
    try:
        gaussian = cv2.GaussianBlur(crop_img, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(crop_img, 1.4, gaussian, -0.4, 0)
        variants.append(("sharpened", sharpened))
    except Exception:
        pass

    return variants[:max_variants]
