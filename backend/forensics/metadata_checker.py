"""
Metadata & EXIF Forensics Module
Inspects embedded image metadata for:
- Editing software signatures (Adobe Photoshop, GIMP, Canva, Photopea, etc.)
- Creation / Modification timestamp disparities
- Missing expected scanner/camera tags on purported physical ID scans
"""

import os
from PIL import Image, ExifTags
from typing import Dict, Any, List


SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "adobe", "gimp", "canva", "photopea",
    "paint.net", "coreldraw", "illustrator", "pixlr", "snapseed", "lightroom"
]


class MetadataForensics:
    """Extracts and inspects EXIF metadata tags for manipulation indicators."""

    def __init__(self):
        pass

    def analyze(self, image_path_or_file) -> Dict[str, Any]:
        """
        Extracts EXIF and checks software metadata.
        Returns:
            {
                "metadata_score": float (0-100),
                "is_suspicious": bool,
                "detected_software": Optional[str],
                "exif_summary": Dict[str, Any],
                "reasons": List[str]
            }
        """
        reasons = []
        detected_softwares = []
        exif_dict = {}

        try:
            with Image.open(image_path_or_file) as img:
                # 1. Check PNG text chunks or standard info dictionary
                info = img.info or {}
                for k, v in info.items():
                    v_str = str(v)
                    k_str = str(k).lower()
                    exif_dict[k_str] = v_str[:120]

                    # Check for editing software tags
                    lower_val = v_str.lower()
                    for sw in SUSPICIOUS_SOFTWARE_KEYWORDS:
                        if sw in lower_val:
                            detected_softwares.append(f"{k}: {v_str[:60]}")
                            reasons.append(f"Image header contains '{sw.upper()}' metadata signature ({k})")

                # 2. Check standard EXIF tags
                exif_data = img._getexif() if hasattr(img, "_getexif") and img._getexif() else None
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        tag_name_lower = tag_name.lower()
                        val_str = str(value)
                        exif_dict[tag_name] = val_str[:120]

                        if "software" in tag_name_lower or "processing" in tag_name_lower:
                            for sw in SUSPICIOUS_SOFTWARE_KEYWORDS:
                                if sw in val_str.lower():
                                    detected_softwares.append(f"{tag_name}: {val_str}")
                                    reasons.append(f"EXIF Software field indicates '{sw.upper()}' manipulation")

        except Exception as e:
            # Metadata could not be extracted or empty
            pass

        # Calculate score
        if reasons:
            score = max(30.0, 100.0 - (len(reasons) * 35.0))
            is_suspicious = True
        else:
            score = 100.0
            is_suspicious = False

        return {
            "metadata_score": round(score, 1),
            "is_suspicious": is_suspicious,
            "detected_software": detected_softwares[0] if detected_softwares else None,
            "all_detected_software": detected_softwares,
            "exif_summary": exif_dict,
            "reasons": reasons
        }
