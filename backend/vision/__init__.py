"""
DocuShield AI — Vision Module Package
Exports modular field detectors for supported document types.
"""

from backend.vision.field_detector import AadhaarFieldDetector, get_field_detector
from backend.vision.pan_field_detector import PANFieldDetector, get_pan_field_detector
from backend.vision.dl_field_detector import DLFieldDetector, get_dl_field_detector

__all__ = [
    "AadhaarFieldDetector",
    "get_field_detector",
    "PANFieldDetector",
    "get_pan_field_detector",
    "DLFieldDetector",
    "get_dl_field_detector",
]
