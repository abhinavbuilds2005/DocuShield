"""
Multimodal Document Authenticity Fusion Engine
Combines Text/NLP Field Validation with Image Forensics (ELA, Font Alignment, Copy-Move, Metadata)
Produces:
- Calibrated Authenticity Score (0 - 100%)
- Categorical Verdict: AUTHENTIC, SUSPICIOUS, FLAGGED / TAMPERED
- Unified list of flagged bounding boxes with layer attribution and explainable reasons
- Evidence strength classification per signal
- Comprehensive signal breakdown for transparency
"""

import cv2
import numpy as np
from typing import Dict, Any, List

from backend.nlp.ocr_engine import OCREngine
from backend.nlp.field_validator import DocumentFieldValidator
from backend.forensics.ela import ErrorLevelAnalysis
from backend.forensics.copy_move import CopyMoveDetector
from backend.forensics.font_alignment import FontAlignmentForensics
from backend.forensics.metadata_checker import MetadataForensics


def _classify_evidence_strength(score: float) -> str:
    """Canonical evidence strength classification across all detectors and debug reports.
    Always returns uppercase: CLEAN, WEAK, MODERATE, STRONG.
    """
    if score >= 90:
        return "CLEAN"
    elif score >= 70:
        return "WEAK"
    elif score >= 50:
        return "MODERATE"
    else:
        return "STRONG"


class DocumentScreeningPipeline:
    """End-to-end multi-layer document forgery detection pipeline."""

    def __init__(self):
        self.ocr_engine = OCREngine()
        self.field_validator = DocumentFieldValidator()
        self.ela_analyzer = ErrorLevelAnalysis()
        self.copy_move_detector = CopyMoveDetector()
        self.font_analyzer = FontAlignmentForensics()
        self.metadata_checker = MetadataForensics()

    def screen_document(self, image_path: str, benchmark_mode: bool = False) -> Dict[str, Any]:
        """
        Executes complete screening pipeline on a given document image path.
        
        Args:
            image_path: Path to the document image file
            benchmark_mode: If False (default), uses ONLY real OCR.
                          If True, may use sidecar OCR for benchmark testing.
        """
        # Load image with OpenCV
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Unable to read image at {image_path}")

        h, w = img_bgr.shape[:2]

        # 1. Layer 1: OCR and NLP Field Validation
        # benchmark_mode flag is propagated to OCR engine — this is the
        # critical gate that prevents sidecar/ground-truth leakage.
        ocr_result = self.ocr_engine.process_image(image_path, benchmark_mode=benchmark_mode)
        nlp_result = self.field_validator.validate(ocr_result)

        # 2. Layer 2: Image Forensics
        ela_result = self.ela_analyzer.analyze(img_bgr)
        text_boxes = [l.get("box") for l in ocr_result.get("lines", []) if "box" in l]
        copy_move_result = self.copy_move_detector.analyze(img_bgr, text_boxes=text_boxes)
        font_result = self.font_analyzer.analyze(img_bgr, ocr_result.get("tokens", []))
        metadata_result = self.metadata_checker.analyze(image_path)

        # 3. Aggregate Flagged Bounding Boxes
        flagged_regions = []

        # Add ELA boxes
        for b in ela_result.get("flagged_boxes", []):
            flagged_regions.append({
                "box": b["box"],
                "layer": "Image Forensics (ELA)",
                "label": b["label"],
                "score": b["score"],
                "color": "#ef4444", # Red
                "reason": b["reason"]
            })

        # Add Copy-Move boxes
        for b in copy_move_result.get("flagged_boxes", []):
            flagged_regions.append({
                "box": b["box"],
                "layer": "Image Forensics (Copy-Move)",
                "label": b["label"],
                "score": b["score"],
                "color": "#f97316", # Orange
                "reason": b["reason"]
            })

        # Add Font Discrepancy boxes
        for b in font_result.get("flagged_boxes", []):
            flagged_regions.append({
                "box": b["box"],
                "layer": "Typography / Rendering Anomaly",
                "label": b["label"],
                "score": b["score"],
                "color": "#eab308", # Yellow
                "reason": b["reason"]
            })

        # Add NLP Failed Field boxes if matched to OCR token coordinates
        for field in nlp_result.get("fields", []):
            if field["status"] == "FAIL":
                # Find corresponding token in OCR if any
                target_val = field["value"]
                matched_box = None
                for t in ocr_result.get("tokens", []) + ocr_result.get("lines", []):
                    clean_t = t.get("text", "").replace(" ", "").upper()
                    clean_v = target_val.replace(" ", "").upper()
                    if clean_v in clean_t or clean_t in clean_v:
                        matched_box = t.get("box")
                        break
                
                if matched_box:
                    flagged_regions.append({
                        "box": matched_box,
                        "layer": "NLP Field Validation",
                        "label": f"Invalid Field: {field['field']}",
                        "score": 0.95,
                        "color": "#ec4899", # Pink
                        "reason": field["details"]
                    })

        # Deduplicate overlapping bounding boxes
        deduped_regions = self._deduplicate_boxes(flagged_regions)

        # 4. Multimodal Fusion Score Calculation
        score_nlp = nlp_result.get("score", 100.0)
        score_ela = ela_result.get("ela_authenticity_score", 100.0)
        score_cm = copy_move_result.get("score", 100.0)
        score_font = font_result.get("font_consistency_score", 100.0)
        score_meta = metadata_result.get("metadata_score", 100.0)

        # Base weighted calculation
        base_weighted_sum = round(
            score_nlp * 0.30 +
            score_ela * 0.25 +
            score_font * 0.20 +
            score_cm * 0.15 +
            score_meta * 0.10,
            2
        )

        # Categorize evidence per detector: NONE, WEAK, MODERATE, STRONG
        detector_evidence = {}
        strong_deterministic_signals = []
        moderate_signals = []
        weak_signals = []
        weak_or_uncertain_reasons = []

        # (a) NLP Field Validation
        failed_fields = [f for f in nlp_result.get("fields", []) if f.get("status") == "FAIL"]
        uncertain_fields = [f for f in nlp_result.get("fields", []) if f.get("status") == "UNCERTAIN" or f.get("evidence_level") == "WEAK"]
        if any(f.get("is_deterministic") and f.get("status") == "FAIL" for f in failed_fields):
            detector_evidence["nlp"] = {"strength": "STRONG", "deterministic": True, "uncertain": False}
            strong_deterministic_signals.append("Deterministic Field Validation Failure (mathematically invalid checksum/format with confident OCR)")
        elif failed_fields:
            detector_evidence["nlp"] = {"strength": _classify_evidence_strength(score_nlp), "deterministic": False, "uncertain": False}
            moderate_signals.append("Field validation / chronology mismatch")
        elif uncertain_fields:
            detector_evidence["nlp"] = {"strength": "WEAK", "deterministic": False, "uncertain": True}
            weak_signals.append("NLP ambiguity / low field OCR confidence")
            weak_or_uncertain_reasons.append("NLP: field validation inconclusive due to low OCR confidence or minor token outlier")
        else:
            detector_evidence["nlp"] = {"strength": "CLEAN", "deterministic": False, "uncertain": False}

        # (b) ELA Forensics & Boundary Cut Seams
        # Note: ELA is an indicator subject to compression/resizing; reserve 'deterministic=True'
        # strictly for definitive physical boundary seams with high step contrast.
        ela_boxes = ela_result.get("flagged_boxes", [])
        if ela_result.get("is_suspicious") and len(ela_boxes) >= 1:
            has_seam = any(
                "cut seam" in b.get("reason", "").lower() or 
                "boundary seam" in b.get("label", "").lower()
                for b in ela_boxes
            )
            has_high_ela = any(b.get("score", 0.0) >= 0.92 for b in ela_boxes)
            if has_seam:
                detector_evidence["ela"] = {"strength": "STRONG", "deterministic": True, "uncertain": False}
                strong_deterministic_signals.append("Portrait boundary cut seam identified")
            elif has_high_ela or len(ela_boxes) >= 2:
                # Strong forensic indicator, but not mathematically deterministic
                detector_evidence["ela"] = {"strength": "STRONG", "deterministic": False, "uncertain": False}
                strong_deterministic_signals.append("High-error compression anomaly hotspot in card substrate")
            else:
                detector_evidence["ela"] = {"strength": _classify_evidence_strength(score_ela), "deterministic": False, "uncertain": False}
                moderate_signals.append("Localized compression anomaly in card substrate")
        else:
            detector_evidence["ela"] = {"strength": "CLEAN", "deterministic": False, "uncertain": False}

        # (c) Font / Typography Forensics
        font_boxes = font_result.get("flagged_boxes", [])
        if font_result.get("anomalies_detected") and len(font_boxes) >= 1:
            has_strong_seam = any(
                "cut seam" in b.get("reason", "").lower() or 
                "boundary discontinuity" in b.get("reason", "").lower()
                for b in font_boxes
            )
            has_high_sharpness = any(b.get("score", 0.0) >= 0.86 for b in font_boxes)
            if has_strong_seam:
                detector_evidence["font"] = {"strength": "STRONG", "deterministic": True, "uncertain": False}
                strong_deterministic_signals.append("Typography cut-seam boundary discontinuity identified")
            elif has_high_sharpness:
                detector_evidence["font"] = {"strength": "STRONG", "deterministic": False, "uncertain": False}
                strong_deterministic_signals.append("Significant typography stroke sharpness disparity")
            else:
                detector_evidence["font"] = {"strength": _classify_evidence_strength(score_font), "deterministic": False, "uncertain": False}
                moderate_signals.append("Moderate typographical rendering variance")
        else:
            detector_evidence["font"] = {"strength": "CLEAN", "deterministic": False, "uncertain": False}

        # (d) Copy-Move Forensics
        if copy_move_result.get("copy_move_detected"):
            cm_boxes = copy_move_result.get("flagged_boxes", [])
            if len(cm_boxes) >= 2:
                detector_evidence["copy_move"] = {"strength": "STRONG", "deterministic": True, "uncertain": False}
                strong_deterministic_signals.append("Verified duplicate image region / cloned graphic patch (RANSAC verified)")
            else:
                detector_evidence["copy_move"] = {"strength": _classify_evidence_strength(score_cm), "deterministic": False, "uncertain": False}
                moderate_signals.append("Copy-move spatial cluster detected")
        else:
            detector_evidence["copy_move"] = {"strength": "CLEAN", "deterministic": False, "uncertain": False}

        # (e) Metadata Forensics
        if metadata_result.get("is_suspicious"):
            sw = metadata_result.get("detected_software")
            if sw:
                detector_evidence["metadata"] = {"strength": "STRONG", "deterministic": True, "uncertain": False}
                strong_deterministic_signals.append(f"Editing software signature identified ({sw})")
            else:
                detector_evidence["metadata"] = {"strength": "WEAK", "deterministic": False, "uncertain": True}
                weak_signals.append("Missing EXIF metadata tags")
                weak_or_uncertain_reasons.append("Metadata: generic missing EXIF / camera tags (common in web uploads)")
        else:
            detector_evidence["metadata"] = {"strength": "CLEAN", "deterministic": False, "uncertain": False}

        # -------------------------------------------------------------
        # Categorical Diagnostic Judgment:
        # CLEAN vs ANOMALY_DETECTED vs DOCUMENT_STRONGLY_SUSPECTED_TAMPERED
        # -------------------------------------------------------------
        if len(strong_deterministic_signals) >= 1 or len(moderate_signals) >= 2:
            diagnostic_status = "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"
        elif len(moderate_signals) == 1 or len(weak_signals) >= 1:
            diagnostic_status = "ANOMALY_DETECTED"
        else:
            diagnostic_status = "CLEAN"

        # -------------------------------------------------------------
        # Confidence-Aware Evidence Fusion Scoring:
        # -------------------------------------------------------------
        rules_triggered = []
        if diagnostic_status == "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED":
            # Multi-layer corroboration or multiple strong signals: FLAGGED / TAMPERED
            if len(strong_deterministic_signals) >= 2 or (len(strong_deterministic_signals) >= 1 and len(moderate_signals) >= 1) or len(moderate_signals) >= 3:
                fused_score = min(base_weighted_sum - 35.0, 42.0)
                rules_triggered.append("Multiple corroborated strong/moderate tampering signals -> Score capped at FLAGGED / TAMPERED")
            else:
                # Single strong deterministic finding (or 2 moderate signals):
                # SUFFICIENT ON ITS OWN to reach DOCUMENT_STRONGLY_SUSPECTED_TAMPERED,
                # ensuring score is at least SUSPICIOUS (score <= 68.0, risk >= 32.0).
                fused_score = min(base_weighted_sum - 25.0, 68.0)
                rules_triggered.append("Single deterministic/high-confidence integrity failure -> Document strongly suspected tampered (Score capped at SUSPICIOUS)")

        elif diagnostic_status == "ANOMALY_DETECTED":
            if len(moderate_signals) == 1:
                # Single uncorroborated moderate anomaly -> SUSPICIOUS review recommended
                fused_score = min(max(52.0, base_weighted_sum - 18.0), 74.0)
                rules_triggered.append("Single unconfirmed moderate anomaly -> Secondary manual review recommended")
            else:
                # Isolated WEAK / uncertain anomaly (e.g. minor OCR noise, normal JPEG recompression, missing EXIF):
                # Must NOT penalize document into SUSPICIOUS! Retains AUTHENTIC (>= 80.0) with diagnostic note.
                fused_score = max(82.0, base_weighted_sum - 5.0)
                rules_triggered.append("Isolated weak anomaly observed; insufficient independent corroboration for tampering verdict -> Retains AUTHENTIC")

        else: # CLEAN
            fused_score = max(88.0, min(100.0, base_weighted_sum))
            rules_triggered.append("All forensic layers clean and consistent -> AUTHENTIC")

        fused_score = round(max(0.0, min(100.0, fused_score)), 1)
        risk_score = round(max(0.0, min(100.0, 100.0 - fused_score)), 1)

        # Categorical Verdict
        if fused_score >= 80.0:
            verdict = "AUTHENTIC"
            verdict_color = "emerald"
            verdict_description = "Document exhibits uniform compression, valid checksum algorithms, consistent typography, and authentic layout structures."
            if diagnostic_status == "ANOMALY_DETECTED":
                verdict_description += " (Minor isolated anomaly noted; insufficient independent corroboration for tampering verdict)."
        elif fused_score >= 50.0:
            verdict = "SUSPICIOUS"
            verdict_color = "amber"
            verdict_description = "Document exhibits localized anomalies or format discrepancies. Secondary manual review recommended."
        else:
            verdict = "FLAGGED / TAMPERED"
            verdict_color = "rose"
            verdict_description = "High-confidence detection of digital forgery, text splicing, photo swap, or invalid identity credentials."

        # Classify evidence strength per signal (harmonized across all layers and debug reports)
        evidence_strengths = {
            "nlp": detector_evidence["nlp"]["strength"],
            "ela": detector_evidence["ela"]["strength"],
            "font": detector_evidence["font"]["strength"],
            "copy_move": detector_evidence["copy_move"]["strength"],
            "metadata": detector_evidence["metadata"]["strength"],
        }

        decision_threshold = {
            "authenticity_threshold": 80.0,
            "risk_threshold": 20.0,
            "suspicious_threshold": 50.0,
            "rule": "score >= 80.0 (risk <= 20.0) -> AUTHENTIC; 50.0 <= score < 80.0 -> SUSPICIOUS; score < 50.0 -> FLAGGED / TAMPERED"
        }

        # Structured OCR extraction info
        ocr_tokens = ocr_result.get("tokens", [])
        ocr_token_count = len(ocr_tokens)
        ocr_status = "SUCCESS" if ocr_token_count > 0 else ("EMPTY" if ocr_result.get("full_text") is not None else "FAILED")
        avg_ocr_conf = round(sum(t.get("confidence", 0.0) for t in ocr_tokens) / max(1, ocr_token_count), 4) if ocr_tokens else 0.0
        ocr_extraction_info = {
            "status": ocr_status,
            "engine": ocr_result.get("ocr_engine", "unknown"),
            "confidence": avg_ocr_conf,
            "token_count": ocr_token_count,
            "line_count": len(ocr_result.get("lines", []))
        }

        # Fusion contributions breakdown
        fusion_contributions = {
            "nlp": {"weight": 0.30, "raw_score": round(score_nlp, 1), "weighted_score": round(score_nlp * 0.30, 2), "evidence_strength": detector_evidence["nlp"]["strength"]},
            "ela": {"weight": 0.25, "raw_score": round(score_ela, 1), "weighted_score": round(score_ela * 0.25, 2), "evidence_strength": detector_evidence["ela"]["strength"]},
            "font": {"weight": 0.20, "raw_score": round(score_font, 1), "weighted_score": round(score_font * 0.20, 2), "evidence_strength": detector_evidence["font"]["strength"]},
            "copy_move": {"weight": 0.15, "raw_score": round(score_cm, 1), "weighted_score": round(score_cm * 0.15, 2), "evidence_strength": detector_evidence["copy_move"]["strength"]},
            "metadata": {"weight": 0.10, "raw_score": round(score_meta, 1), "weighted_score": round(score_meta * 0.10, 2), "evidence_strength": detector_evidence["metadata"]["strength"]},
            "base_weighted_sum": base_weighted_sum,
            "penalties_applied": rules_triggered,
            "diagnostic_status": diagnostic_status
        }

        # Determine checksum validation status for top-level report
        checksum_fields = [
            f for f in nlp_result.get("fields", [])
            if "aadhaar" in f.get("field", "").lower() or "national id" in f.get("field", "").lower()
        ]
        if checksum_fields:
            if any(f.get("status") == "FAIL" for f in checksum_fields):
                chk_status = "FAIL"
            elif any(f.get("status") == "UNCERTAIN" for f in checksum_fields):
                chk_status = "UNCERTAIN"
            else:
                chk_status = "PASS"
            chk_deterministic = any(f.get("is_deterministic") and f.get("status") == "FAIL" for f in checksum_fields)
        else:
            chk_status = "N/A"
            chk_deterministic = False

        # Complete Forensic Decision Debug Report
        forensic_decision_debug = {
            "final_verdict": verdict,
            "final_risk_score": risk_score,
            "confidence": round(avg_ocr_conf if ocr_token_count > 0 else 0.90, 4),
            "diagnostic_status": diagnostic_status,
            "ocr_status_and_confidence": ocr_extraction_info,
            "checksum_result": {
                "status": chk_status,
                "deterministic": chk_deterministic,
                "fields": checksum_fields
            },
            "regex_field_validation_result": {
                "document_type": nlp_result.get("document_type"),
                "fields_checked": len(nlp_result.get("fields", [])),
                "failed_fields": failed_fields
            },
            "ela_result_and_score": {
                "score": round(score_ela, 1),
                "evidence_strength": detector_evidence["ela"]["strength"],
                "hotspot_count": len(ela_boxes),
                "is_suspicious": ela_result.get("is_suspicious", False)
            },
            "copy_move_result_and_score": {
                "score": round(score_cm, 1),
                "evidence_strength": detector_evidence["copy_move"]["strength"],
                "detected": copy_move_result.get("copy_move_detected", False)
            },
            "typography_rendering_result_and_score": {
                "score": round(score_font, 1),
                "evidence_strength": detector_evidence["font"]["strength"],
                "anomalies_detected": font_result.get("anomalies_detected", False),
                "flagged_count": len(font_boxes)
            },
            "metadata_result": {
                "score": round(score_meta, 1),
                "evidence_strength": detector_evidence["metadata"]["strength"],
                "software": metadata_result.get("detected_software"),
                "is_suspicious": metadata_result.get("is_suspicious", False)
            },
            "detector_contributions": fusion_contributions,
            "triggered_signals": strong_deterministic_signals + moderate_signals + weak_signals,
            "rules_triggered": rules_triggered,
            "decision_threshold": decision_threshold,
            "weak_or_uncertain_signals": weak_or_uncertain_reasons,
            "is_anomaly_detected": diagnostic_status == "ANOMALY_DETECTED",
            "is_strongly_suspected_tampered": diagnostic_status == "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"
        }

        # Compile comprehensive response
        return {
            "authenticity_score": fused_score,
            "risk_score": risk_score,
            "verdict": verdict,
            "verdict_color": verdict_color,
            "diagnostic_status": diagnostic_status,
            "decision_threshold": decision_threshold,
            "ocr_extraction": ocr_extraction_info,
            "fusion_contributions": fusion_contributions,
            "forensic_decision_debug": forensic_decision_debug,
            "summary_explanation": verdict_description,
            "critical_triggers": strong_deterministic_signals + moderate_signals,
            "image_dimensions": {"width": w, "height": h},
            "flagged_regions": deduped_regions,
            "evidence_strengths": evidence_strengths,
            "ocr_engine_used": ocr_result.get("ocr_engine", "unknown"),
            "visualizations": {
                "ela_heatmap": ela_result.get("heatmap_data_uri"),
                "edge_gradient_map": font_result.get("edge_map_base64"),
                "copy_move_matches": copy_move_result.get("visualization_base64")
            },
            "signals": {
                "nlp_validation": {
                    "layer_name": "Text / NLP Field Validation",
                    "score": round(score_nlp, 1),
                    "evidence_strength": evidence_strengths["nlp"],
                    "document_type": nlp_result.get("document_type"),
                    "extracted_full_text": ocr_result.get("full_text", ""),
                    "field_checks": nlp_result.get("fields", []),
                    "reasons": nlp_result.get("reasons", [])
                },
                "ela_forensics": {
                    "layer_name": "Error Level Analysis (ELA)",
                    "score": round(score_ela, 1),
                    "evidence_strength": evidence_strengths["ela"],
                    "hotspot_count": ela_result.get("hotspot_count", 0),
                    "metrics": ela_result.get("metrics", {}),
                    "format_note": ela_result.get("format_note", "")
                },
                "font_typography": {
                    "layer_name": "Typography / Rendering Anomaly",
                    "score": round(score_font, 1),
                    "evidence_strength": evidence_strengths["font"],
                    "details": font_result.get("details", "")
                },
                "copy_move": {
                    "layer_name": "Copy-Move Forgery Detection",
                    "score": round(score_cm, 1),
                    "evidence_strength": evidence_strengths["copy_move"],
                    "detected": copy_move_result.get("copy_move_detected", False),
                    "details": copy_move_result.get("details", "")
                },
                "metadata_forensics": {
                    "layer_name": "EXIF & Metadata Inspection",
                    "score": round(score_meta, 1),
                    "evidence_strength": evidence_strengths["metadata"],
                    "detected_software": metadata_result.get("detected_software"),
                    "reasons": metadata_result.get("reasons", [])
                }
            }
        }

    def _deduplicate_boxes(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Removes or merges near-identical overlapping bounding boxes."""
        if not regions:
            return []

        deduped = []
        for reg in regions:
            bx1, by1, bw1, bh1 = reg["box"]
            is_dup = False
            for existing in deduped:
                bx2, by2, bw2, bh2 = existing["box"]
                # Compute Intersection over Union (IoU)
                xi1 = max(bx1, bx2)
                yi1 = max(by1, by2)
                xi2 = min(bx1 + bw1, bx2 + bw2)
                yi2 = min(by1 + bh1, by2 + bh2)
                
                iw = max(0, xi2 - xi1)
                ih = max(0, yi2 - yi1)
                intersection = iw * ih
                union = (bw1 * bh1) + (bw2 * bh2) - intersection
                iou = intersection / (union + 1e-5)
                
                if iou > 0.45:
                    is_dup = True
                    # If current has higher severity/score, update reason
                    if reg.get("score", 0) > existing.get("score", 0):
                        existing["reason"] += f" | {reg['reason']}"
                        existing["label"] = f"{existing['label']} + {reg['label']}"
                    break

            if not is_dup:
                deduped.append(reg)

        return deduped
