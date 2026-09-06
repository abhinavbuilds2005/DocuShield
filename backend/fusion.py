"""
Multimodal Document Authenticity Fusion Engine
Combines Text/NLP Field Validation with Image Forensics (ELA, Font Alignment, Copy-Move, Metadata)
Enhanced for Real-World Robustness:
- 4-Level Evidence Hierarchy (Level 1: Deterministic, Level 2: Strong, Level 3: Moderate, Level 4: Weak)
- Evidence Correlation Matrix (groups related compression/quality signals to prevent artificial double-counting)
- Multi-family corroboration requirement before escalating to tamper verdicts
- Condition-aware normalization (JPEG quality, blur, perspective, resolution)
- UNCERTAIN / Human-Review routing for ambiguous OCR
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
from backend.forensics.condition_analyzer import DocumentConditionAnalyzer


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
        self.condition_analyzer = DocumentConditionAnalyzer()

    def screen_document(self, image_path: str, benchmark_mode: bool = False) -> Dict[str, Any]:
        """
        Executes complete screening pipeline on a given document image path.
        
        Args:
            image_path: Path to the document image file
            benchmark_mode: If False (default), uses ONLY real OCR.
                          If True, may use sidecar OCR for benchmark testing.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Unable to read image at {image_path}")

        h, w = img_bgr.shape[:2]

        # 0. Condition Assessment (Provides context for physical & digital distortions)
        condition = self.condition_analyzer.analyze(img_bgr)

        # 1. Layer 1: OCR and NLP Field Validation
        ocr_result = self.ocr_engine.process_image(image_path, benchmark_mode=benchmark_mode)
        nlp_result = self.field_validator.validate(ocr_result)

        # 2. Layer 2: Image Forensics (with condition context and text masking)
        text_boxes = [l.get("box") for l in ocr_result.get("lines", []) if "box" in l]
        ela_result = self.ela_analyzer.analyze(img_bgr, text_boxes=text_boxes, condition=condition)
        copy_move_result = self.copy_move_detector.analyze(img_bgr, text_boxes=text_boxes, condition=condition)
        font_result = self.font_analyzer.analyze(img_bgr, ocr_result.get("tokens", []), condition=condition)
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
                "color": "#ef4444",
                "reason": b["reason"]
            })

        # Add Copy-Move boxes
        for b in copy_move_result.get("flagged_boxes", []):
            flagged_regions.append({
                "box": b["box"],
                "layer": "Image Forensics (Copy-Move)",
                "label": b["label"],
                "score": b["score"],
                "color": "#f97316",
                "reason": b["reason"]
            })

        # Add Font Discrepancy boxes
        for b in font_result.get("flagged_boxes", []):
            flagged_regions.append({
                "box": b["box"],
                "layer": "Typography / Rendering Anomaly",
                "label": b["label"],
                "score": b["score"],
                "color": "#eab308",
                "reason": b["reason"]
            })

        # Add NLP Failed Field boxes
        for field in nlp_result.get("fields", []):
            if field["status"] == "FAIL":
                target_val = field.get("value", "")
                matched_box = None
                for t in ocr_result.get("tokens", []) + ocr_result.get("lines", []):
                    clean_t = t.get("text", "").replace(" ", "").upper()
                    clean_v = target_val.replace(" ", "").upper()
                    if clean_v and (clean_v in clean_t or clean_t in clean_v):
                        matched_box = t.get("box")
                        break
                
                if matched_box:
                    flagged_regions.append({
                        "box": matched_box,
                        "layer": "NLP Field Validation",
                        "label": f"Invalid Field: {field['field']}",
                        "score": 0.95,
                        "color": "#ec4899",
                        "reason": field["details"]
                    })

        deduped_regions = self._deduplicate_boxes(flagged_regions)

        # 4. Multimodal Fusion Score Calculation
        score_nlp = nlp_result.get("score", 100.0)
        score_ela = ela_result.get("ela_authenticity_score", 100.0)
        score_cm = copy_move_result.get("score", 100.0)
        score_font = font_result.get("font_consistency_score", 100.0)
        score_meta = metadata_result.get("metadata_score", 100.0)

        base_weighted_sum = round(
            score_nlp * 0.30 +
            score_ela * 0.25 +
            score_font * 0.20 +
            score_cm * 0.15 +
            score_meta * 0.10,
            2
        )

        # -------------------------------------------------------------
        # 5-Level Evidence Hierarchy & Correlation Grouping
        # -------------------------------------------------------------
        # LEVEL 1: NORMAL IMAGE ARTIFACT (WhatsApp, blur, screenshot, DCT grid)
        # LEVEL 2: WEAK FORENSIC ANOMALY (isolated typography variation, mild ELA hotspot)
        # LEVEL 3: MODERATE FORENSIC ANOMALY (strong localized ELA, typography inconsistency)
        # LEVEL 4: STRONG FORENSIC EVIDENCE (verified splice seam, RANSAC copy-move, strong discontinuity)
        # LEVEL 5: DETERMINISTIC EVIDENCE (confident Verhoeff checksum failure, invalid PAN, editing software)

        # Evidence Families:
        # 1. COMPRESSION FAMILY: ELA, JPEG/DCT condition
        # 2. STRUCTURAL FAMILY: boundary seam, copy-move, typography
        # 3. CONTENT INTEGRITY FAMILY: checksum, PAN validation, date validation, OCR field validation, editing software metadata

        level1_artifacts = []
        level2_weak_signals = []
        level3_moderate_signals = []
        level4_strong_signals = []
        level5_deterministic_signals = []

        positive_checks = []
        cautions = []
        detected_anomalies = []
        critical_evidence = []

        # Track active families by level to prevent intra-family double counting
        family_level4_active = set()
        family_level5_active = set()

        # Detector detailed records
        detector_evidence = {}

        # -------------------------------------------------------------
        # (a) OCR & Content Integrity Family (Field Validation & Checksums)
        # -------------------------------------------------------------
        ocr_tokens = ocr_result.get("tokens", [])
        ocr_token_count = len(ocr_tokens)
        avg_ocr_conf = round(sum(t.get("confidence", 0.0) for t in ocr_tokens) / max(1, ocr_token_count), 4) if ocr_tokens else 0.0

        if ocr_token_count > 0:
            positive_checks.append(f"OCR successfully extracted {ocr_token_count} text tokens (avg confidence: {avg_ocr_conf:.2f})")
        else:
            cautions.append("OCR extraction returned zero text tokens; document may be low-contrast or blurred")

        failed_fields = [f for f in nlp_result.get("fields", []) if f.get("status") == "FAIL"]
        uncertain_fields = [f for f in nlp_result.get("fields", []) if f.get("status") == "UNCERTAIN" or f.get("evidence_level") == "WEAK"]
        passed_fields = [f for f in nlp_result.get("fields", []) if f.get("status") == "PASS"]

        deterministic_fails = [f for f in failed_fields if f.get("is_deterministic")]

        if deterministic_fails:
            reasons = [f"{f['field']}: {f['details']}" for f in deterministic_fails]
            sig_text = f"Deterministic checksum/format failure: {'; '.join(reasons)}"
            level5_deterministic_signals.append(sig_text)
            critical_evidence.append(sig_text)
            family_level5_active.add("content_integrity")
            field_status = "STRONG"
            field_level_name = "LEVEL 5: DETERMINISTIC EVIDENCE"
            field_expl = sig_text
        elif failed_fields:
            reasons = [f"{f['field']}: {f['details']}" for f in failed_fields]
            sig_text = f"Field validation mismatch: {'; '.join(reasons)}"
            if score_nlp <= 50.0:
                level4_strong_signals.append(sig_text)
                critical_evidence.append(sig_text)
                family_level4_active.add("content_integrity")
                field_status = "STRONG"
                field_level_name = "LEVEL 4: STRONG FORENSIC EVIDENCE"
            else:
                level3_moderate_signals.append(sig_text)
                detected_anomalies.append(sig_text)
                field_status = "MODERATE"
                field_level_name = "LEVEL 3: MODERATE FORENSIC ANOMALY"
            field_expl = sig_text
        elif uncertain_fields:
            reasons = [f"{f['field']}: {f.get('details', 'Low OCR clarity')}" for f in uncertain_fields]
            caut_text = f"Field validation inconclusive due to low OCR clarity ({'; '.join(reasons)})"
            cautions.append(caut_text)
            level2_weak_signals.append(caut_text)
            field_status = "WEAK"
            field_level_name = "LEVEL 2: WEAK FORENSIC ANOMALY"
            field_expl = caut_text
        else:
            positive_checks.append("All extracted identity fields pass algorithmic format, regex, and checksum checks")
            field_status = "CLEAN"
            field_level_name = "LEVEL 1: NORMAL IMAGE ARTIFACT"
            field_expl = f"Verified {len(passed_fields)} identity credential fields with no checksum anomalies."

        detector_evidence["field_validation"] = {
            "status": field_status,
            "confidence": round(avg_ocr_conf if ocr_token_count > 0 else 0.90, 4),
            "evidence_level": field_level_name,
            "explanation": field_expl
        }

        # -------------------------------------------------------------
        # (b) Compression Family: Error Level Analysis (ELA) & Condition
        # -------------------------------------------------------------
        ela_boxes = ela_result.get("flagged_boxes", [])
        has_cut_seam = any(
            "cut seam" in b.get("reason", "").lower() or 
            "boundary seam" in b.get("label", "").lower()
            for b in ela_boxes
        )

        if has_cut_seam:
            # Cut seam around photo/portrait is a structural splice
            seam_text = "Portrait boundary cut seam identified (photo-swap indicator)"
            level4_strong_signals.append(seam_text)
            critical_evidence.append(seam_text)
            family_level4_active.add("structural_visual")
            ela_status = "STRONG"
            ela_level_name = "LEVEL 4: STRONG FORENSIC EVIDENCE"
            ela_expl = seam_text
        elif ela_result.get("is_suspicious") and len(ela_boxes) >= 1:
            high_contrast = any(b.get("score", 0.0) >= 0.88 for b in ela_boxes)
            if high_contrast:
                anom_text = f"High-contrast localized compression anomaly hotspot ({len(ela_boxes)} region(s))"
                level3_moderate_signals.append(anom_text)
                detected_anomalies.append(anom_text)
                family_level4_active.add("compression_quality")
                ela_status = "MODERATE"
                ela_level_name = "LEVEL 3: MODERATE FORENSIC ANOMALY"
                ela_expl = anom_text
            else:
                anom_text = f"Mild localized compression variance ({len(ela_boxes)} region(s))"
                level2_weak_signals.append(anom_text)
                cautions.append(anom_text)
                ela_status = "WEAK"
                ela_level_name = "LEVEL 2: WEAK FORENSIC ANOMALY"
                ela_expl = anom_text
        else:
            positive_checks.append("Error Level Analysis shows uniform compression with no localized tampering hotspots")
            ela_status = "CLEAN"
            ela_level_name = "LEVEL 1: NORMAL IMAGE ARTIFACT"
            ela_expl = "Uniform error level distribution across document surface; no recompression hotspots."

        detector_evidence["ela"] = {
            "status": ela_status,
            "confidence": round(score_ela / 100.0, 4),
            "evidence_level": ela_level_name,
            "explanation": ela_expl
        }

        # -------------------------------------------------------------
        # (c) Structural Family: Copy-Move / Cloned Region Forensics
        # -------------------------------------------------------------
        if copy_move_result.get("copy_move_detected"):
            cm_boxes = copy_move_result.get("flagged_boxes", [])
            # Gate copy-move signals by condition: on heavily compressed/uniform images,
            # copy-move matches on background patterns are unreliable
            is_cm_condition_degraded = (
                condition.get("is_heavily_compressed", False) or
                condition.get("condition_profile") == "SEVERELY_DEGRADED" or
                condition.get("is_blurry", False)
            )
            if is_cm_condition_degraded:
                # Downgrade to WEAK — copy-move on degraded images is unreliable as sole evidence
                cm_text = f"Copy-move match on degraded/compressed image ({len(cm_boxes)} regions) — insufficient for independent verdict"
                level2_weak_signals.append(cm_text)
                cautions.append(cm_text)
                cm_status = "WEAK"
                cm_level_name = "LEVEL 2: WEAK FORENSIC ANOMALY"
                cm_expl = cm_text
            else:
                cm_text = f"Verified duplicated image region / cloned patch ({len(cm_boxes)} matched regions with RANSAC verification)"
                level4_strong_signals.append(cm_text)
                critical_evidence.append(cm_text)
                family_level4_active.add("structural_visual")
                cm_status = "STRONG"
                cm_level_name = "LEVEL 4: STRONG FORENSIC EVIDENCE"
                cm_expl = cm_text
        else:
            positive_checks.append("No cloned or duplicated image regions detected outside standard borders")
            cm_status = "CLEAN"
            cm_level_name = "LEVEL 1: NORMAL IMAGE ARTIFACT"
            cm_expl = "No copy-move forgery or repeated graphical cloning detected."

        detector_evidence["copy_move"] = {
            "status": cm_status,
            "confidence": round(score_cm / 100.0, 4),
            "evidence_level": cm_level_name,
            "explanation": cm_expl
        }

        # -------------------------------------------------------------
        # (d) Structural Family: Typography / Font Alignment Forensics
        # -------------------------------------------------------------
        font_boxes = font_result.get("flagged_boxes", [])
        if font_result.get("anomalies_detected") and len(font_boxes) >= 1:
            has_font_seam = any(
                "cut seam" in b.get("reason", "").lower() or 
                "boundary discontinuity" in b.get("reason", "").lower()
                for b in font_boxes
            )
            has_high_sharpness = any(b.get("score", 0.0) >= 0.85 for b in font_boxes)
            if has_font_seam or (has_high_sharpness and len(font_boxes) >= 2):
                font_text = f"Significant typographical rendering discontinuity on {len(font_boxes)} token(s)"
                level4_strong_signals.append(font_text)
                critical_evidence.append(font_text)
                family_level4_active.add("structural_visual")
                font_status = "STRONG"
                font_level_name = "LEVEL 4: STRONG FORENSIC EVIDENCE"
                font_expl = font_text
            else:
                font_text = f"Isolated stroke sharpness or edge gradient variance on {len(font_boxes)} token(s)"
                level2_weak_signals.append(font_text)
                cautions.append(font_text)
                font_status = "WEAK"
                font_level_name = "LEVEL 2: WEAK FORENSIC ANOMALY"
                font_expl = font_text
        else:
            positive_checks.append("Stroke sharpness, character height, and edge densities are uniformly consistent across text lines")
            font_status = "CLEAN"
            font_level_name = "LEVEL 1: NORMAL IMAGE ARTIFACT"
            font_expl = "Typography rendering and anti-aliasing are consistent across comparable fields."

        detector_evidence["typography"] = {
            "status": font_status,
            "confidence": round(score_font / 100.0, 4),
            "evidence_level": font_level_name,
            "explanation": font_expl
        }

        # -------------------------------------------------------------
        # (e) Content Integrity Family: Metadata / EXIF Inspection
        # -------------------------------------------------------------
        if metadata_result.get("is_suspicious"):
            sw = metadata_result.get("detected_software")
            if sw:
                meta_text = f"Image editing software signature identified in metadata ({sw})"
                level5_deterministic_signals.append(meta_text)
                critical_evidence.append(meta_text)
                family_level5_active.add("content_integrity")
                meta_status = "STRONG"
                meta_level_name = "LEVEL 5: DETERMINISTIC EVIDENCE"
                meta_expl = meta_text
            else:
                meta_text = "Missing standard camera EXIF metadata tags"
                level1_artifacts.append(meta_text)
                cautions.append(meta_text + " (normal for WhatsApp/web downloads)")
                meta_status = "WEAK"
                meta_level_name = "LEVEL 1: NORMAL IMAGE ARTIFACT"
                meta_expl = meta_text
        else:
            positive_checks.append("No digital manipulation software signatures found in file metadata")
            meta_status = "CLEAN"
            meta_level_name = "LEVEL 1: NORMAL IMAGE ARTIFACT"
            meta_expl = "Clean file metadata with no manipulation tools detected."

        detector_evidence["metadata"] = {
            "status": meta_status,
            "confidence": round(score_meta / 100.0, 4),
            "evidence_level": meta_level_name,
            "explanation": meta_expl
        }

        # OCR Engine explanation card
        detector_evidence["ocr"] = {
            "status": "CLEAN" if avg_ocr_conf >= 0.70 else ("WEAK" if avg_ocr_conf >= 0.40 else "MODERATE"),
            "confidence": avg_ocr_conf,
            "evidence_level": "LEVEL 1: NORMAL IMAGE ARTIFACT" if avg_ocr_conf >= 0.70 else "LEVEL 2: WEAK FORENSIC ANOMALY",
            "explanation": f"Extracted {ocr_token_count} tokens across {len(ocr_result.get('lines', []))} lines with average clarity {avg_ocr_conf:.2f}."
        }

        # Condition observations (blur, compression) -> Level 1
        if condition.get("is_blurry"):
            var_val = condition.get('laplacian_variance', 0.0)
            cautions.append(f"Image blur detected (Laplacian variance: {float(var_val):.1f})")
            level1_artifacts.append("Camera blur / defocus")
        if condition.get("is_compressed"):
            cautions.append(f"Heavy JPEG/recompression detected (estimated quality: {condition.get('compression_level', 0)})")
            level1_artifacts.append("JPEG recompression")

        # -------------------------------------------------------------
        # Categorical Diagnostic Judgment:
        # CLEAN vs ANOMALY_DETECTED vs DOCUMENT_STRONGLY_SUSPECTED_TAMPERED
        # -------------------------------------------------------------
        # Require either:
        # 1. At least 1 Level 5 deterministic finding, OR
        # 2. At least 2 independent Level 4 strong signals from DIFFERENT correlation families, OR
        # 3. 1 Level 4 strong signal + >= 2 Level 3 moderate signals across families.
        # Physical / structural splice detection (portrait cut seam or verified RANSAC cloned graphic patch)
        has_structural_splice = (
            has_cut_seam or
            (cm_status == "STRONG" and copy_move_result.get("copy_move_detected") and len(copy_move_result.get("flagged_boxes", [])) >= 2 and score_cm <= 55.0)
        )

        total_strong_families = len(family_level4_active | family_level5_active)
        has_independent_corroboration = total_strong_families >= 2

        is_strongly_suspected = (
            len(level5_deterministic_signals) >= 1 or
            has_structural_splice or
            has_independent_corroboration or
            (len(level4_strong_signals) >= 1 and len(level3_moderate_signals) >= 2)
        )

        if is_strongly_suspected:
            diagnostic_status = "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"
        elif len(level4_strong_signals) >= 1 or len(level3_moderate_signals) >= 1 or len(level2_weak_signals) >= 1:
            diagnostic_status = "ANOMALY_DETECTED"
        else:
            diagnostic_status = "CLEAN"

        # -------------------------------------------------------------
        # Confidence-Aware Evidence Fusion Scoring:
        # -------------------------------------------------------------
        rules_triggered = []
        if diagnostic_status == "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED":
            # Multi-layer independent corroboration: FLAGGED / TAMPERED (score <= 42.0)
            if has_independent_corroboration or (len(level5_deterministic_signals) >= 1 and (len(level4_strong_signals) >= 1 or len(level3_moderate_signals) >= 1)):
                fused_score = min(base_weighted_sum - 35.0, 42.0)
                rules_triggered.append("Multiple independent strong/deterministic signals across families -> FLAGGED / TAMPERED")
            elif has_structural_splice:
                fused_score = min(base_weighted_sum - 28.0, 52.0)
                rules_triggered.append("Verified structural splice / portrait seam detected -> Document strongly suspected tampered")
            elif len(level5_deterministic_signals) >= 2:
                # Multiple deterministic failures (e.g. checksum + template forgery) -> FLAGGED
                fused_score = min(base_weighted_sum - 32.0, 45.0)
                rules_triggered.append("Multiple independent deterministic integrity failures -> FLAGGED / TAMPERED")
            elif len(level5_deterministic_signals) >= 1 and (len(level2_weak_signals) >= 1 or len(level1_artifacts) >= 1):
                # Single deterministic + any weak secondary signal: moderate SUSPICIOUS toward FLAGGED
                fused_score = min(base_weighted_sum - 28.0, 55.0)
                rules_triggered.append("Deterministic integrity failure with secondary weak corroboration -> SUSPICIOUS (approaching FLAGGED)")
            else:
                # Single deterministic finding alone (e.g. Checksum failure): SUSPICIOUS (score <= 58.0)
                fused_score = min(base_weighted_sum - 25.0, 58.0)
                rules_triggered.append("Single deterministic integrity failure -> Document strongly suspected tampered (Score capped at SUSPICIOUS)")

        elif diagnostic_status == "ANOMALY_DETECTED":
            if len(level4_strong_signals) >= 1:
                # Single isolated strong forensic signal without independent corroboration:
                # Review needed, but avoid extreme over-penalization into false positive
                fused_score = min(max(52.0, base_weighted_sum - 22.0), 62.0)
                rules_triggered.append("Single isolated strong forensic anomaly without independent corroboration -> SUSPICIOUS / REVIEW NEEDED")
            elif len(level3_moderate_signals) >= 1:
                # Moderate anomaly (e.g. Missing primary PAN, consistent checksum failure, localized ELA)
                # Calibrated into SUSPICIOUS / REVIEW NEEDED range (50-69)
                fused_score = min(68.0, max(52.0, base_weighted_sum - 18.0))
                rules_triggered.append("Moderate forensic anomaly observed -> SUSPICIOUS / REVIEW NEEDED")
            else:
                # Only WEAK / natural artifacts (e.g. Uniform JPEG recompression, missing EXIF, minor OCR noise):
                # Must NOT penalize document into SUSPICIOUS! Retains AUTHENTIC (>= 80.0).
                fused_score = max(82.0, base_weighted_sum - 4.0)
                rules_triggered.append("Natural image artifacts observed; insufficient independent corroboration for tampering -> Retains AUTHENTIC")

        else: # CLEAN
            fused_score = max(88.0, min(100.0, base_weighted_sum))
            rules_triggered.append("All forensic layers clean and consistent -> AUTHENTIC")

        # Insufficient Evidence Policy: Severely degraded thumbnails cannot be certified authentic autonomously
        if condition.get("condition_profile") == "SEVERELY_DEGRADED" or condition.get("resolution", {}).get("megapixels", 1.0) < 0.08:
            if fused_score > 68.0:
                fused_score = 65.0
                sig_text = "Severe resolution limitation (< 0.08 MP): fine forensic features (micro-print, Guilloche lattice, ELA residuals) cannot be certified authentic autonomously. Mandatory human review required."
                cautions.append(sig_text)
                rules_triggered.append("Resolution insufficient for autonomous certification -> Routed to REVIEW NEEDED")
                if diagnostic_status == "CLEAN":
                    diagnostic_status = "ANOMALY_DETECTED"

        fused_score = round(max(0.0, min(100.0, fused_score)), 1)
        risk_score = round(max(0.0, min(100.0, 100.0 - fused_score)), 1)

        # Categorical Verdict
        if fused_score >= 80.0:
            verdict = "AUTHENTIC"
            verdict_color = "emerald"
            verdict_description = "Document appears authentic with uniform compression, valid checksums, consistent typography, and authentic layout structures."
            if diagnostic_status == "ANOMALY_DETECTED":
                verdict_description += " (Minor isolated natural artifacts noted; no significant tampering indicators detected)."
        elif fused_score >= 50.0:
            verdict = "SUSPICIOUS"
            verdict_color = "amber"
            verdict_description = "Document exhibits localized anomalies or format discrepancies. Secondary manual review recommended."
        else:
            verdict = "FLAGGED / TAMPERED"
            verdict_color = "rose"
            verdict_description = "High-confidence detection of digital forgery, text splicing, photo swap, or invalid identity credentials."

        # Structured OCR extraction info
        ocr_status = "SUCCESS" if ocr_token_count > 0 else ("EMPTY" if ocr_result.get("full_text") is not None else "FAILED")
        ocr_extraction_info = {
            "status": ocr_status,
            "engine": ocr_result.get("ocr_engine", "unknown"),
            "confidence": avg_ocr_conf,
            "token_count": ocr_token_count,
            "line_count": len(ocr_result.get("lines", []))
        }

        # Checksum top-level status
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

        # Classify evidence strength per signal (harmonized uppercase strings)
        evidence_strengths = {
            "nlp": detector_evidence["field_validation"]["status"],
            "ela": detector_evidence["ela"]["status"],
            "font": detector_evidence["typography"]["status"],
            "copy_move": detector_evidence["copy_move"]["status"],
            "metadata": detector_evidence["metadata"]["status"],
        }

        decision_threshold = {
            "authenticity_threshold": 80.0,
            "risk_threshold": 20.0,
            "suspicious_threshold": 50.0,
            "rule": "score >= 80.0 (risk <= 20.0) -> AUTHENTIC; 50.0 <= score < 80.0 -> SUSPICIOUS; score < 50.0 -> FLAGGED / TAMPERED"
        }

        fusion_contributions = {
            "nlp": {"weight": 0.30, "raw_score": round(score_nlp, 1), "weighted_score": round(score_nlp * 0.30, 2), "evidence_strength": evidence_strengths["nlp"]},
            "ela": {"weight": 0.25, "raw_score": round(score_ela, 1), "weighted_score": round(score_ela * 0.25, 2), "evidence_strength": evidence_strengths["ela"]},
            "font": {"weight": 0.20, "raw_score": round(score_font, 1), "weighted_score": round(score_font * 0.20, 2), "evidence_strength": evidence_strengths["font"]},
            "copy_move": {"weight": 0.15, "raw_score": round(score_cm, 1), "weighted_score": round(score_cm * 0.15, 2), "evidence_strength": evidence_strengths["copy_move"]},
            "metadata": {"weight": 0.10, "raw_score": round(score_meta, 1), "weighted_score": round(score_meta * 0.10, 2), "evidence_strength": evidence_strengths["metadata"]},
            "base_weighted_sum": base_weighted_sum,
            "penalties_applied": rules_triggered,
            "diagnostic_status": diagnostic_status
        }

        # Complete Forensic Decision Debug Report
        forensic_decision_debug = {
            "final_verdict": verdict,
            "final_risk_score": risk_score,
            "confidence": round(avg_ocr_conf if ocr_token_count > 0 else 0.90, 4),
            "diagnostic_status": diagnostic_status,
            "ocr_status_and_confidence": ocr_extraction_info,
            "condition_assessment": condition,
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
                "evidence_strength": evidence_strengths["ela"],
                "hotspot_count": len(ela_boxes),
                "is_suspicious": ela_result.get("is_suspicious", False)
            },
            "copy_move_result_and_score": {
                "score": round(score_cm, 1),
                "evidence_strength": evidence_strengths["copy_move"],
                "detected": copy_move_result.get("copy_move_detected", False)
            },
            "typography_rendering_result_and_score": {
                "score": round(score_font, 1),
                "evidence_strength": evidence_strengths["font"],
                "anomalies_detected": font_result.get("anomalies_detected", False),
                "flagged_count": len(font_boxes)
            },
            "metadata_result": {
                "score": round(score_meta, 1),
                "evidence_strength": evidence_strengths["metadata"],
                "software": metadata_result.get("detected_software"),
                "is_suspicious": metadata_result.get("is_suspicious", False)
            },
            "detector_contributions": fusion_contributions,
            "triggered_signals": level5_deterministic_signals + level4_strong_signals + level3_moderate_signals + level2_weak_signals + level1_artifacts,
            "rules_triggered": rules_triggered,
            "decision_threshold": decision_threshold,
            "cautions": cautions,
            "weak_or_uncertain_signals": cautions,
            "is_anomaly_detected": diagnostic_status == "ANOMALY_DETECTED",
            "is_strongly_suspected_tampered": diagnostic_status == "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED"
        }

        # Human Review Requirement Flag
        human_review_required = (
            verdict != "AUTHENTIC" or
            diagnostic_status == "DOCUMENT_STRONGLY_SUSPECTED_TAMPERED" or
            len(uncertain_fields) > 0 or
            condition.get("is_blurry", False)
        )

        # Build Why This Verdict structure
        why_this_verdict = {
            "summary": verdict_description,
            "positive_checks": positive_checks,
            "cautions": cautions,
            "detected_anomalies": detected_anomalies,
            "critical_evidence": critical_evidence,
            "human_review_required": human_review_required,
            "detector_explanations": detector_evidence
        }

        return {
            "authenticity_score": fused_score,
            "risk_score": risk_score,
            "verdict": verdict,
            "verdict_color": verdict_color,
            "diagnostic_status": diagnostic_status,
            "decision_threshold": decision_threshold,
            "why_this_verdict": why_this_verdict,
            "ocr_extraction": ocr_extraction_info,
            "condition_assessment": condition,
            "fusion_contributions": fusion_contributions,
            "forensic_decision_debug": forensic_decision_debug,
            "summary_explanation": verdict_description,
            "critical_triggers": level5_deterministic_signals + level4_strong_signals + level3_moderate_signals,
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
                    if reg.get("score", 0) > existing.get("score", 0):
                        existing["reason"] += f" | {reg['reason']}"
                        existing["label"] = f"{existing['label']} + {reg['label']}"
                    break

            if not is_dup:
                deduped.append(reg)

        return deduped
