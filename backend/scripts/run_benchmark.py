"""
Benchmark Evaluation Script
Screens all 20 documents in backend/data/ against ground_truth.json
Computes:
- Confusion Matrix (TP, FP, TN, FN)
- Detection Accuracy, Precision, Recall, F1 Score
- Detailed breakdown per document & tampering attack

Supports two modes:
- Sidecar OCR (default): Uses pre-computed .ocr.json for controlled testing
- Real OCR (use_real_ocr=True): Uses actual OCR engine, no sidecars
"""

import os
import json
import time
from backend.fusion import DocumentScreeningPipeline

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GT_PATH = os.path.join(DATA_DIR, "ground_truth.json")

# Reuse a single pipeline instance for benchmark runs
_benchmark_pipeline = None


def _get_pipeline():
    global _benchmark_pipeline
    if _benchmark_pipeline is None:
        _benchmark_pipeline = DocumentScreeningPipeline()
    return _benchmark_pipeline


def run_benchmark(use_real_ocr: bool = False):
    """
    Runs the benchmark evaluation.
    
    Args:
        use_real_ocr: If True, uses real OCR only (benchmark_mode=False).
                     If False (default), allows sidecar OCR (benchmark_mode=True).
    """
    pipeline = _get_pipeline()
    with open(GT_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Count dataset composition
    genuine_count = sum(1 for g in ground_truth.values() if g["label"] == "GENUINE")
    tampered_count = sum(1 for g in ground_truth.values() if g["label"] == "TAMPERED")

    results = []
    tp = 0  # Tampered correctly flagged as TAMPERED or SUSPICIOUS
    fp = 0  # Genuine falsely flagged as TAMPERED
    tn = 0  # Genuine correctly identified as AUTHENTIC
    fn = 0  # Tampered missed (classified as AUTHENTIC)
    errors = []

    # benchmark_mode is the INVERSE of use_real_ocr:
    # use_real_ocr=True  → benchmark_mode=False  (forces real OCR)
    # use_real_ocr=False → benchmark_mode=True   (allows sidecar)
    benchmark_mode = not use_real_ocr

    start_time = time.time()

    for filename, gt in ground_truth.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            errors.append({"filename": filename, "error": "File not found"})
            continue

        try:
            res = pipeline.screen_document(filepath, benchmark_mode=benchmark_mode)
            score = res["authenticity_score"]
            verdict = res["verdict"]
            label = gt["label"]
            ocr_engine = res.get("ocr_engine_used", "unknown")

            # Classification logic:
            # Genuine should be AUTHENTIC
            # Tampered should be FLAGGED / TAMPERED or SUSPICIOUS
            predicted_tampered = (verdict != "AUTHENTIC")
            actual_tampered = (label == "TAMPERED")

            if actual_tampered and predicted_tampered:
                tp += 1
                status = "CORRECT_DETECTION"
            elif not actual_tampered and not predicted_tampered:
                tn += 1
                status = "CORRECT_GENUINE"
            elif not actual_tampered and predicted_tampered:
                fp += 1
                status = "FALSE_POSITIVE"
            else:
                fn += 1
                status = "FALSE_NEGATIVE"

            signals = res.get("signals", {})
            nlp_sig = signals.get("nlp_validation", {})
            ela_sig = signals.get("ela_forensics", {})
            font_sig = signals.get("font_typography", {})
            cm_sig = signals.get("copy_move", {})
            meta_sig = signals.get("metadata_forensics", {})
            ocr_ext = res.get("ocr_extraction", {})

            doc_entry = {
                # 1. filename / document ID
                "document_id": filename,
                "filename": filename,
                # 2. ground-truth label
                "ground_truth_label": label,
                # 3. final prediction
                "final_prediction": verdict,
                # 4. final risk/fraud score
                "final_risk_fraud_score": res.get("risk_score", round(100.0 - score, 1)),
                "authenticity_score": score,
                # 5. OCR extraction status
                "ocr_extraction_status": ocr_ext.get("status", "SUCCESS" if ocr_ext.get("token_count", 0) > 0 else "EMPTY"),
                # 6. OCR confidence
                "ocr_confidence": ocr_ext.get("confidence", 0.0),
                # 7. field validation/checksum results
                "field_validation_checksum_results": {
                    "document_type": nlp_sig.get("document_type"),
                    "score": nlp_sig.get("score"),
                    "evidence_strength": nlp_sig.get("evidence_strength"),
                    "fields": nlp_sig.get("field_checks", []),
                    "reasons": nlp_sig.get("reasons", [])
                },
                # 8. ELA score/evidence
                "ela_score_evidence": {
                    "score": ela_sig.get("score"),
                    "evidence_strength": ela_sig.get("evidence_strength"),
                    "hotspot_count": ela_sig.get("hotspot_count", 0),
                    "metrics": ela_sig.get("metrics", {})
                },
                # 9. copy-move score/evidence
                "copy_move_score_evidence": {
                    "score": cm_sig.get("score"),
                    "evidence_strength": cm_sig.get("evidence_strength"),
                    "detected": cm_sig.get("detected", False),
                    "details": cm_sig.get("details", "")
                },
                # 10. typography/rendering anomaly score/evidence
                "typography_rendering_anomaly_score_evidence": {
                    "score": font_sig.get("score"),
                    "evidence_strength": font_sig.get("evidence_strength"),
                    "details": font_sig.get("details", "")
                },
                # 11. metadata anomaly score/evidence
                "metadata_anomaly_score_evidence": {
                    "score": meta_sig.get("score"),
                    "evidence_strength": meta_sig.get("evidence_strength"),
                    "detected_software": meta_sig.get("detected_software"),
                    "reasons": meta_sig.get("reasons", [])
                },
                # 12. fusion contribution from each signal
                "fusion_contributions": res.get("fusion_contributions", {}),
                # 13. final decision threshold
                "final_decision_threshold": res.get("decision_threshold", {
                    "authenticity_threshold": 80.0,
                    "risk_threshold": 20.0,
                    "suspicious_threshold": 50.0
                }),
                # Classification status
                "classification_status": status,
                "tampering_types": gt.get("tampering_types", []),
                "flagged_regions_count": len(res.get("flagged_regions", [])),
                "critical_triggers": res.get("critical_triggers", []),
                "ocr_engine": ocr_engine
            }

            results.append(doc_entry)

            print(f"[{status:17}] {filename} -> {verdict} (Risk: {doc_entry['final_risk_fraud_score']:.1f}, Auth: {score:.1f}, GT: {label})")
            if status in ("FALSE_NEGATIVE", "FALSE_POSITIVE"):
                print(f"   Signals: NLP={nlp_sig.get('score')} | ELA={ela_sig.get('score')} | Font={font_sig.get('score')} | CM={cm_sig.get('score')} | Meta={meta_sig.get('score')}")
                if res.get("critical_triggers"):
                    print(f"   Triggers: {res.get('critical_triggers')}")

        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})
            if gt["label"] == "TAMPERED":
                fn += 1

    elapsed = round(time.time() - start_time, 2)
    total = len(results)
    accuracy = round(((tp + tn) / total) * 100, 2) if total > 0 else 0
    precision = round((tp / (tp + fp)) * 100, 2) if (tp + fp) > 0 else 0
    recall = round((tp / (tp + fn)) * 100, 2) if (tp + fn) > 0 else 0
    f1 = round(2 * (precision * recall) / (precision + recall + 1e-5), 2) if (precision + recall) > 0 else 0

    benchmark_mode_label = "real_ocr" if use_real_ocr else "sidecar_ocr"

    benchmark_summary = {
        "total_documents": total,
        "elapsed_seconds": elapsed,
        "benchmark_mode": benchmark_mode_label,
        "dataset_composition": {
            "genuine": genuine_count,
            "tampered": tampered_count,
            "total": genuine_count + tampered_count
        },
        "qualification": (
            f"Results from {total}-document synthetic benchmark "
            f"({'with real OCR engine' if use_real_ocr else 'with sidecar OCR for controlled testing'}). "
            f"This is NOT a claim of real-world accuracy."
        ),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        },
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        },
        "details": results,
        "errors": errors
    }

    # Save complete structured results to file
    out_file = os.path.join(DATA_DIR, f"benchmark_results_{benchmark_mode_label}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\n{'='*52}")
    print(f" BENCHMARK REPORT ({benchmark_mode_label})")
    print(f"{'='*52}")
    print(f"Total Documents Tested: {total} in {elapsed}s")
    print(f"Dataset: {genuine_count} genuine + {tampered_count} tampered")
    print(f"Accuracy:  {accuracy}%")
    print(f"Precision: {precision}%")
    print(f"Recall:    {recall}%")
    print(f"F1-Score:  {f1}%")
    print(f"Confusion Matrix: TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    if errors:
        print(f"Errors: {len(errors)} documents failed to screen")
    print(f"Full structured breakdown saved to: {out_file}")
    print(f"{'='*52}\n")

    return benchmark_summary


if __name__ == "__main__":
    import sys
    use_real = "--real-ocr" in sys.argv
    run_benchmark(use_real_ocr=use_real)
