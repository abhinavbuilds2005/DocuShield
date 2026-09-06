"""
DocuShield AI — Real-World Batch Benchmark Runner
Automated batch evaluation for genuine and forged identity documents across arbitrary directory trees.
Enforces strict production inference pipeline: real EasyOCR, orientation analysis, condition assessment,
ELA, typography analysis, copy-move detection, EXIF/metadata inspection, NLP/field validation, and multimodal fusion.

Zero sidecar reading. Zero hard-coded predictions. Zero filename bias.
"""

import os
import sys
import time
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from backend.fusion import DocumentScreeningPipeline

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

DEFAULT_GENUINE_DIR = r"D:\original test data"
DEFAULT_FAKE_DIR = r"D:\fake data"
DEFAULT_OUTPUT_DIR = os.path.join("reports", "batch_test")


def discover_images(directory: str) -> List[str]:
    """Recursively finds all supported image files in a directory."""
    if not os.path.exists(directory):
        print(f"[WARN] Directory not found: {directory}")
        return []
    
    discovered = []
    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                discovered.append(os.path.join(root, f))
    
    return sorted(discovered)


def categorize_failure(row: Dict[str, Any], result: Dict[str, Any]) -> str:
    """Determines the forensic root-cause failure category."""
    expected = row["expected_label"]
    pred_verdict = row["final_verdict"]
    
    # Check if there is a failure at all
    if expected == "AUTHENTIC" and pred_verdict == "AUTHENTIC":
        return "None (Correct)"
    if expected == "TAMPERED" and pred_verdict in ["SUSPICIOUS", "FLAGGED / TAMPERED"]:
        return "None (Correct)"

    condition = result.get("condition_assessment", {})
    triggers = result.get("critical_triggers", [])
    signals = result.get("signals", {})
    ocr_res = result.get("ocr_extraction", {})
    detector_exps = result.get("why_this_verdict", {}).get("detector_explanations", {})

    # 1. OCR clarity failure
    avg_conf = ocr_res.get("average_confidence", 1.0)
    token_count = len(ocr_res.get("tokens", []))
    if token_count < 3 or avg_conf < 0.45:
        return "OCR failure / Low resolution"

    # 2. Condition-based triggers
    blur_level = condition.get("blur_level", "SHARP")
    if blur_level == "BLURRED":
        return "Blur issue"

    orientation_deg = condition.get("estimated_orientation_degrees", 0)
    if orientation_deg != 0:
        return "Rotation issue"

    # 3. Detector-specific false positives or misses
    ela_exp = detector_exps.get("ela", {})
    if ela_exp.get("status") in ["MODERATE", "STRONG"] and expected == "AUTHENTIC":
        if "JPEG" in condition.get("compression_level", ""):
            return "Compression issue / ELA false positive"
        return "ELA false positive"

    font_exp = detector_exps.get("typography", {})
    if font_exp.get("status") in ["MODERATE", "STRONG"] and expected == "AUTHENTIC":
        return "Typography false positive"

    cm_exp = detector_exps.get("copy_move", {})
    if cm_exp.get("status") in ["MODERATE", "STRONG"] and expected == "AUTHENTIC":
        return "Copy-move false positive"

    nlp_exp = detector_exps.get("field_validation", {})
    if nlp_exp.get("status") in ["MODERATE", "STRONG"] and expected == "AUTHENTIC":
        return "Field validation issue"

    meta_exp = detector_exps.get("metadata", {})
    if meta_exp.get("status") in ["MODERATE", "STRONG"] and expected == "AUTHENTIC":
        return "Metadata issue"

    if expected == "TAMPERED" and pred_verdict == "AUTHENTIC":
        return "Missed tampering / Low contrast or subtle edit"

    return "Fusion/scoring issue"


def run_batch_test(
    genuine_dir: str = DEFAULT_GENUINE_DIR,
    fake_dir: str = DEFAULT_FAKE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    real_ocr: bool = True
) -> Dict[str, Any]:
    """
    Executes the automated batch benchmark on genuine and fake document datasets.
    Zero synthetic sidecars. Real EasyOCR and multi-modal forensic pipelines.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 64)
    print(" DOCUSHIELD AI — AUTOMATED REAL-WORLD BATCH BENCHMARK")
    print("=" * 64)
    print(f"Genuine Dataset:  {genuine_dir}")
    print(f"Tampered Dataset: {fake_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Inference Mode:   {'REAL EASYOCR (Strict Production)' if real_ocr else 'STANDARD'}")
    print("=" * 64 + "\n")

    genuine_files = discover_images(genuine_dir)
    fake_files = discover_images(fake_dir)

    print(f"[DATASET] Discovered {len(genuine_files)} genuine documents.")
    print(f"[DATASET] Discovered {len(fake_files)} tampered documents.")
    total_images = len(genuine_files) + len(fake_files)
    print(f"[DATASET] Total documents to process: {total_images}\n")

    if total_images == 0:
        print("[ERROR] No image files found in the specified directories. Exiting.")
        return {}

    # Initialize the real production pipeline
    pipeline = DocumentScreeningPipeline()

    all_records = []
    failures = []

    # Distribution and metrics counters
    tp = 0  # Tampered correctly identified as SUSPICIOUS or FLAGGED
    tn = 0  # Genuine correctly identified as AUTHENTIC
    fp = 0  # Genuine incorrectly identified as SUSPICIOUS or FLAGGED
    fn = 0  # Tampered incorrectly identified as AUTHENTIC

    verdict_counts = {
        "AUTHENTIC": 0,
        "SUSPICIOUS": 0,
        "FLAGGED / TAMPERED": 0,
        "UNKNOWN": 0
    }

    detector_status_counts = {
        "ela": {"CLEAN": 0, "WEAK": 0, "MODERATE": 0, "STRONG": 0},
        "typography": {"CLEAN": 0, "WEAK": 0, "MODERATE": 0, "STRONG": 0},
        "copy_move": {"CLEAN": 0, "WEAK": 0, "MODERATE": 0, "STRONG": 0},
        "metadata": {"CLEAN": 0, "WEAK": 0, "MODERATE": 0, "STRONG": 0},
        "field_validation": {"CLEAN": 0, "WEAK": 0, "MODERATE": 0, "STRONG": 0},
        "ocr": {"CLEAN": 0, "WEAK": 0, "MODERATE": 0, "STRONG": 0},
    }

    test_plan = []
    for p in genuine_files:
        test_plan.append((p, "AUTHENTIC"))
    for p in fake_files:
        test_plan.append((p, "TAMPERED"))

    overall_start_time = time.time()

    for idx, (img_path, expected_label) in enumerate(test_plan, 1):
        filename = os.path.basename(img_path)
        print(f"[{idx:02d}/{total_images:02d}] Processing: {filename:38s} | Expected: {expected_label:9s}", end="", flush=True)

        t_start = time.time()
        try:
            # benchmark_mode=False strictly guarantees real OCR and production flow
            result = pipeline.screen_document(img_path, benchmark_mode=False)
            exec_time = round(time.time() - t_start, 2)
        except Exception as err:
            exec_time = round(time.time() - t_start, 2)
            print(f" -> ERROR ({err})")
            continue

        verdict = result.get("verdict", "UNKNOWN")
        auth_score = result.get("authenticity_score", 0.0)
        risk_score = result.get("risk_score", 0.0)
        diag_status = result.get("diagnostic_status", "CLEAN")
        why = result.get("why_this_verdict", {})
        det_exps = why.get("detector_explanations", {})
        triggers = result.get("critical_triggers", [])
        cautions = why.get("cautions", [])
        positives = why.get("positive_checks", [])
        condition = result.get("condition_assessment", {})

        # Tally verdict
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        # Determine predicted binary label
        # AUTHENTIC -> predicted AUTHENTIC
        # SUSPICIOUS / FLAGGED -> predicted TAMPERED
        if verdict == "AUTHENTIC":
            pred_label = "AUTHENTIC"
        else:
            pred_label = "TAMPERED"

        # Classification outcome
        is_correct = (pred_label == expected_label)
        if expected_label == "TAMPERED":
            if pred_label == "TAMPERED":
                tp += 1
                status_tag = "CORRECT_TAMPERED"
            else:
                fn += 1
                status_tag = "FALSE_NEGATIVE "
        else:
            if pred_label == "AUTHENTIC":
                tn += 1
                status_tag = "CORRECT_GENUINE "
            else:
                fp += 1
                status_tag = "FALSE_POSITIVE "

        print(f" -> {verdict:18s} (Auth: {auth_score:4.1f}%, Risk: {risk_score:4.1f}%, Time: {exec_time:4.2f}s) [{status_tag.strip()}]")

        # Tally detector status levels
        for d_key in detector_status_counts.keys():
            exp = det_exps.get(d_key, {})
            lvl = exp.get("status", "CLEAN")
            if lvl in detector_status_counts[d_key]:
                detector_status_counts[d_key][lvl] += 1

        # Extract detector values
        ela_exp = det_exps.get("ela", {})
        font_exp = det_exps.get("typography", {})
        cm_exp = det_exps.get("copy_move", {})
        meta_exp = det_exps.get("metadata", {})
        nlp_exp = det_exps.get("field_validation", {})
        ocr_exp = det_exps.get("ocr", {})

        record = {
            "filename": filename,
            "full_path": os.path.abspath(img_path),
            "expected_label": expected_label,
            "predicted_label": pred_label,
            "final_verdict": verdict,
            "authenticity_score": auth_score,
            "risk_score": risk_score,
            "diagnostic_status": diag_status,
            "document_type": result.get("signals", {}).get("nlp_validation", {}).get("document_type", "unknown"),
            "processing_time_sec": exec_time,
            "ocr_confidence": round(float(ocr_exp.get("confidence", 0.0)), 4),
            "detected_anomaly_count": len(triggers) + len(result.get("flagged_regions", [])),
            "status_tag": status_tag.strip(),
            # Detectors
            "ela_status": ela_exp.get("status", "CLEAN"),
            "ela_confidence": ela_exp.get("confidence", 1.0),
            "ela_explanation": ela_exp.get("explanation", ""),
            "typography_status": font_exp.get("status", "CLEAN"),
            "typography_confidence": font_exp.get("confidence", 1.0),
            "typography_explanation": font_exp.get("explanation", ""),
            "copy_move_status": cm_exp.get("status", "CLEAN"),
            "copy_move_confidence": cm_exp.get("confidence", 1.0),
            "copy_move_explanation": cm_exp.get("explanation", ""),
            "metadata_status": meta_exp.get("status", "CLEAN"),
            "metadata_confidence": meta_exp.get("confidence", 1.0),
            "metadata_explanation": meta_exp.get("explanation", ""),
            "nlp_field_status": nlp_exp.get("status", "CLEAN"),
            "nlp_field_confidence": nlp_exp.get("confidence", 1.0),
            "nlp_field_explanation": nlp_exp.get("explanation", ""),
            "condition_quality": condition.get("quality_rating", "GOOD"),
            "condition_blur": condition.get("blur_level", "SHARP"),
            "condition_compression": condition.get("compression_level", "UNIFORM_HIGH_QUALITY"),
            # Explainability
            "why_this_verdict_summary": why.get("summary", ""),
            "positive_verifications": "; ".join(positives),
            "cautions": "; ".join(cautions),
            "critical_triggers": "; ".join(triggers)
        }
        all_records.append(record)

        if not is_correct or verdict == "SUSPICIOUS":
            fail_category = categorize_failure(record, result)
            
            # Find strongest detector
            detector_candidates = [
                ("ELA", ela_exp.get("status", "CLEAN"), ela_exp.get("confidence", 1.0)),
                ("Typography", font_exp.get("status", "CLEAN"), font_exp.get("confidence", 1.0)),
                ("Copy-Move", cm_exp.get("status", "CLEAN"), cm_exp.get("confidence", 1.0)),
                ("Metadata", meta_exp.get("status", "CLEAN"), meta_exp.get("confidence", 1.0)),
                ("NLP/Checksum", nlp_exp.get("status", "CLEAN"), nlp_exp.get("confidence", 1.0)),
            ]
            # Prioritize STRONG > MODERATE > WEAK
            rank = {"STRONG": 3, "MODERATE": 2, "WEAK": 1, "CLEAN": 0}
            strongest_det = max(detector_candidates, key=lambda x: rank.get(x[1], 0))

            failure_entry = {
                "image_name": filename,
                "expected_result": expected_label,
                "predicted_result": pred_label,
                "final_verdict": verdict,
                "authenticity_score": auth_score,
                "risk_score": risk_score,
                "strongest_detector": f"{strongest_det[0]} ({strongest_det[1]})",
                "detector_confidence": strongest_det[2],
                "failure_category": fail_category,
                "why_system_decided": why.get("summary", ""),
                "critical_triggers": "; ".join(triggers),
                "possible_reason": f"Category: {fail_category}. Cautions: {'; '.join(cautions) or 'None'}"
            }
            failures.append(failure_entry)

    total_evaluated = tp + tn + fp + fn
    total_time = round(time.time() - overall_start_time, 2)

    # Calculate standard metrics
    accuracy = round((tp + tn) / max(1, total_evaluated) * 100.0, 2)
    precision = round(tp / max(1, tp + fp) * 100.0, 2)
    recall = round(tp / max(1, tp + fn) * 100.0, 2)
    f1 = round(2 * (precision * recall) / max(1e-5, precision + recall), 2) if (precision + recall) > 0 else 0.0

    fpr = round(fp / max(1, fp + tn) * 100.0, 2)
    fnr = round(fn / max(1, fn + tp) * 100.0, 2)

    genuine_acc = round(tn / max(1, len(genuine_files)) * 100.0, 2)
    tampered_acc = round(tp / max(1, len(fake_files)) * 100.0, 2)

    # Tri-State / Review-aware distribution
    genuine_authentic = sum(1 for r in all_records if r["expected_label"] == "AUTHENTIC" and r["final_verdict"] == "AUTHENTIC")
    genuine_suspicious = sum(1 for r in all_records if r["expected_label"] == "AUTHENTIC" and r["final_verdict"] == "SUSPICIOUS")
    genuine_flagged = sum(1 for r in all_records if r["expected_label"] == "AUTHENTIC" and r["final_verdict"] == "FLAGGED / TAMPERED")

    tampered_authentic = sum(1 for r in all_records if r["expected_label"] == "TAMPERED" and r["final_verdict"] == "AUTHENTIC")
    tampered_suspicious = sum(1 for r in all_records if r["expected_label"] == "TAMPERED" and r["final_verdict"] == "SUSPICIOUS")
    tampered_flagged = sum(1 for r in all_records if r["expected_label"] == "TAMPERED" and r["final_verdict"] == "FLAGGED / TAMPERED")

    # Write batch_results.csv
    csv_path = os.path.join(output_dir, "batch_results.csv")
    if all_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)

    # Write failures.csv
    failures_csv_path = os.path.join(output_dir, "failures.csv")
    if failures:
        with open(failures_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
    else:
        with open(failures_csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("image_name,expected_result,predicted_result,final_verdict,authenticity_score,risk_score,strongest_detector,detector_confidence,failure_category,why_system_decided,critical_triggers,possible_reason\n")

    # Generate benchmark summary text
    summary_lines = [
        "=" * 64,
        "DOCUSHIELD AI — REAL WORLD BATCH BENCHMARK",
        "=" * 64,
        f"Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Genuine folder:   {genuine_dir}",
        f"Tampered folder:  {fake_dir}",
        f"Total Processing: {total_time}s across {total_evaluated} documents",
        "",
        f"Total Documents:  {total_evaluated}",
        f"Genuine:          {len(genuine_files)}",
        f"Tampered:         {len(fake_files)}",
        "",
        f"TP: {tp}",
        f"TN: {tn}",
        f"FP: {fp}",
        f"FN: {fn}",
        "",
        f"Accuracy:  {accuracy}% on evaluated batch dataset of {total_evaluated} documents",
        f"Precision: {precision}%",
        f"Recall:    {recall}%",
        f"F1 Score:  {f1}%",
        "",
        f"False Positive Rate: {fpr}%",
        f"False Negative Rate: {fnr}%",
        "",
        f"Genuine Detection Accuracy:  {genuine_acc}%",
        f"Tampered Detection Accuracy: {tampered_acc}%",
        "",
        "-" * 64,
        "CONFUSION MATRIX",
        "-" * 64,
        f"                     Predicted AUTHENTIC    Predicted TAMPERED",
        f"Actual AUTHENTIC:           {tn:4d} (TN)            {fp:4d} (FP)",
        f"Actual TAMPERED:            {fn:4d} (FN)            {tp:4d} (TP)",
        "",
        "-" * 64,
        "TRI-STATE VERDICT BREAKDOWN (Review-Aware)",
        "-" * 64,
        f"Genuine Documents ({len(genuine_files)} total):",
        f"  - AUTHENTIC (Directly Cleared):             {genuine_authentic:2d} ({genuine_authentic/max(1,len(genuine_files))*100:.1f}%)",
        f"  - SUSPICIOUS (Routed to Human Review):      {genuine_suspicious:2d} ({genuine_suspicious/max(1,len(genuine_files))*100:.1f}%)",
        f"  - FLAGGED / TAMPERED (Directly Rejected):   {genuine_flagged:2d} ({genuine_flagged/max(1,len(genuine_files))*100:.1f}%)",
        "",
        f"Tampered Documents ({len(fake_files)} total):",
        f"  - FLAGGED / TAMPERED (Directly Caught):     {tampered_flagged:2d} ({tampered_flagged/max(1,len(fake_files))*100:.1f}%)",
        f"  - SUSPICIOUS (Caught / Review Mandated):     {tampered_suspicious:2d} ({tampered_suspicious/max(1,len(fake_files))*100:.1f}%)",
        f"  - AUTHENTIC (Missed / False Negative):      {tampered_authentic:2d} ({tampered_authentic/max(1,len(fake_files))*100:.1f}%)",
        "",
        "-" * 64,
        "DETECTOR STATUS DISTRIBUTION",
        "-" * 64
    ]

    for det_name, levels in detector_status_counts.items():
        summary_lines.append(f"{det_name.upper()}:")
        for lvl, cnt in levels.items():
            summary_lines.append(f"  {lvl:8s}: {cnt:2d}")

    summary_lines.extend([
        "=" * 64,
        f"FAILED / UNCERTAIN CASES: {len(failures)}",
        "=" * 64
    ])

    if failures:
        for f_idx, fail in enumerate(failures, 1):
            summary_lines.append(f"{f_idx}. {fail['image_name']} -> Expected: {fail['expected_result']}, Verdict: {fail['final_verdict']} (Score: {fail['authenticity_score']}%)")
            summary_lines.append(f"   Category: {fail['failure_category']}")
            summary_lines.append(f"   Strongest Detector: {fail['strongest_detector']}")
            summary_lines.append(f"   Deciding Rationale: {fail['why_system_decided']}")
            if fail['critical_triggers']:
                summary_lines.append(f"   Triggers: {fail['critical_triggers']}")
            summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # Save summary text file
    summary_txt_path = os.path.join(output_dir, "benchmark_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Save structured batch_results.json
    results_json_path = os.path.join(output_dir, "batch_results.json")
    batch_json_data = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "genuine_dir": genuine_dir,
        "fake_dir": fake_dir,
        "total_documents": total_evaluated,
        "genuine_count": len(genuine_files),
        "tampered_count": len(fake_files),
        "metrics": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "genuine_accuracy": genuine_acc,
            "tampered_accuracy": tampered_acc
        },
        "verdict_distribution": verdict_counts,
        "detector_status_distribution": detector_status_counts,
        "failure_count": len(failures),
        "failures": failures,
        "all_records": all_records
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(batch_json_data, f, indent=2)

    print(f"\n[OUTPUT] batch_results.csv     -> {csv_path}")
    print(f"[OUTPUT] batch_results.json    -> {results_json_path}")
    print(f"[OUTPUT] benchmark_summary.txt -> {summary_txt_path}")
    print(f"[OUTPUT] failures.csv          -> {failures_csv_path}\n")

    return batch_json_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocuShield AI Real-World Automated Batch Benchmark")
    parser.add_argument("--genuine-dir", type=str, default=DEFAULT_GENUINE_DIR, help="Path to genuine images directory")
    parser.add_argument("--fake-dir", type=str, default=DEFAULT_FAKE_DIR, help="Path to fake/tampered images directory")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to save report files")
    parser.add_argument("--real-ocr", action="store_true", default=True, help="Force real EasyOCR production pipeline")
    
    args = parser.parse_args()
    run_batch_test(
        genuine_dir=args.genuine_dir,
        fake_dir=args.fake_dir,
        output_dir=args.output_dir,
        real_ocr=args.real_ocr
    )
