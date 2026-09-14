"""
Held-Out Generalization & Robustness Benchmark Evaluator
Evaluates the full screening pipeline on 11 completely unseen documents and transformations:
- 5 Genuine variations (Original, JPEG Q=70, Resized, Screenshot/screen photo, Perspective/lighting)
- 6 Tampered vectors (Checksum alteration, Text splice, Copy-move, Font splice, Category splice, Realistic Photo-Swap with single normal JPEG re-save)

Operates strictly in REAL OCR MODE. Never reads sidecars or ground truth during screening.
"""

import os
import sys

# Ensure repository root is on sys.path for direct script invocation
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import time
import json
from typing import Dict, Any

from backend.fusion import DocumentScreeningPipeline

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROBUST_DIR = os.path.join(DATA_DIR, "robustness")
GT_PATH = os.path.join(ROBUST_DIR, "ground_truth.json")


def run_robustness_benchmark() -> Dict[str, Any]:
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Robustness ground truth not found at {GT_PATH}. Run generate_robustness_dataset.py first.")

    with open(GT_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    pipeline = DocumentScreeningPipeline()
    tp, tn, fp, fn = 0, 0, 0, 0
    detailed_results = []

    print("====================================================")
    print(" HELD-OUT GENERALIZATION & ROBUSTNESS BENCHMARK")
    print(" Mode: REAL OCR INFERENCE (No sidecars)")
    print("====================================================")

    start_time = time.time()

    for filename in sorted(ground_truth.keys()):
        gt_entry = ground_truth[filename]
        gt_label = gt_entry["label"]
        file_path = os.path.join(ROBUST_DIR, filename)

        if not os.path.exists(file_path):
            print(f"[WARN] File {filename} not found, skipping...")
            continue

        # Strictly real OCR mode: benchmark_mode=False
        try:
            res = pipeline.screen_document(file_path, benchmark_mode=False)
        except Exception as e:
            print(f"[ERROR] Screening failed for {filename}: {e}")
            continue

        pred_verdict = res.get("verdict")
        pred_score = res.get("authenticity_score", 100.0)
        risk_score = res.get("risk_score", 0.0)
        diagnostic_status = res.get("diagnostic_status", "CLEAN")
        debug_report = res.get("forensic_decision_debug", {})

        is_tampered_pred = pred_verdict in ["SUSPICIOUS", "FLAGGED / TAMPERED"]
        is_tampered_gt = gt_label == "TAMPERED"

        if is_tampered_gt and is_tampered_pred:
            tp += 1
            status_tag = "CORRECT_DETECTION"
        elif not is_tampered_gt and not is_tampered_pred:
            tn += 1
            status_tag = "CORRECT_GENUINE  "
        elif not is_tampered_gt and is_tampered_pred:
            fp += 1
            status_tag = "FALSE_POSITIVE   "
        else:
            fn += 1
            status_tag = "FALSE_NEGATIVE   "

        print(f"[{status_tag}] {filename:42s} -> {pred_verdict:18s} (Risk: {risk_score:4.1f}, Auth: {pred_score:4.1f}, Status: {diagnostic_status}, GT: {gt_label})")

        detailed_results.append({
            "filename": filename,
            "ground_truth_label": gt_label,
            "final_prediction": pred_verdict,
            "final_risk_score": risk_score,
            "authenticity_score": pred_score,
            "diagnostic_status": diagnostic_status,
            "classification_status": status_tag.strip(),
            "forensic_decision_debug": debug_report
        })

    elapsed = round(time.time() - start_time, 2)
    total = tp + tn + fp + fn
    accuracy = round((tp + tn) / max(1, total) * 100.0, 2)
    precision = round(tp / max(1, tp + fp) * 100.0, 2)
    recall = round(tp / max(1, tp + fn) * 100.0, 2)
    f1 = round(2 * (precision * recall) / max(1e-5, precision + recall), 2) if (precision + recall) > 0 else 0.0

    report = {
        "benchmark_name": "HELD_OUT_GENERALIZATION_AND_ROBUSTNESS",
        "total_documents": total,
        "elapsed_seconds": elapsed,
        "dataset_composition": {"genuine": tn + fp, "tampered": tp + fn, "total": total},
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
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "details": detailed_results
    }

    print("\n====================================================")
    print(" GENERALIZATION & ROBUSTNESS BENCHMARK REPORT")
    print("====================================================")
    print(f"Total Documents Tested: {total} in {elapsed}s")
    print(f"Accuracy:  {accuracy}%")
    print(f"Precision: {precision}%")
    print(f"Recall:    {recall}%")
    print(f"F1-Score:  {f1}%")
    print(f"Confusion Matrix: TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    out_json = os.path.join(DATA_DIR, "benchmark_results_robustness.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full results to: {out_json}")
    print("====================================================\n")
    return report


if __name__ == "__main__":
    run_robustness_benchmark()
