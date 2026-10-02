"""
Training & Evaluation Script for PAN Card Field Detection (YOLOv8n)
DocuShield AI — Phase 2 Modular Expansion

Features:
- Transfer learning from pretrained YOLOv8n
- Uses GPU (NVIDIA RTX 3050) when available with PyTorch CUDA
- Workers=0 and chunked file copy for Windows Python 3.13 memory stability
- Conservative document-safe augmentations (no horizontal/vertical flipping, no mosaic/mixup)
- Letterbox resizing (rect=True)
- Saves best model to models/pan_field_detector.pt (preserves models/aadhaar_field_detector.pt)
- Evaluates on held-out validation set and saves full metrics report to reports/pan_field_detection/
- Strictly respects PII masking and forensic disclaimers
"""

import os
import sys
import json
import time
from pathlib import Path
import torch
from ultralytics.engine.trainer import BaseTrainer

# 1. Chunked copy fix for Python 3.13 on Windows
def _safe_chunked_copy(src, dst):
    with open(src, "rb") as rf, open(dst, "wb") as wf:
        while chunk := rf.read(65536):
            wf.write(chunk)

def _direct_file_save_model(self):
    from copy import deepcopy
    self.wdir.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "epoch": self.epoch,
        "best_fitness": self.best_fitness,
        "model": deepcopy(self.ema.ema).half(),
        "updates": self.ema.updates,
        "train_args": vars(self.args),
        "train_metrics": {**self.metrics, "fitness": self.fitness},
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    torch.save(save_dict, str(self.last))
    if self.best_fitness == self.fitness:
        _safe_chunked_copy(str(self.last), str(self.best))
    return True

BaseTrainer.save_model = _direct_file_save_model

from ultralytics import YOLO

DATASET_ROOT = Path("data/processed/pan_field_detection")
DATA_YAML = DATASET_ROOT / "data.yaml"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/pan_field_detection")
PROJECT_RUNS = Path("runs/field_detection")


def train_pan_detector(epochs: int = 20, batch_size: int = 8, imgsz: int = 640):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING PAN CARD FIELD DETECTION MODEL (YOLOv8n)")
    print("=" * 70)
    print(f"Device: {device} ({device_name})")
    print(f"Dataset config: {DATA_YAML.resolve()}")
    print(f"Epochs: {epochs} (Early stopping patience: 10)")
    print(f"Batch size: {batch_size}, Image size: {imgsz}")
    print("=" * 70)

    # Transfer learning from pretrained YOLOv8n
    model = YOLO("yolov8n.pt")

    start_train_time = time.time()
    results = model.train(
        data=str(DATA_YAML.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=0,
        seed=42,
        patience=10,
        project=str(PROJECT_RUNS),
        name="pan_yolov8n",
        exist_ok=True,
        # Conservative document augmentations
        degrees=3.0,
        translate=0.04,
        scale=0.08,
        shear=1.0,
        perspective=0.0003,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        mosaic=0.0,
        mixup=0.0,
        flipud=0.0,
        fliplr=0.0,
        rect=True,          # Letterbox rectangular resizing
        verbose=True
    )
    train_duration = time.time() - start_train_time

    # Save to models/pan_field_detector.pt
    run_dir = PROJECT_RUNS / "pan_yolov8n"
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = run_dir / "weights" / "last.pt"

    dest_model_path = MODELS_DIR / "pan_field_detector.pt"
    if best_weights.exists():
        _safe_chunked_copy(str(best_weights), str(dest_model_path))
        print(f"\n[MODEL SAVED] Best PAN model saved to: {dest_model_path}")
    else:
        print("[WARNING] Best weights not found, using last.pt")
        dest_model_path = best_weights

    model_size_mb = os.path.getsize(dest_model_path) / (1024 * 1024)

    # Validation evaluation
    print("\n" + "=" * 70)
    print("EVALUATING ON HELD-OUT VALIDATION SET")
    print("=" * 70)

    eval_model = YOLO(str(dest_model_path))
    val_metrics = eval_model.val(
        data=str(DATA_YAML.resolve()),
        split="val",
        device=device,
        workers=0,
        rect=True
    )

    class_names = ['pan_number', 'name', 'father_name', 'date_of_birth']
    per_class_results = {}
    
    # Extract per-class metrics
    try:
        mp = float(val_metrics.box.mp)
        mr = float(val_metrics.box.mr)
        map50 = float(val_metrics.box.map50)
        map50_95 = float(val_metrics.box.map)
        
        # Per class arrays
        p_per_class = val_metrics.box.p.tolist() if hasattr(val_metrics.box, 'p') and val_metrics.box.p is not None else []
        r_per_class = val_metrics.box.r.tolist() if hasattr(val_metrics.box, 'r') and val_metrics.box.r is not None else []
        map50_per_class = val_metrics.box.ap50.tolist() if hasattr(val_metrics.box, 'ap50') and val_metrics.box.ap50 is not None else []
        map_per_class = val_metrics.box.ap.tolist() if hasattr(val_metrics.box, 'ap') and val_metrics.box.ap is not None else []
        
        for idx, cname in enumerate(class_names):
            per_class_results[cname] = {
                "precision": round(p_per_class[idx], 4) if idx < len(p_per_class) else None,
                "recall": round(r_per_class[idx], 4) if idx < len(r_per_class) else None,
                "mAP50": round(map50_per_class[idx], 4) if idx < len(map50_per_class) else None,
                "mAP50_95": round(map_per_class[idx], 4) if idx < len(map_per_class) else None,
            }
    except Exception as e:
        print(f"[METRIC EXTRACTION NOTE]: {e}")
        mp, mr, map50, map50_95 = 0.0, 0.0, 0.0, 0.0

    # Read training history from results.csv if present
    train_history = []
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        import csv
        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row = {k.strip(): float(v.strip()) for k, v in row.items() if v.strip()}
                train_history.append(cleaned_row)

    final_losses = {}
    if train_history:
        last_row = train_history[-1]
        final_losses = {
            "train/box_loss": last_row.get("train/box_loss"),
            "train/cls_loss": last_row.get("train/cls_loss"),
            "train/dfl_loss": last_row.get("train/dfl_loss"),
            "val/box_loss": last_row.get("val/box_loss"),
            "val/cls_loss": last_row.get("val/cls_loss"),
            "val/dfl_loss": last_row.get("val/dfl_loss"),
        }

    # Confusion matrix
    cm_path = run_dir / "confusion_matrix.png"
    cm_normalized_path = run_dir / "confusion_matrix_normalized.png"
    if cm_path.exists():
        _safe_chunked_copy(str(cm_path), str(REPORTS_DIR / "confusion_matrix.png"))
    if cm_normalized_path.exists():
        _safe_chunked_copy(str(cm_normalized_path), str(REPORTS_DIR / "confusion_matrix_normalized.png"))

    eval_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "YOLOv8n (nano)",
        "model_file": str(dest_model_path.resolve()),
        "model_size_mb": round(model_size_mb, 2),
        "target_document": "PAN Card (Permanent Account Number)",
        "training_dataset": {
            "source": "smartxtract/pan-card-ygz7o (Roboflow Universe)",
            "license": "CC BY 4.0",
            "train_unique_images": 107,
            "train_annotations": 428,
            "val_unique_images": 44,
            "val_annotations": 176,
            "total_unique_images": 151,
            "deduplication_method": "Exact image MD5 hash filter",
            "rect_letterbox": True
        },
        "training_parameters": {
            "epochs_trained": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "optimizer": "auto",
            "seed": 42,
            "patience": 10,
            "train_duration_seconds": round(train_duration, 1),
            "device": device_name
        },
        "validation_metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "final_losses": final_losses,
            "per_class": per_class_results
        },
        "disclaimer": (
            "Field detection metrics reflect bounding box localization performance on the held-out "
            "Roboflow validation split. These metrics do NOT represent document genuineness classification, "
            "forgery detection accuracy, or official Income Tax Department verification."
        )
    }

    report_json = REPORTS_DIR / "pan_field_detection_evaluation.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print("\n" + "=" * 70)
    print("PAN FIELD DETECTION TRAINING & EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model saved to: {dest_model_path} ({model_size_mb:.2f} MB)")
    print(f"Overall Precision: {mp:.4f}")
    print(f"Overall Recall:    {mr:.4f}")
    print(f"Overall mAP50:     {map50:.4f}")
    print(f"Overall mAP50-95:  {map50_95:.4f}")
    print("\nPer-Class Breakdown:")
    for cname, m in per_class_results.items():
        print(f"  {cname:15s} | P: {m['precision']} | R: {m['recall']} | mAP50: {m['mAP50']} | mAP50-95: {m['mAP50_95']}")
    print(f"\nFinal Losses: {final_losses}")
    print(f"Full evaluation report: {report_json}")
    print("=" * 70)

    return eval_report


if __name__ == "__main__":
    train_pan_detector(epochs=20, batch_size=8, imgsz=640)
