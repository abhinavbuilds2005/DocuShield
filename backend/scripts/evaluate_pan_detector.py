"""
Evaluation and Metrics Generation for Trained PAN Field Detector
"""

import os
import sys
import json
import time
from pathlib import Path
import torch

def _safe_chunked_copy(src, dst):
    with open(src, "rb") as rf, open(dst, "wb") as wf:
        while chunk := rf.read(65536):
            wf.write(chunk)

from ultralytics import YOLO

DATASET_ROOT = Path("data/processed/pan_field_detection")
DATA_YAML = DATASET_ROOT / "data.yaml"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/pan_field_detection")
RUN_DIR = Path("runs/detect/runs/field_detection/pan_yolov8n")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

best_weights = RUN_DIR / "weights" / "best.pt"
dest_model_path = MODELS_DIR / "pan_field_detector.pt"

assert best_weights.exists(), f"Could not find weights at {best_weights}"
_safe_chunked_copy(str(best_weights), str(dest_model_path))
print(f"[SUCCESS] Copied {best_weights} -> {dest_model_path}")

model_size_mb = os.path.getsize(dest_model_path) / (1024 * 1024)
print(f"Model file size: {model_size_mb:.2f} MB")

device = "0" if torch.cuda.is_available() else "cpu"
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

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

mp = float(val_metrics.box.mp)
mr = float(val_metrics.box.mr)
map50 = float(val_metrics.box.map50)
map50_95 = float(val_metrics.box.map)

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

# Read training history from results.csv
train_history = []
results_csv = RUN_DIR / "results.csv"
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
        "train/box_loss": round(last_row.get("train/box_loss", 0), 4),
        "train/cls_loss": round(last_row.get("train/cls_loss", 0), 4),
        "train/dfl_loss": round(last_row.get("train/dfl_loss", 0), 4),
        "val/box_loss": round(last_row.get("val/box_loss", 0), 4),
        "val/cls_loss": round(last_row.get("val/cls_loss", 0), 4),
        "val/dfl_loss": round(last_row.get("val/dfl_loss", 0), 4),
    }

# Copy plots
for plot_name in ["confusion_matrix.png", "confusion_matrix_normalized.png", "results.png"]:
    src_p = RUN_DIR / plot_name
    if src_p.exists():
        _safe_chunked_copy(str(src_p), str(REPORTS_DIR / plot_name))

eval_report = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_architecture": "YOLOv8n (nano)",
    "model_file": str(dest_model_path.resolve()),
    "model_size_mb": round(model_size_mb, 2),
    "target_document": "PAN Card (Permanent Account Number)",
    "dataset_used": {
        "source": "smartxtract/pan-card-ygz7o (Roboflow Universe)",
        "license": "CC BY 4.0",
        "train_unique_images": 107,
        "train_annotations": 428,
        "val_unique_images": 44,
        "val_annotations": 176,
        "total_unique_images": 151,
        "total_annotations": 604,
        "deduplication_method": "Exact image MD5 hash filter",
        "letterbox_rect": True
    },
    "training_configuration": {
        "epochs_trained": 20,
        "batch_size": 8,
        "imgsz": 640,
        "optimizer": "AdamW (auto)",
        "seed": 42,
        "patience": 10,
        "device": device_name,
        "conservative_augmentations": {
            "degrees": 3.0,
            "translate": 0.04,
            "scale": 0.08,
            "shear": 1.0,
            "perspective": 0.0003,
            "mosaic": 0.0,
            "mixup": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "rect": True
        }
    },
    "metrics_held_out_validation": {
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "per_class": per_class_results,
        "final_losses": final_losses
    },
    "forensic_disclaimer": (
        "Field detection metrics reflect bounding box localization performance on the held-out "
        "Roboflow validation split. These metrics do NOT represent document genuineness classification, "
        "forgery detection accuracy, or official Income Tax Department verification."
    )
}

report_json = REPORTS_DIR / "pan_field_detection_evaluation.json"
with open(report_json, "w", encoding="utf-8") as f:
    json.dump(eval_report, f, indent=2)

print("\n" + "=" * 70)
print("PAN FIELD DETECTION EVALUATION COMPLETE")
print("=" * 70)
print(f"Model: {dest_model_path} ({model_size_mb:.2f} MB)")
print(f"Overall Precision: {mp:.4f}")
print(f"Overall Recall:    {mr:.4f}")
print(f"Overall mAP50:     {map50:.4f}")
print(f"Overall mAP50-95:  {map50_95:.4f}")
print("\nPer-Class Metrics:")
for cname, m in per_class_results.items():
    print(f"  {cname:15s} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | mAP50: {m['mAP50']:.4f} | mAP50-95: {m['mAP50_95']:.4f}")
print(f"\nFinal Losses: {final_losses}")
print(f"Evaluation report saved to: {report_json}")
print("=" * 70)
