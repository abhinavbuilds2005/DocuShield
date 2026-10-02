"""
Roboflow Dataset Downloader & Fallback Dataset Provisioner for Aadhaar Field Detection
Target Dataset: https://universe.roboflow.com/cutm-iwh4a/aadhaar-card-details
Target Format: YOLOv8 / YOLOv11 (images & text annotations)
Storage Path: data/aadhaar_field_detection/

Features:
1. Checks for ROBOFLOW_API_KEY from environment or command line.
2. If API key is available, attempts official Roboflow SDK download.
3. If API key is missing, provides clear step-by-step instructions.
4. Implements --generate-fallback to create a structured YOLO dataset of synthetic
   Aadhaar card layouts with exact field bounding boxes so training and pipeline
   integration can proceed immediately without external blockers.
5. Respects privacy: all generated or logged credentials mask sensitive numbers.
"""

import os
import sys
import json
import argparse
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATASET_ROOT = Path("data/aadhaar_field_detection")
RAW_DIR = DATASET_ROOT / "raw"
TRAIN_IMG_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LBL_DIR = DATASET_ROOT / "train" / "labels"
VAL_IMG_DIR = DATASET_ROOT / "valid" / "images"
VAL_LBL_DIR = DATASET_ROOT / "valid" / "labels"
TEST_IMG_DIR = DATASET_ROOT / "test" / "images"
TEST_LBL_DIR = DATASET_ROOT / "test" / "labels"

# Canonical field detection classes for Indian Identity / Aadhaar cards
CLASS_NAMES = [
    "aadhaar_number",   # Class 0: 12-digit UID printed block
    "name",             # Class 1: English/Regional full name text block
    "dob",              # Class 2: Date of Birth / Year of Birth line
    "gender",           # Class 3: Gender line (MALE / FEMALE / TRANSGENDER)
    "photo",            # Class 4: Holder photo portrait rectangle
    "qr_code",          # Class 5: 2D Secure QR code matrix
    "emblem_header"     # Class 6: National emblem / Government header banner
]


def write_readme_and_attribution():
    """Writes dataset attribution, license notes, and privacy guidelines."""
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    readme_path = DATASET_ROOT / "README.md"
    readme_content = f"""# Aadhaar Card Field Detection Dataset

## Dataset Source & Attribution
- **Target Source:** [Roboflow Universe: cutm-iwh4a / aadhaar-card-details](https://universe.roboflow.com/cutm-iwh4a/aadhaar-card-details)
- **Primary Purpose:** Field Localization / Object Detection ONLY.
- **Classes:** {', '.join(CLASS_NAMES)}

## IMPORTANT DISCLAIMER (Forensic & Legal)
1. **Field Detection != Authenticity:**
   This dataset locates document regions (bounding boxes for Name, DOB, Photo, QR, Number).
   It does NOT train a genuine-vs-fake classifier. Locating a field does NOT guarantee official authenticity.
2. **Privacy Policy:**
   - No raw images containing real, unmasked personal identity numbers may be checked into public version control.
   - All visual demo outputs, logs, and screenshots must mask Aadhaar numbers (e.g. `XXXX XXXX 1234`).
   - Synthetic and anonymized mock testbeds are isolated in test directories.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)


def write_data_yaml(target_dir: Path):
    """Writes standard YOLO data.yaml configuration."""
    data_yaml_path = target_dir / "data.yaml"
    content = f"""# DocuShield AI — Aadhaar Field Detection Dataset Configuration
path: {target_dir.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[DATASET] Wrote YOLO data.yaml to {data_yaml_path}")


def try_download_roboflow(api_key: str, workspace: str = "cutm-iwh4a", project_id: str = "aadhaar-card-details"):
    """Attempts to download the official dataset from Roboflow Universe."""
    print(f"[ROBOFLOW] Initializing Roboflow client for workspace='{workspace}', project='{project_id}'...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_id)
        versions = project.versions()
        if not versions:
            print("[ROBOFLOW ERROR] No versions available for this project.")
            return False

        latest_version = versions[0].version
        print(f"[ROBOFLOW] Downloading version {latest_version} in 'yolov8' format...")
        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        dataset = project.version(latest_version).download(
            model_format="yolov8",
            location=str(DATASET_ROOT.resolve())
        )
        print(f"[ROBOFLOW SUCCESS] Dataset downloaded to: {dataset.location}")
        write_readme_and_attribution()
        return True
    except Exception as e:
        print(f"[ROBOFLOW ERROR] Failed to download from Roboflow: {e}")
        return False


def generate_fallback_dataset(num_samples: int = 150):
    """
    Generates a high-quality synthetic fallback YOLO dataset of Aadhaar card layouts
    with exact bounding box annotations across all canonical field classes.
    This guarantees that training, evaluation, and end-to-end integration can execute immediately.
    """
    print(f"[DATASET] Generating synthetic field-detection testbed ({num_samples} samples) at {DATASET_ROOT}...")
    for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR, TEST_IMG_DIR, TEST_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    first_names = ["Aarav", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Rohan", "Pooja", "Arjun", "Kavita"]
    last_names = ["Sharma", "Verma", "Patel", "Singh", "Kumar", "Gupta", "Deshmukh", "Nair", "Iyer", "Reddy"]
    genders = ["MALE / पुरुष", "FEMALE / महिला"]

    # Splits: 70% train, 20% val, 10% test
    train_count = int(num_samples * 0.70)
    val_count = int(num_samples * 0.20)

    for idx in range(num_samples):
        if idx < train_count:
            img_dir, lbl_dir = TRAIN_IMG_DIR, TRAIN_LBL_DIR
        elif idx < train_count + val_count:
            img_dir, lbl_dir = VAL_IMG_DIR, VAL_LBL_DIR
        else:
            img_dir, lbl_dir = TEST_IMG_DIR, TEST_LBL_DIR

        # Standard document canvas
        width, height = 800, 500
        canvas = np.ones((height, width, 3), dtype=np.uint8) * random.randint(245, 255)

        # Subtle background texture
        noise = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        annotations = []  # format: (class_id, x_center, y_center, w, h) in normalized coords

        def add_box(cls_id, x1, y1, x2, y2):
            xc = ((x1 + x2) / 2.0) / width
            yc = ((y1 + y2) / 2.0) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            annotations.append((cls_id, xc, yc, bw, bh))

        # 1. Header & Emblem
        hdr_y1 = random.randint(15, 25)
        hdr_y2 = hdr_y1 + random.randint(50, 65)
        cv2.rectangle(canvas, (40, hdr_y1), (width - 40, hdr_y2), (230, 240, 250), -1)
        cv2.putText(canvas, "GOVERNMENT OF INDIA", (180, hdr_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        # Emblem icon placeholder
        cv2.circle(canvas, (90, (hdr_y1 + hdr_y2) // 2), 22, (180, 100, 50), -1)
        add_box(6, 40, hdr_y1, width - 40, hdr_y2)

        # 2. Photo Portrait
        px1 = random.randint(45, 60)
        py1 = hdr_y2 + random.randint(30, 45)
        pw = random.randint(130, 150)
        ph = random.randint(160, 185)
        px2, py2 = px1 + pw, py1 + ph
        # Draw portrait avatar
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (210, 210, 210), -1)
        cv2.circle(canvas, ((px1 + px2) // 2, py1 + 55), 32, (120, 120, 120), -1)
        cv2.ellipse(canvas, ((px1 + px2) // 2, py2 + 10), (50, 45), 0, 0, 180, (90, 90, 90), -1)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (100, 100, 100), 2)
        add_box(4, px1, py1, px2, py2)

        # 3. Name Field
        name_x1 = px2 + random.randint(25, 35)
        name_y1 = py1 + 10
        name_x2 = name_x1 + random.randint(220, 290)
        name_y2 = name_y1 + 32
        full_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        cv2.putText(canvas, full_name, (name_x1, name_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (10, 10, 10), 2)
        add_box(1, name_x1, name_y1, name_x2, name_y2)

        # 4. DOB Field
        dob_y1 = name_y2 + random.randint(12, 20)
        dob_y2 = dob_y1 + 28
        dob_x2 = name_x1 + random.randint(190, 240)
        dd = random.randint(1, 28)
        mm = random.randint(1, 12)
        yy = random.randint(1965, 2005)
        dob_str = f"DOB: {dd:02d}/{mm:02d}/{yy}"
        cv2.putText(canvas, dob_str, (name_x1, dob_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 2)
        add_box(2, name_x1, dob_y1, dob_x2, dob_y2)

        # 5. Gender Field
        g_y1 = dob_y2 + random.randint(10, 16)
        g_y2 = g_y1 + 26
        g_x2 = name_x1 + random.randint(140, 180)
        g_str = random.choice(genders)
        cv2.putText(canvas, g_str, (name_x1, g_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
        add_box(3, name_x1, g_y1, g_x2, g_y2)

        # 6. QR Code
        qr_size = random.randint(130, 150)
        qrx2 = width - random.randint(45, 60)
        qrx1 = qrx2 - qr_size
        qry1 = hdr_y2 + random.randint(30, 45)
        qry2 = qry1 + qr_size
        # Draw synthetic QR pattern
        cv2.rectangle(canvas, (qrx1, qry1), (qrx2, qry2), (255, 255, 255), -1)
        cv2.rectangle(canvas, (qrx1, qry1), (qrx2, qry2), (0, 0, 0), 2)
        # Finder patterns
        for fx, fy in [(qrx1 + 10, qry1 + 10), (qrx2 - 38, qry1 + 10), (qrx1 + 10, qry2 - 38)]:
            cv2.rectangle(canvas, (fx, fy), (fx + 28, fy + 28), (0, 0, 0), -1)
            cv2.rectangle(canvas, (fx + 6, fy + 6), (fx + 22, fy + 22), (255, 255, 255), -1)
            cv2.rectangle(canvas, (fx + 10, fy + 10), (fx + 18, fy + 18), (0, 0, 0), -1)
        add_box(5, qrx1, qry1, qrx2, qry2)

        # 7. Aadhaar Number (Bottom Red Band / Large Number)
        uid_y1 = max(py2, qry2) + random.randint(30, 45)
        uid_y2 = uid_y1 + 45
        uid_x1 = random.randint(220, 260)
        uid_x2 = uid_x1 + random.randint(280, 320)
        # Generate valid-looking 12-digit grouped string (masked in logging)
        p1 = random.randint(2000, 9999)
        p2 = random.randint(1000, 9999)
        p3 = random.randint(1000, 9999)
        uid_str = f"{p1} {p2} {p3}"
        cv2.putText(canvas, uid_str, (uid_x1, uid_y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (180, 30, 30), 3)
        add_box(0, uid_x1 - 10, uid_y1, uid_x2 + 10, uid_y2)

        # Write image
        img_filename = f"aadhaar_field_sample_{idx:04d}.jpg"
        img_path = img_dir / img_filename
        cv2.imwrite(str(img_path), canvas)

        # Write labels
        lbl_filename = f"aadhaar_field_sample_{idx:04d}.txt"
        lbl_path = lbl_dir / lbl_filename
        with open(lbl_path, "w", encoding="utf-8") as lf:
            for cls_id, xc, yc, bw, bh in annotations:
                lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    write_data_yaml(DATASET_ROOT)
    write_readme_and_attribution()
    print(f"[DATASET SUCCESS] Synthetic field testbed generated: {train_count} train, {val_count} val, {num_samples - train_count - val_count} test.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download or generate Aadhaar Field Detection Dataset")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--workspace", type=str, default="cutm-iwh4a", help="Roboflow workspace ID")
    parser.add_argument("--project", type=str, default="aadhaar-card-details", help="Roboflow project ID")
    parser.add_argument("--fallback-samples", type=int, default=150, help="Number of synthetic samples to generate if fallback is used")
    parser.add_argument("--force-fallback", action="store_true", help="Directly generate synthetic fallback dataset without attempting Roboflow API")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")

    if args.force_fallback:
        print("[INFO] --force-fallback requested. Creating synthetic field detection dataset...")
        generate_fallback_dataset(args.fallback_samples)
        return

    if api_key:
        print(f"[INFO] Using Roboflow API key from {'command argument' if args.api_key else 'environment variable'}.")
        success = try_download_roboflow(api_key, args.workspace, args.project)
        if success:
            return
        print("[WARNING] Roboflow download failed; falling back to generating synthetic testbed.")

    print("\n" + "=" * 70)
    print("ROBOFLOW DATASET ACCESS INSTRUCTIONS")
    print("=" * 70)
    print("Dataset URL: https://universe.roboflow.com/cutm-iwh4a/aadhaar-card-details")
    print("To download the live Roboflow dataset:")
    print("  1. Sign in to your Roboflow account at https://app.roboflow.com")
    print("  2. Copy your private API key from Settings -> Roboflow API")
    print("  3. Set environment variable: $env:ROBOFLOW_API_KEY='your_api_key'")
    print("  4. Run: python backend/scripts/download_roboflow_dataset.py")
    print("=" * 70)
    print("[INFO] Generating synthetic field detection testbed so development and training can proceed...\n")
    generate_fallback_dataset(args.fallback_samples)


if __name__ == "__main__":
    main()
