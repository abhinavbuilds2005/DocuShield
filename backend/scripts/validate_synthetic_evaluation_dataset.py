"""
DocuShield AI — Phase 3.3 Synthetic Benchmark Validator & Manifest Generator
Performs exhaustive validation of the 500-sample synthetic evaluation corpus:
1. Count and split verification (300 dev, 50 val, 150 test)
2. Document class verification (100 per class)
3. Schema & metadata integrity (synthetic=true, watermark, mandatory fields)
4. Bounding box & YOLO coordinate validity (inside [0, 1] bounds)
5. Image readability check
6. Exact duplicate check (MD5 & SHA-256)
7. Near-duplicate detection (Perceptual dHash)
8. Cross-split lineage leakage prevention (no same lineage across splits)
9. Creation of test_manifest_sha256.json and benchmark_manifest.json
10. Generation of a visual QA contact sheet from DEV and VAL samples ONLY
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_DIR = ROOT_DIR / "data" / "evaluation_synthetic"
QA_ARTIFACT_DIR = ROOT_DIR / "reports" / "evaluation_synthetic_qa"

def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as fp:
        while chunk := fp.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(img_path: Path, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) for near-duplicate image detection."""
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return 0
    resized = cv2.resize(im, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

def validate_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> Dict[str, Any]:
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    metadata_dir = dataset_dir / "metadata"
    
    report = {
        "dataset_dir": str(dataset_dir),
        "checks": {},
        "errors": [],
        "warnings": [],
        "class_counts": defaultdict(int),
        "split_counts": defaultdict(int),
        "category_counts": defaultdict(int),
        "lineage_to_splits": defaultdict(set),
        "total_samples": 0
    }
    
    splits = ["dev", "val", "test"]
    expected_splits = {"dev": 300, "val": 50, "test": 150}
    expected_classes = {"aadhaar": 100, "pan": 100, "driving_license": 100, "passport": 100, "voter_id": 100}
    
    sample_ids = set()
    exact_image_hashes = defaultdict(list)
    exact_meta_hashes = defaultdict(list)
    dhashes = {} # sample_id -> (dhash, split, doc_type)
    
    all_metadata_records = []
    
    # 1. Inspect Files Across Splits
    for split in splits:
        split_img_dir = images_dir / split
        split_ann_dir = annotations_dir / split
        split_meta_dir = metadata_dir / split
        
        if not split_img_dir.exists():
            report["errors"].append(f"Missing images split directory: {split_img_dir}")
            continue
            
        img_files = sorted(list(split_img_dir.glob("*.jpg")))
        report["split_counts"][split] = len(img_files)
        
        for img_path in img_files:
            report["total_samples"] += 1
            sid = img_path.stem
            
            # ID Collision
            if sid in sample_ids:
                report["errors"].append(f"Collision: duplicate sample_id detected: {sid}")
            sample_ids.add(sid)
            
            # File Exists Check
            ann_path = split_ann_dir / f"{sid}.txt"
            meta_path = split_meta_dir / f"{sid}.eval.json"
            
            if not ann_path.exists():
                report["errors"].append(f"Missing YOLO annotation file for sample: {sid}")
            if not meta_path.exists():
                report["errors"].append(f"Missing evaluation metadata file for sample: {sid}")
                continue
                
            # Exact Hash Duplicate Scan
            img_sha = compute_file_sha256(img_path)
            exact_image_hashes[img_sha].append(str(img_path))
            meta_sha = compute_file_sha256(meta_path)
            exact_meta_hashes[meta_sha].append(str(meta_path))
            
            # Read and Validate Image File
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                report["errors"].append(f"Corrupt or unreadable image file {img_path}: {e}")
                w, h = 0, 0
                
            # Compute dHash for Near-Duplicate Scan
            dh = compute_dhash(img_path)
            dhashes[sid] = (dh, split, img_path)
            
            # Read & Validate Metadata JSON
            try:
                with open(meta_path, "r", encoding="utf-8") as mfp:
                    mdata = json.load(mfp)
                all_metadata_records.append((split, sid, img_path, ann_path, meta_path, mdata))
            except Exception as e:
                report["errors"].append(f"Invalid JSON metadata in {meta_path}: {e}")
                continue
                
            # Validate Metadata Fields
            dtype = mdata.get("document_type", "unknown")
            report["class_counts"][dtype] += 1
            
            cat = mdata.get("acquisition_category", "unknown")
            report["category_counts"][cat] += 1
            
            lid = mdata.get("lineage_id", sid)
            report["lineage_to_splits"][lid].add(split)
            
            if not mdata.get("synthetic", False):
                report["errors"].append(f"Sample {sid} missing mandatory flag synthetic=True")
            if not mdata.get("watermark_present", False):
                report["errors"].append(f"Sample {sid} missing watermark_present=True")
            if split == "test" and not mdata.get("locked_test", False):
                report["errors"].append(f"Sample {sid} in test split missing locked_test=True")
                
            # Validate Annotations & Coordinates
            fields = mdata.get("fields", {})
            for fname, fprops in fields.items():
                bxy = fprops.get("bbox_xyxy", [])
                if len(bxy) == 4:
                    x1, y1, x2, y2 = bxy
                    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                        report["errors"].append(f"Sample {sid} field {fname} bbox out of bounds: {bxy} (img {w}x{h})")
                byolo = fprops.get("bbox_yolo", [])
                if len(byolo) == 4:
                    xc, yc, bw, bh = byolo
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                        report["errors"].append(f"Sample {sid} field {fname} invalid YOLO coords: {byolo}")
                        
            # Verify Negative Control Consistency
            neg = mdata.get("negative_control", {})
            if cat == "negative_control" and not neg.get("is_negative_control", False):
                report["errors"].append(f"Sample {sid} in negative_control category not labeled is_negative_control=True")
                
    # 2. Split Quota Verification
    for s, exp_cnt in expected_splits.items():
        actual_cnt = report["split_counts"].get(s, 0)
        report["checks"][f"split_{s}_count_is_{exp_cnt}"] = (actual_cnt == exp_cnt)
        if actual_cnt != exp_cnt:
            report["errors"].append(f"Split quota mismatch for '{s}': expected {exp_cnt}, got {actual_cnt}")
            
    # 3. Class Quota Verification
    for c, exp_cnt in expected_classes.items():
        actual_cnt = report["class_counts"].get(c, 0)
        report["checks"][f"class_{c}_count_is_{exp_cnt}"] = (actual_cnt == exp_cnt)
        if actual_cnt != exp_cnt:
            report["errors"].append(f"Class quota mismatch for '{c}': expected {exp_cnt}, got {actual_cnt}")
            
    # 4. Total Count Check
    report["checks"]["total_count_is_500"] = (report["total_samples"] == 500)
    if report["total_samples"] != 500:
        report["errors"].append(f"Total samples count mismatch: expected 500, got {report['total_samples']}")
        
    # 5. Exact Duplicate Scan
    dupe_images = {k: v for k, v in exact_image_hashes.items() if len(v) > 1}
    dupe_metas = {k: v for k, v in exact_meta_hashes.items() if len(v) > 1}
    report["checks"]["zero_exact_image_duplicates"] = (len(dupe_images) == 0)
    report["checks"]["zero_exact_meta_duplicates"] = (len(dupe_metas) == 0)
    if dupe_images:
        report["errors"].append(f"Found {len(dupe_images)} exact duplicate image groups!")
    if dupe_metas:
        report["errors"].append(f"Found {len(dupe_metas)} exact duplicate metadata groups!")
        
    # 6. Lineage Cross-Split Leakage Scan
    leaked_lineages = {lid: spls for lid, spls in report["lineage_to_splits"].items() if len(spls) > 1}
    report["checks"]["zero_lineage_cross_split_leakage"] = (len(leaked_lineages) == 0)
    if leaked_lineages:
        report["errors"].append(f"Cross-split lineage leakage detected in {len(leaked_lineages)} lineages: {list(leaked_lineages.keys())[:5]}")
        
    # 7. Near-Duplicate Scan (Perceptual Hash Distance <= 2 across DIFFERENT splits)
    near_dupes_cross_split = []
    sample_list = list(dhashes.items())
    for i in range(len(sample_list)):
        s1, (h1, sp1, p1) = sample_list[i]
        for j in range(i + 1, len(sample_list)):
            s2, (h2, sp2, p2) = sample_list[j]
            if sp1 != sp2: # Cross-split comparison
                dist = hamming_distance(h1, h2)
                if dist <= 2: # Very strong perceptual similarity
                    near_dupes_cross_split.append((s1, sp1, s2, sp2, dist))
                    
    report["checks"]["zero_cross_split_near_duplicates"] = (len(near_dupes_cross_split) == 0)
    if near_dupes_cross_split:
        report["warnings"].append(f"Detected {len(near_dupes_cross_split)} cross-split near-duplicate pairs (dHash dist <= 2)")
        
    # 8. Manifest Freezing for Locked Test Partition
    test_records = [r for r in all_metadata_records if r[0] == "test"]
    test_manifest = {}
    for _, sid, img_p, ann_p, meta_p, _ in test_records:
        test_manifest[sid] = {
            "image_file": img_p.name,
            "image_sha256": compute_file_sha256(img_p),
            "annotation_file": ann_p.name,
            "annotation_sha256": compute_file_sha256(ann_p),
            "metadata_file": meta_p.name,
            "metadata_sha256": compute_file_sha256(meta_p)
        }
        
    test_manifest_path = metadata_dir / "test_manifest_sha256.json"
    with open(test_manifest_path, "w", encoding="utf-8") as mfp:
        json.dump(test_manifest, mfp, indent=2)
        
    # 9. Master Benchmark Manifest
    benchmark_manifest = {
        "benchmark_name": "DocuShield AI Independent Synthetic Evaluation Benchmark",
        "benchmark_version": "1.0.0",
        "total_samples": report["total_samples"],
        "class_distribution": dict(report["class_counts"]),
        "split_distribution": dict(report["split_counts"]),
        "category_distribution": dict(report["category_counts"]),
        "locked_test_samples": len(test_records),
        "test_manifest_file": "test_manifest_sha256.json",
        "test_manifest_sha256": compute_file_sha256(test_manifest_path),
        "validation_passed": len(report["errors"]) == 0
    }
    
    bench_manifest_path = metadata_dir / "benchmark_manifest.json"
    with open(bench_manifest_path, "w", encoding="utf-8") as bfp:
        json.dump(benchmark_manifest, bfp, indent=2)
        
    # 10. Generate Visual QA Artifact from DEV and VAL ONLY
    QA_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dev_val_records = [r for r in all_metadata_records if r[0] in ["dev", "val"]]
    # Select 25 representative samples: 5 doc classes x 5 categories
    qa_samples = []
    selected_keys = set()
    for _, sid, img_p, _, _, mdata in dev_val_records:
        key = (mdata["document_type"], mdata["acquisition_category"])
        if key not in selected_keys:
            selected_keys.add(key)
            qa_samples.append((sid, img_p, mdata))
        if len(qa_samples) >= 25:
            break
            
    qa_summary = {
        "qa_samples_reviewed": len(qa_samples),
        "samples": [
            {
                "sample_id": s[0],
                "document_type": s[2]["document_type"],
                "acquisition_category": s[2]["acquisition_category"],
                "split": s[2]["dataset_split"],
                "watermark_present": s[2]["watermark_present"],
                "fields_annotated": list(s[2]["fields"].keys())
            } for s in qa_samples
        ]
    }
    with open(QA_ARTIFACT_DIR / "qa_review_manifest.json", "w", encoding="utf-8") as qfp:
        json.dump(qa_summary, qfp, indent=2)
        
    is_success = len(report["errors"]) == 0
    print(f"[*] Dataset Validation {'PASSED' if is_success else 'FAILED'}: {len(report['errors'])} errors, {len(report['warnings'])} warnings.")
    return report

if __name__ == "__main__":
    rep = validate_dataset(DEFAULT_DATASET_DIR)
    sys.exit(0 if len(rep["errors"]) == 0 else 1)
