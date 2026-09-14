"""
Dataset Loader for Future ML/Deep Learning Document Tampering Models
Supports:
- Structured train / validation / test splits
- Multimodal data loading (image arrays + metadata annotations)
- Document category filtering (passport, visa, national_id, driving_license, permit)
"""

import os
import json
import glob
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np


class DocumentDatasetLoader:
    """
    Loads labeled identity document datasets for training and evaluation.
    """

    def __init__(self, root_dir: Optional[str] = None):
        if root_dir is None:
            # Anchor to project root datasets/ directory
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.root_dir = os.path.join(workspace_root, "datasets")
        else:
            self.root_dir = root_dir
        self.annotations_dir = os.path.join(self.root_dir, "annotations")

    def load_dataset_manifest(self, manifest_file: str = "synthetic_testbed_manifest.json") -> Dict[str, Any]:
        """Loads annotation manifest from annotations directory."""
        path = os.path.join(self.annotations_dir, manifest_file)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_dataset_splits(
        self,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Splits available samples into train, validation, and test partitions.
        Returns: {"train": [...], "val": [...], "test": [...]}
        """
        manifest = self.load_dataset_manifest()
        items = []

        for fn, meta in manifest.items():
            doc_type = meta.get("type", "unknown")
            subpath = os.path.join(self.root_dir, doc_type, fn)
            if os.path.exists(subpath):
                items.append({
                    "filename": fn,
                    "filepath": subpath,
                    "document_type": doc_type,
                    "label": 1 if meta.get("label") == "TAMPERED" else 0,
                    "label_str": meta.get("label", "GENUINE")
                })

        np.random.seed(seed)
        indices = np.random.permutation(len(items))

        n_total = len(items)
        if n_total == 0:
            return {"train": [], "val": [], "test": []}

        n_test = max(1, int(n_total * test_ratio)) if n_total > 2 else 1
        n_val = max(1, int(n_total * val_ratio)) if n_total > 3 else 0
        n_train = max(1, n_total - n_test - n_val)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return {
            "train": [items[i] for i in train_idx],
            "val": [items[i] for i in val_idx],
            "test": [items[i] for i in test_idx]
        }
