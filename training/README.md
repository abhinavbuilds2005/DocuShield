# DocuShield Training Pipeline Specification
**SIH 2026 Problem Statement: SIH26188 — AI-Based Fake Identity & Document Screening System**

---

## 1. Overview

This directory provides the modular machine learning and deep learning training infrastructure for DocuShield. It enables training document classification and tampering detection backbones when an authorized, legitimate labeled dataset is obtained.

### Modular Architecture
```
training/
├── dataset_loader.py    # Loads dataset splits (train/val/test) across 5 document types
├── preprocessing.py     # Image normalization, resizing, SRM noise residuals
├── augmentation.py      # Camera blur, compression, lighting, and noise simulation
├── train.py             # Training loop, loss optimization, artifact saving
├── evaluate.py          # Accuracy, precision, recall, F1, confusion matrix
├── inference.py         # Inference wrapper connecting weights to FastAPI
├── artifacts/           # Saved weights, checkpoints, and evaluation reports
└── README.md            # Documentation
```

---

## 2. Technical Honesty & Real-World Constraints

> [!NOTE]
> Currently, no publicly licensed real-world dataset containing millions of authentic government passports, visas, and national IDs exists due to sovereign privacy regulations.
> 
> Therefore, this pipeline is structured to operate in **Controlled Prototype Evaluation** mode using synthetic samples.
> **We DO NOT claim false 100% accuracy or real-world model training on proprietary datasets.**

---

## 3. How to Run Training

To run the training pipeline:

```bash
python training/train.py
```

Trained model artifacts and metadata will be saved to `training/artifacts/model_metadata.json`.
