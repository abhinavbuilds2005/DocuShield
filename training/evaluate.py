"""
Evaluation and Metrics Tracking Module
Computes:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix (TP, FP, TN, FN)
- Calibration statement (Prototype evaluation vs production)
"""

from typing import Dict, Any, List


def compute_screening_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    Computes standard classification evaluation metrics.
    1 = Tampered/Fraudulent, 0 = Genuine/Authentic
    """
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "sample_count": 0
        }

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = round(((tp + tn) / total) * 100.0, 2)
    precision = round((tp / (tp + fp)) * 100.0, 2) if (tp + fp) > 0 else 0.0
    recall = round((tp / (tp + fn)) * 100.0, 2) if (tp + fn) > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 2) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sample_count": total,
        "evaluation_note": "Controlled prototype evaluation." if total < 50 else "Full benchmark evaluation."
    }
