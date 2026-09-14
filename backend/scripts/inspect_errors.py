import os
import sys

# Ensure repository root is on sys.path for direct script invocation
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
from backend.scripts.run_benchmark import run_benchmark

b = run_benchmark()
print("\n--- MISCLASSIFICATIONS ---")
misclassifications = 0
for r in b.get("details", []):
    status = r.get("classification_status") or r.get("status")
    if status in ("FALSE_POSITIVE", "FALSE_NEGATIVE"):
        misclassifications += 1
        actual = r.get("ground_truth_label") or r.get("actual_label")
        pred = r.get("final_prediction") or r.get("predicted_verdict")
        score = r.get("authenticity_score", 0.0)
        triggers = r.get("critical_triggers", [])
        tamper_types = r.get("tampering_types", [])
        print(f"[{status}] {r.get('filename')}: actual={actual} pred={pred} score={score}")
        print(f"   Triggers: {triggers}")
        print(f"   Tamper types: {tamper_types}")

if misclassifications == 0:
    print("Zero misclassifications detected. All evaluations aligned with ground truth.")
