import json
from backend.scripts.run_benchmark import run_benchmark

b = run_benchmark()
print("\n--- MISCLASSIFICATIONS ---")
for r in b["details"]:
    if r["status"] in ("FALSE_POSITIVE", "FALSE_NEGATIVE"):
        print(f"[{r['status']}] {r['filename']}: actual={r['actual_label']} pred={r['predicted_verdict']} score={r['authenticity_score']}")
        print(f"   Triggers: {r['critical_triggers']}")
        print(f"   Tamper types: {r['tampering_types']}")
