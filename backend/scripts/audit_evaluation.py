import os
import sys
import json
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.fusion import DocumentScreeningPipeline

GENUINE_DIR = r"D:\original test data"
FAKE_DIR = r"D:\fake data"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def discover(d):
    res = []
    for root, _, files in os.walk(d):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                res.append(os.path.join(root, f))
    return sorted(res)

def main():
    pipeline = DocumentScreeningPipeline()
    genuine_files = discover(GENUINE_DIR)
    fake_files = discover(FAKE_DIR)
    
    print(f"Discovered {len(genuine_files)} genuine files and {len(fake_files)} fake files.")
    
    records = []
    
    # Process genuine
    for p in genuine_files:
        records.append((p, "AUTHENTIC"))
    # Process fake
    for p in fake_files:
        records.append((p, "TAMPERED"))
        
    results = []
    tp = tn = fp = fn = 0
    false_negatives = []
    false_positives = []
    
    for idx, (path, expected) in enumerate(records, 1):
        fn_name = os.path.basename(path)
        print(f"[{idx:02d}/{len(records):02d}] Screening: {fn_name:40s} | Expected: {expected:9s} ... ", end="", flush=True)
        t0 = time.time()
        res = pipeline.screen_document(path, benchmark_mode=False)
        dt = time.time() - t0
        
        verdict = res.get("verdict", "UNKNOWN")
        auth_score = res.get("authenticity_score", 0.0)
        risk_score = res.get("risk_score", 0.0)
        doc_type = res.get("document_type", "unknown")
        
        # Binary prediction:
        # AUTHENTIC -> AUTHENTIC
        # SUSPICIOUS or FLAGGED / TAMPERED -> TAMPERED
        pred = "AUTHENTIC" if verdict == "AUTHENTIC" else "TAMPERED"
        
        if expected == "TAMPERED":
            if pred == "TAMPERED":
                tp += 1
                status = "TP (Correct Tampered)"
            else:
                fn += 1
                status = "FN (FALSE NEGATIVE)"
                false_negatives.append((path, res))
        else:
            if pred == "AUTHENTIC":
                tn += 1
                status = "TN (Correct Genuine)"
            else:
                fp += 1
                status = "FP (False Positive)"
                false_positives.append((path, res))
                
        print(f"-> {verdict:18s} (Auth: {auth_score:.1f}, Risk: {risk_score:.1f}) [{status}] in {dt:.1f}s")
        
        results.append({
            "path": path,
            "filename": fn_name,
            "expected": expected,
            "verdict": verdict,
            "predicted": pred,
            "auth_score": auth_score,
            "risk_score": risk_score,
            "doc_type": doc_type,
            "raw_result": res
        })
        
    total = len(records)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    
    print("\n" + "="*80)
    print("BASELINE AUDIT SUMMARY")
    print("="*80)
    print(f"Total: {total}")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"FPR:       {fpr*100:.2f}%")
    print(f"FNR:       {fnr*100:.2f}%")
    print("="*80)
    
    print(f"\nFALSE NEGATIVES ({len(false_negatives)}):")
    for path, res in false_negatives:
        fn_name = os.path.basename(path)
        print("\n" + "-"*80)
        print(f"FALSE NEGATIVE: {fn_name}")
        print(f"  Path: {path}")
        print(f"  Verdict: {res.get('verdict')} | Auth: {res.get('authenticity_score')} | Risk: {res.get('risk_score')}")
        print(f"  Doc Type: {res.get('document_type')} (SIH: {res.get('sih_document_type')})")
        print(f"  Diagnostic Status: {res.get('diagnostic_status')}")
        
        # Signals & Detectors
        signals = res.get("signals", {})
        print("  Raw Detector Scores:")
        for k, v in signals.items():
            print(f"    - {k}: {v}")
            
        why = res.get("why_this_verdict", {})
        det_exps = why.get("detector_explanations", {})
        print("  Detector Explanations:")
        for k, v in det_exps.items():
            print(f"    - {k}: status={v.get('status')} conf={v.get('confidence')} expl={v.get('explanation')}")
            
        print("  Fusion Rules / Penalties Applied:")
        for r in res.get("penalties_applied", []):
            print(f"    - {r}")
            
        print("  Critical Triggers:")
        for t in res.get("critical_triggers", []):
            print(f"    - {t}")
            
        print("  Cautions:")
        for c in why.get("cautions", []):
            print(f"    - {c}")
            
        ocr = res.get("ocr_extraction", {})
        print(f"  OCR: tokens={ocr.get('token_count')} conf={ocr.get('confidence')} status={ocr.get('status')}")
        tokens = [t.get('text') for t in res.get('raw_ocr_tokens', [])[:15]]
        print(f"  OCR Sample Tokens: {tokens}")
        
        fields = res.get("nlp_validation", {}).get("fields", [])
        print("  NLP Fields:")
        for f in fields:
            print(f"    - {f.get('field')}: val='{f.get('value')}' status={f.get('status')} det={f.get('details')}")
            
        schema = res.get("schema_fields", {})
        if schema:
            print("  SIH Schema Fields:")
            for k, v in schema.items():
                print(f"    - {k}: val='{v.get('value')}' status={v.get('status')} conf={v.get('confidence')}")
                
        # MRZ info if any
        mrz = res.get("mrz_validation", {})
        if mrz and mrz.get("mrz_detected"):
            print(f"  MRZ: detected={mrz.get('mrz_detected')} valid={mrz.get('is_valid')} errors={mrz.get('errors')}")
            
        # Decision logic breakdown
        dec = res.get("forensic_decision", {})
        print(f"  Forensic Decision Hierarchy: {dec.get('hierarchy_summary')}")
        print(f"  Levels Active: {dec.get('levels_active')}")
        print(f"  Families Active: {dec.get('families_active')}")

    # Also save JSON report
    out_dir = os.path.join("reports", "audit")
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "baseline_audit_results.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "fpr": fpr, "fnr": fnr
            },
            "results": [
                {k: v for k, v in r.items() if k != "raw_result"} for r in results
            ]
        }, f, indent=2)
    print(f"\nReport written to: {report_file}")

if __name__ == "__main__":
    main()
