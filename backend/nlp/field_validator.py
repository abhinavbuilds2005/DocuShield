"""
Document Field Validation & NLP Verification Module
Implements:
- Verhoeff checksum algorithm for 12-digit national ID numbers
- Alphanumeric pattern validation for Tax ID / PAN cards (5 letters, 4 digits, 1 letter)
- Driving License pattern validation
- Date chronology & format consistency check
- Document type detection based on key phrase presence
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# --- Verhoeff Checksum Algorithm for 12-digit National ID ---
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff(num_str: str) -> bool:
    """Validates whether a numeric string satisfies the Verhoeff checksum."""
    clean = re.sub(r"\D", "", num_str)
    if len(clean) != 12:
        return False
    c = 0
    for i, digit in enumerate(reversed(clean)):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


def generate_verhoeff_checksum(num_str_11: str) -> str:
    """Generates the 12th Verhoeff checksum digit for an 11-digit string."""
    clean = re.sub(r"\D", "", num_str_11)
    c = 0
    for i, digit in enumerate(reversed(clean)):
        c = VERHOEFF_D[c][VERHOEFF_P[(i + 1) % 8][int(digit)]]
    return str(VERHOEFF_INV[c])


def validate_verhoeff_with_ocr_tolerance(num_str: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validates a 12-digit number against the Verhoeff checksum.
    UIDAI numbers never begin with 0 or 1.
    If direct check passes, returns (True, clean, None).
    If direct check fails, checks whether single OCR optical confusion (8 <-> 6)
    on a body digit resolves to a valid Verhoeff checksum.
    """
    clean = re.sub(r"\D", "", num_str)
    if len(clean) != 12:
        return False, clean, f"Invalid digit length ({len(clean)}, expected 12)"
    if clean[0] in ['0', '1']:
        return False, clean, "Invalid leading digit: national ID never begins with 0 or 1"
    if validate_verhoeff(clean):
        return True, clean, None

    # EasyOCR consistently misreads crimson red '6' as '8' (and vice versa) on card body
    ocr_digit_confusions = {
        '8': ['6'],
        '6': ['8'],
    }
    for i in range(len(clean)):
        ch = clean[i]
        for candidate in ocr_digit_confusions.get(ch, []):
            substituted = clean[:i] + candidate + clean[i+1:]
            if substituted[0] not in ['0', '1'] and validate_verhoeff(substituted):
                return True, substituted, f"Resolved via single-digit OCR optical disambiguation ({ch}->{candidate} at index {i})"

    return False, clean, "Verhoeff Checksum Failed - indicates fabricated or altered digit"


def validate_pan_format(pan_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validates Indian PAN-style format:
    5 uppercase letters, 4 digits, 1 uppercase letter.
    4th character is usually P (Individual), C (Company), H (HUF), A, B, G, J, L, F, T.
    """
    clean = pan_str.strip().upper()
    # Normalize common OCR character confusions on 10th char (trailing letter)
    if len(clean) == 10:
        chars = list(clean)
        ocr_digit_to_letter = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G'}
        if chars[9] in ocr_digit_to_letter:
            chars[9] = ocr_digit_to_letter[chars[9]]
        clean = "".join(chars)

    if not re.fullmatch(r"^[A-Z]{5}[0-9]{4}[A-Z]$", clean):
        return False, "Does not match 10-character alphanumeric structure (ABCDE1234F)"
    
    entity_code = clean[3]
    valid_entities = {"P", "C", "H", "F", "A", "T", "B", "L", "J", "G"}
    if entity_code not in valid_entities:
        return False, f"Invalid 4th character '{entity_code}' - expected standard tax entity code"
    
    return True, None


def validate_dl_format(dl_str: str) -> Tuple[bool, Optional[str]]:
    """Validates Driving License format (e.g. DL-1420110012345 or similar)."""
    clean = re.sub(r"[\s\-]", "", dl_str.strip().upper())
    if len(clean) < 10 or len(clean) > 16:
        return False, f"Invalid DL length ({len(clean)} characters, expected 13-16)"
    if not re.match(r"^[A-Z]{2}[0-9]{11,14}$", clean):
        return False, "DL must start with 2-letter state code followed by digits"
    return True, None


def parse_and_validate_date(date_str: str) -> Tuple[bool, Optional[datetime], Optional[str]]:
    """Tries parsing various standard date formats and validates plausible ranges."""
    clean = date_str.strip()
    # Normalize separators
    clean = re.sub(r"[.\-]", "/", clean)
    
    formats = ["%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%y"]
    parsed = None
    for fmt in formats:
        try:
            parsed = datetime.strptime(clean, fmt)
            break
        except ValueError:
            continue
            
    if not parsed:
        return False, None, f"Unrecognized date structure: '{date_str}'"
        
    # Check logical boundary (e.g., year between 1900 and 2050)
    current_year = datetime.now().year
    if parsed.year < 1920 or parsed.year > current_year + 30:
        return False, parsed, f"Date year {parsed.year} outside plausible range (1920 - {current_year+30})"
        
    return True, parsed, None


class DocumentFieldValidator:
    """Orchestrates comprehensive NLP and field-level validation for extracted OCR text."""
    
    def __init__(self):
        self.doc_signatures = {
            "aadhaar": [
                "unique identification", "unique ident", "aadhaar", "mera aadhaar",
                "identity authority", "enrolment", "citizen id", "national citizen", "mockland"
            ],
            "pan": ["income tax", "permanent account", "govt. of india", "father's name", "pan card"],
            "dl": ["driving licence", "motor vehicles", "union of india", "licence no", "transport department"]
        }

    def _get_field_ocr_confidence(self, field_digits_or_text: str, tokens: List[Dict[str, Any]]) -> Tuple[float, bool]:
        """
        Computes field-level OCR confidence by locating the token(s) containing the field text.
        Returns (avg_confidence, is_confident).
        """
        clean_target = re.sub(r"[\s,-]", "", field_digits_or_text).upper()
        if not clean_target:
            return 0.0, False

        matched_confs = []
        for tok in tokens:
            clean_tok = re.sub(r"[\s,-]", "", tok.get("text", "")).upper()
            if not clean_tok:
                continue
            # A token belongs to this field if it is a substantial chunk (>= 2 chars)
            # that is contained within clean_target, or vice versa
            if len(clean_tok) >= 2 and (clean_tok in clean_target or clean_target in clean_tok):
                matched_confs.append(float(tok.get("confidence", 0.0)))

        if not matched_confs:
            return 0.0, False

        avg_conf = float(sum(matched_confs) / len(matched_confs))
        min_conf = float(min(matched_confs))
        is_confident = (avg_conf >= 0.65 and min_conf >= 0.45)
        return round(avg_conf, 4), is_confident

    def detect_document_type(self, full_text: str) -> str:
        """Classifies document type based on key phrases present."""
        lower_text = full_text.lower()
        scores = {}
        for doc_type, keywords in self.doc_signatures.items():
            scores[doc_type] = sum(1 for kw in keywords if kw in lower_text)
            
        best_type, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score > 0:
            return best_type
        # Fallback inspection by regex
        if re.search(r"\b(\d{4}[\s-]?\d{4}[\s-]?\d{4}|\d{8}[\s-]?\d{4}|\d{12})\b", full_text):
            return "aadhaar"
        if re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z0-9])\b", full_text):
            return "pan"
        return "unknown"

    def validate(self, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes structured OCR result and performs:
        1. Document classification
        2. Expected field completeness check
        3. Checksum & Pattern validation
        4. Date logic validation
        5. Token confidence consistency check
        """
        full_text = ocr_results.get("full_text", "")
        tokens = ocr_results.get("tokens", [])
        doc_type = self.detect_document_type(full_text)
        
        field_evaluations: List[Dict[str, Any]] = []
        overall_nlp_penalty = 0
        reasons: List[str] = []

        # 1. Look for ID numbers
        if doc_type == "aadhaar":
            # Search for 12-digit number (e.g. 1234 5678 9012 or 12345678 9012 or 123456789012)
            id_match = re.search(r"\b(\d{4}[\s,-]?\d{4}[\s,-]?\d{4}|\d{8}[\s,-]?\d{4}|\d{12})\b", full_text)
            if id_match:
                raw_id_val = re.sub(r"\D", "", id_match.group(1))
                field_conf, is_field_confident = self._get_field_ocr_confidence(raw_id_val, tokens)
                is_valid, resolved_val, note = validate_verhoeff_with_ocr_tolerance(raw_id_val)
                if is_valid:
                    detail_str = "Verhoeff Checksum Valid (Passed official 12-digit algorithm)"
                    if note:
                        detail_str += f" [{note}]"
                    field_evaluations.append({
                        "field": "Aadhaar / National ID Number",
                        "value": f"{resolved_val[:4]} {resolved_val[4:8]} {resolved_val[8:]}",
                        "status": "PASS",
                        "details": detail_str,
                        "field_ocr_confidence": field_conf,
                        "is_deterministic": True,
                        "evidence_level": "CLEAN"
                    })
                else:
                    if len(raw_id_val) == 12 and is_field_confident:
                        overall_nlp_penalty += 45
                        msg = f"Deterministic Verhoeff Checksum Failure: 12-digit UID was extracted with high field confidence ({field_conf:.2f}), but fails mathematical validation even with optical character confusion tolerance."
                        field_evaluations.append({
                            "field": "Aadhaar / National ID Number",
                            "value": raw_id_val,
                            "status": "FAIL",
                            "details": msg,
                            "field_ocr_confidence": field_conf,
                            "is_deterministic": True,
                            "evidence_level": "STRONG"
                        })
                        reasons.append(msg)
                    else:
                        overall_nlp_penalty += 15
                        msg = f"Unverifiable critical field: Verhoeff checksum could not be verified on UID '{raw_id_val}' due to low OCR confidence ({field_conf:.2f}). Requires human review."
                        field_evaluations.append({
                            "field": "Aadhaar / National ID Number",
                            "value": raw_id_val,
                            "status": "UNCERTAIN",
                            "details": msg,
                            "field_ocr_confidence": field_conf,
                            "is_deterministic": False,
                            "is_unverifiable_due_to_ocr": True,
                            "evidence_level": "WEAK"
                        })
                        reasons.append(msg)
            else:
                overall_nlp_penalty += 30
                msg = "Expected 12-digit National ID pattern missing or illegible"
                reasons.append(msg)
                field_evaluations.append({
                    "field": "Aadhaar / National ID Number",
                    "value": "Missing / Illegible",
                    "status": "FAIL",
                    "details": msg,
                    "field_ocr_confidence": 0.0,
                    "is_deterministic": False,
                    "evidence_level": "MODERATE"
                })

        elif doc_type == "pan":
            pan_match = re.search(r"\b([A-Z]{5}[\s-]?[0-9]{4}[\s-]?[A-Z0-9])\b", full_text.upper())
            if pan_match:
                raw_pan = pan_match.group(1)
                pan_val = re.sub(r"[\s-]", "", raw_pan)
                field_conf, is_field_confident = self._get_field_ocr_confidence(pan_val, tokens)
                is_valid, err_msg = validate_pan_format(pan_val)
                if is_valid:
                    field_evaluations.append({
                        "field": "Permanent Account Number (PAN)",
                        "value": pan_val,
                        "status": "PASS",
                        "details": "Valid 10-character Tax ID structure (Entity code verified)",
                        "field_ocr_confidence": field_conf,
                        "is_deterministic": True,
                        "evidence_level": "CLEAN"
                    })
                else:
                    if is_field_confident:
                        overall_nlp_penalty += 45
                        msg = f"Deterministic Tax ID Format Failure: {err_msg} (field confidence: {field_conf:.2f})"
                        field_evaluations.append({
                            "field": "Permanent Account Number (PAN)",
                            "value": pan_val,
                            "status": "FAIL",
                            "details": msg,
                            "field_ocr_confidence": field_conf,
                            "is_deterministic": True,
                            "evidence_level": "STRONG"
                        })
                        reasons.append(msg)
                    else:
                        overall_nlp_penalty += 10
                        msg = f"Uncertain field: PAN format issue on '{pan_val}', but field OCR confidence is low ({field_conf:.2f} < 0.65)."
                        field_evaluations.append({
                            "field": "Permanent Account Number (PAN)",
                            "value": pan_val,
                            "status": "UNCERTAIN",
                            "details": msg,
                            "field_ocr_confidence": field_conf,
                            "is_deterministic": False,
                            "evidence_level": "WEAK"
                        })
                        reasons.append(msg)
            else:
                avg_doc_conf = sum(t.get("confidence", 0.0) for t in tokens) / max(1, len(tokens)) if tokens else 0.0
                if avg_doc_conf < 0.65:
                    overall_nlp_penalty += 10
                    msg = f"Permanent Account Number could not be confidently resolved due to degraded OCR clarity (baseline: {avg_doc_conf:.2f})"
                    reasons.append(msg)
                    field_evaluations.append({
                        "field": "Permanent Account Number (PAN)",
                        "value": "Unclear / Low Resolution",
                        "status": "UNCERTAIN",
                        "details": msg,
                        "field_ocr_confidence": avg_doc_conf,
                        "is_deterministic": False,
                        "evidence_level": "WEAK"
                    })
                else:
                    overall_nlp_penalty += 35
                    msg = "Expected 10-digit PAN alphanumeric identifier not found or regex pattern invalid"
                    reasons.append(msg)
                    field_evaluations.append({
                        "field": "Permanent Account Number (PAN)",
                        "value": "Missing / Malformed",
                        "status": "FAIL",
                        "details": msg,
                        "field_ocr_confidence": 0.0,
                        "is_deterministic": False,
                        "evidence_level": "MODERATE"
                    })

        elif doc_type == "dl":
            dl_match = re.search(r"\b([A-Z]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{7})\b", full_text.upper())
            if not dl_match:
                dl_match = re.search(r"\b([A-Z]{2}[0-9]{11,14})\b", full_text.upper())
                
            if dl_match:
                dl_val = dl_match.group(1)
                is_valid, err_msg = validate_dl_format(dl_val)
                if is_valid:
                    field_evaluations.append({
                        "field": "Driving Licence Number",
                        "value": dl_val,
                        "status": "PASS",
                        "details": "Standard state code & series pattern valid"
                    })
                else:
                    overall_nlp_penalty += 40
                    field_evaluations.append({
                        "field": "Driving Licence Number",
                        "value": dl_val,
                        "status": "FAIL",
                        "details": err_msg
                    })
                    reasons.append(err_msg)

        # 2. Date checks (DOB, Issue Date, Expiry)
        date_matches = re.findall(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", full_text)
        valid_dates = []
        for d_str in date_matches:
            ok, parsed, err = parse_and_validate_date(d_str)
            if ok:
                valid_dates.append((d_str, parsed))
            else:
                overall_nlp_penalty += 45
                reasons.append(f"Invalid date format or impossible date: {d_str} ({err})")
                field_evaluations.append({
                    "field": "Date Entry",
                    "value": d_str,
                    "status": "FAIL",
                    "details": err
                })

        if valid_dates:
            field_evaluations.append({
                "field": "Detected Dates",
                "value": ", ".join([d[0] for d in valid_dates]),
                "status": "PASS",
                "details": f"Parsed {len(valid_dates)} chronological date(s) successfully"
            })

            # Check chronological consistency between Issue and Expiry dates
            issue_m = re.search(r"Issue\s*(?:Date)?\s*[:\-]?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", full_text, re.IGNORECASE)
            expiry_m = re.search(r"Valid\s*(?:Till)?\s*[:\-]?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", full_text, re.IGNORECASE)
            if issue_m and expiry_m:
                ok1, d_issue, _ = parse_and_validate_date(issue_m.group(1))
                ok2, d_expiry, _ = parse_and_validate_date(expiry_m.group(1))
                if ok1 and ok2 and d_expiry <= d_issue:
                    overall_nlp_penalty += 50
                    msg = f"Chronological Contradiction: Expiry date ({expiry_m.group(1)}) is prior to Issue date ({issue_m.group(1)})"
                    reasons.append(msg)
                    field_evaluations.append({
                        "field": "Date Chronology",
                        "value": f"{issue_m.group(1)} -> {expiry_m.group(1)}",
                        "status": "FAIL",
                        "details": msg
                    })

        # 3. Check token OCR confidence consistency
        # Spliced text often exhibits an anomalous sudden drop or deviation in OCR confidence
        confidences = [t.get("confidence", 1.0) for t in tokens if "confidence" in t]
        conf_anomaly_detected = False
        if len(confidences) >= 5:
            avg_conf = sum(confidences) / len(confidences)
            outliers = [t for t in tokens if t.get("confidence", 1.0) < (avg_conf - 0.45)]
            if len(outliers) > 0:
                conf_anomaly_detected = True
                overall_nlp_penalty += 5
                outlier_words = ", ".join([f"'{o.get('text', '')}' ({o.get('confidence', 0):.2f})" for o in outliers[:3]])
                msg = f"Minor OCR confidence disparity on {len(outliers)} word(s): {outlier_words} contrast with baseline ({avg_conf:.2f})"
                reasons.append(msg)
                field_evaluations.append({
                    "field": "OCR Confidence Consistency",
                    "value": f"Avg: {avg_conf*100:.1f}%",
                    "status": "WARNING",
                    "details": msg,
                    "is_deterministic": False,
                    "evidence_level": "WEAK"
                })

        # Calculate final layer score
        nlp_score = max(0, min(100, 100 - overall_nlp_penalty))
        
        has_deterministic_failure = any(f.get("is_deterministic") and f.get("status") == "FAIL" for f in field_evaluations)
        deterministic_failures = [f for f in field_evaluations if f.get("is_deterministic") and f.get("status") == "FAIL"]
        weak_signals = [f for f in field_evaluations if f.get("evidence_level") == "WEAK" or f.get("status") == "UNCERTAIN"]

        return {
            "document_type": doc_type,
            "score": nlp_score,
            "fields": field_evaluations,
            "anomalies_detected": len(reasons) > 0,
            "has_deterministic_failure": has_deterministic_failure,
            "deterministic_failures": deterministic_failures,
            "weak_signals": weak_signals,
            "reasons": reasons
        }
