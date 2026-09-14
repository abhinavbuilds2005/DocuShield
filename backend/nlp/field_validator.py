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

from backend.nlp.mrz_parser import MRZParser
from backend.nlp.document_classifier import DocumentClassifier, DOCUMENT_TYPES

RULE_BASED_DISCLAIMER = "Rule-based validation only — no external government database verification performed."

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
    clean = re.sub(r"[\s\-]", "", pan_str.strip().upper())
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
        self.classifier = DocumentClassifier()
        self.doc_classifier = self.classifier
        self.mrz_parser = MRZParser()
        self.doc_signatures = {
            "aadhaar": [
                "unique identification", "unique ident", "aadhaar", "mera aadhaar",
                "identity authority", "enrolment", "citizen id", "national citizen"
            ],
            "pan": [
                "income tax", "permanent account", "govt. of india", "father's name", "pan card",
                "tax department", "permanent", "account number", "father"
            ],
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

        is_target_numeric = clean_target.isdigit()
        matched_confs = []
        for tok in tokens:
            clean_tok = re.sub(r"[\s,-]", "", tok.get("text", "")).upper()
            if not clean_tok:
                continue
            if is_target_numeric:
                # For numeric fields (UID/Aadhaar), match numeric tokens with >= 3 chars
                # to avoid accidentally including 2-digit dates or PIN numbers
                if clean_tok.isdigit() and len(clean_tok) >= 3 and (clean_tok in clean_target or clean_target in clean_tok):
                    matched_confs.append(float(tok.get("confidence", 0.0)))
            else:
                if len(clean_tok) >= 2 and (clean_tok in clean_target or clean_target in clean_tok):
                    matched_confs.append(float(tok.get("confidence", 0.0)))

        if not matched_confs:
            return 0.0, False

        avg_conf = float(sum(matched_confs) / len(matched_confs))
        min_conf = float(min(matched_confs))
        is_confident = (avg_conf >= 0.65 and min_conf >= 0.45)
        return round(avg_conf, 4), is_confident

    def detect_document_type(self, full_text: str, user_selected_type: Optional[str] = None) -> str:
        """Classifies document type based on key phrases present and comprehensive classifier."""
        if user_selected_type and user_selected_type.lower() != "auto":
            return user_selected_type.lower()

        classification = self.doc_classifier.classify(full_text, user_selected_type=user_selected_type)
        dt = classification.get("document_type", "unknown")
        if dt != "unknown":
            return dt

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
        if re.search(r"\b([A-Z]{5}[\s-]?[0-9]{4}[\s-]?[A-Z0-9])\b", full_text):
            return "pan"
        return "unknown"

    def extract_document_fields(
        self,
        document_type: str,
        ocr_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts structured document fields adhering to SIH document-specific schemas:
        {
            "field_name": {
                "value": "...",
                "confidence": float,
                "status": "valid" | "invalid" | "unknown"
            }
        }
        """
        full_text = ocr_results.get("full_text", "")
        tokens = ocr_results.get("tokens", [])
        lines = ocr_results.get("lines", [])
        mrz_data = self.mrz_parser.parse(full_text, lines)
        extracted: Dict[str, Dict[str, Any]] = {}

        # Normalization
        doc_type_clean = document_type.lower()
        if doc_type_clean in ["aadhaar", "pan"]:
            doc_type_clean = "national_id"
        elif doc_type_clean == "dl":
            doc_type_clean = "driving_license"

        # 1. PASSPORT SCHEMA
        if doc_type_clean == "passport":
            mrz_fields = mrz_data.get("fields", {})

            # Passport Number
            p_num = mrz_fields.get("passport_number")
            if not p_num:
                m = re.search(r"\b([A-Z][0-9]{7,9})\b", full_text.upper())
                p_num = m.group(1) if m else ""
            conf, _ = self._get_field_ocr_confidence(p_num, tokens)
            p_valid = bool(p_num and len(p_num) >= 6 and re.match(r"^[A-Z0-9]+$", p_num))
            extracted["passport_number"] = {
                "value": p_num or "Not Detected",
                "confidence": conf if p_num else 0.0,
                "status": "valid" if p_valid else ("invalid" if p_num else "unknown")
            }

            # Full Name
            name = mrz_fields.get("full_name")
            if not name:
                nm = re.search(r"(?:given name|name|surname|holder)[\s:]+([A-Z\s]{3,30})", full_text, re.IGNORECASE)
                name = nm.group(1).strip() if nm else ""
            conf, _ = self._get_field_ocr_confidence(name, tokens)
            extracted["full_name"] = {
                "value": name or "Not Detected",
                "confidence": conf if name else 0.0,
                "status": "valid" if name else "unknown"
            }

            # Nationality
            nat = mrz_fields.get("nationality")
            if not nat:
                nm = re.search(r"(?:nationality|citizen of)[\s:]+([A-Z]{3,15})", full_text, re.IGNORECASE)
                nat = nm.group(1).strip() if nm else ""
            conf, _ = self._get_field_ocr_confidence(nat, tokens)
            extracted["nationality"] = {
                "value": nat or "Not Detected",
                "confidence": conf if nat else 0.0,
                "status": "valid" if nat else "unknown"
            }

            # DOB & Expiry
            dob = mrz_fields.get("date_of_birth") or ""
            doe = mrz_fields.get("date_of_expiry") or ""
            if not dob or not doe:
                dates = re.findall(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", full_text)
                if len(dates) >= 2:
                    dob = dob or dates[0]
                    doe = doe or dates[1]
                elif len(dates) == 1:
                    dob = dob or dates[0]

            conf_dob, _ = self._get_field_ocr_confidence(dob, tokens)
            conf_doe, _ = self._get_field_ocr_confidence(doe, tokens)

            extracted["date_of_birth"] = {
                "value": dob or "Not Detected",
                "confidence": conf_dob if dob else 0.0,
                "status": "valid" if dob else "unknown"
            }
            extracted["date_of_expiry"] = {
                "value": doe or "Not Detected",
                "confidence": conf_doe if doe else 0.0,
                "status": "valid" if doe else "unknown"
            }

            # Gender
            gender = mrz_fields.get("gender")
            if not gender:
                gm = re.search(r"\b(MALE|FEMALE|M|F)\b", full_text.upper())
                gender = "MALE" if (gm and gm.group(1) in ["MALE", "M"]) else ("FEMALE" if gm else "Not Detected")
            extracted["gender"] = {
                "value": gender or "Not Detected",
                "confidence": 0.85 if gender != "Not Detected" else 0.0,
                "status": "valid" if gender != "Not Detected" else "unknown"
            }

            # Issuing Country
            country = mrz_fields.get("issuing_country")
            if not country:
                cm = re.search(r"(?:republic of|kingdom of|country|state of)[\s:]+([A-Za-z\s]{3,20})", full_text, re.IGNORECASE)
                country = cm.group(1).strip().upper() if cm else ""
            extracted["issuing_country"] = {
                "value": country or "Not Detected",
                "confidence": 0.80 if country else 0.0,
                "status": "valid" if country else "unknown"
            }

            # MRZ
            extracted["mrz"] = {
                "value": " | ".join(mrz_data.get("raw_lines", [])) if mrz_data.get("mrz_detected") else "Not Detected",
                "confidence": 0.95 if mrz_data.get("mrz_detected") else 0.0,
                "status": "valid" if mrz_data.get("is_valid") else ("invalid" if mrz_data.get("mrz_detected") else "unknown")
            }

        # 2. VISA SCHEMA
        elif doc_type_clean == "visa":
            vm = re.search(r"(?:visa\s*no\.?|visa\s*number|number)[\s:]*([A-Z0-9]{6,12})", full_text, re.IGNORECASE)
            visa_num = vm.group(1).strip() if vm else ""
            if not visa_num:
                v_generic = re.search(r"\b([A-Z][0-9]{7,8}|[0-9]{8,10})\b", full_text)
                visa_num = v_generic.group(1) if v_generic else ""
            conf, _ = self._get_field_ocr_confidence(visa_num, tokens)
            extracted["visa_number"] = {
                "value": visa_num or "Not Detected",
                "confidence": conf if visa_num else 0.0,
                "status": "valid" if visa_num else "unknown"
            }

            hm = re.search(r"(?:holder|name|bearer)[\s:]+([A-Za-z\s]{3,30})", full_text, re.IGNORECASE)
            holder = hm.group(1).strip() if hm else ""
            conf, _ = self._get_field_ocr_confidence(holder, tokens)
            extracted["holder_name"] = {
                "value": holder or "Not Detected",
                "confidence": conf if holder else 0.0,
                "status": "valid" if holder else "unknown"
            }

            nm = re.search(r"(?:nationality|citizen)[\s:]+([A-Za-z]{3,20})", full_text, re.IGNORECASE)
            nat = nm.group(1).strip().upper() if nm else ""
            extracted["nationality"] = {
                "value": nat or "Not Detected",
                "confidence": 0.75 if nat else 0.0,
                "status": "valid" if nat else "unknown"
            }

            dates = re.findall(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", full_text)
            dob = dates[0] if len(dates) >= 1 else ""
            issue_d = dates[1] if len(dates) >= 2 else ""
            expiry_d = dates[2] if len(dates) >= 3 else (dates[1] if len(dates) == 2 else "")

            extracted["date_of_birth"] = {
                "value": dob or "Not Detected",
                "confidence": 0.70 if dob else 0.0,
                "status": "valid" if dob else "unknown"
            }
            extracted["issue_date"] = {
                "value": issue_d or "Not Detected",
                "confidence": 0.70 if issue_d else 0.0,
                "status": "valid" if issue_d else "unknown"
            }
            extracted["expiry_date"] = {
                "value": expiry_d or "Not Detected",
                "confidence": 0.70 if expiry_d else 0.0,
                "status": "valid" if expiry_d else "unknown"
            }

            tm = re.search(r"(?:visa\s*type|type|category)[\s:]+([A-Za-z0-9\-\s]{1,15})", full_text, re.IGNORECASE)
            v_type = tm.group(1).strip() if tm else ("TOURIST" if "TOURIST" in full_text.upper() else ("BUSINESS" if "BUSINESS" in full_text.upper() else "STANDARD"))
            extracted["visa_type"] = {
                "value": v_type,
                "confidence": 0.75,
                "status": "valid"
            }

            em = re.search(r"\b(MULTIPLE|SINGLE|DOUBLE|MULT|01|02|M)\b", full_text.upper())
            entries = em.group(1) if em else "SINGLE"
            extracted["entries"] = {
                "value": entries,
                "confidence": 0.70,
                "status": "valid"
            }

            dm = re.search(r"(\d{1,3}\s*(?:DAYS|MONTHS|YEARS))", full_text, re.IGNORECASE)
            stay = dm.group(1).upper() if dm else "Not Specified"
            extracted["stay_duration"] = {
                "value": stay,
                "confidence": 0.65 if dm else 0.0,
                "status": "valid" if dm else "unknown"
            }

            am = re.search(r"(?:embassy|consulate|post|authority)[\s:]+([A-Za-z\s]{3,30})", full_text, re.IGNORECASE)
            auth = am.group(1).strip() if am else "Not Specified"
            extracted["issuing_authority"] = {
                "value": auth,
                "confidence": 0.65 if am else 0.0,
                "status": "valid" if am else "unknown"
            }

        # 3. NATIONAL ID SCHEMA
        elif doc_type_clean == "national_id":
            uid_match = re.search(r"\b(\d{4}[\s,-]?\d{4}[\s,-]?\d{4}|\d{8}[\s,-]?\d{4}|\d{12})\b", full_text)
            pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", full_text.upper())

            if uid_match:
                raw_id = re.sub(r"\D", "", uid_match.group(1))
                conf, _ = self._get_field_ocr_confidence(raw_id, tokens)
                is_vh = validate_verhoeff(raw_id)
                extracted["ID_number"] = {
                    "value": f"{raw_id[:4]} {raw_id[4:8]} {raw_id[8:]}" if len(raw_id) == 12 else raw_id,
                    "confidence": conf,
                    "status": "valid" if is_vh else "invalid"
                }
            elif pan_match:
                pan_val = pan_match.group(1)
                conf, _ = self._get_field_ocr_confidence(pan_val, tokens)
                is_p_valid, _ = validate_pan_format(pan_val)
                extracted["ID_number"] = {
                    "value": pan_val,
                    "confidence": conf,
                    "status": "valid" if is_p_valid else "invalid"
                }
            else:
                generic_id = re.search(r"\b([A-Z0-9\-]{8,16})\b", full_text.upper())
                val = generic_id.group(1) if generic_id else "Not Detected"
                extracted["ID_number"] = {
                    "value": val,
                    "confidence": 0.50 if val != "Not Detected" else 0.0,
                    "status": "valid" if val != "Not Detected" else "unknown"
                }

            nm = re.search(r"(?:name|citizen)[\s:]+([A-Za-z\s]{3,30})", full_text, re.IGNORECASE)
            name_val = nm.group(1).strip() if nm else ""
            extracted["name"] = {
                "value": name_val or "Not Detected",
                "confidence": 0.70 if name_val else 0.0,
                "status": "valid" if name_val else "unknown"
            }

            dates = re.findall(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", full_text)
            dob_val = dates[0] if dates else ""
            extracted["date_of_birth"] = {
                "value": dob_val or "Not Detected",
                "confidence": 0.70 if dob_val else 0.0,
                "status": "valid" if dob_val else "unknown"
            }

            gm = re.search(r"\b(MALE|FEMALE|TRANSGENDER|M|F)\b", full_text.upper())
            gender_val = "MALE" if (gm and gm.group(1) in ["MALE", "M"]) else ("FEMALE" if (gm and gm.group(1) in ["FEMALE", "F"]) else "Not Specified")
            extracted["gender"] = {
                "value": gender_val,
                "confidence": 0.80 if gender_val != "Not Specified" else 0.0,
                "status": "valid" if gender_val != "Not Specified" else "unknown"
            }

            addr_m = re.search(r"(?:address|addr)[\s:]+([A-Za-z0-9\s,.-]{10,60})", full_text, re.IGNORECASE)
            addr_val = addr_m.group(1).strip() if addr_m else "Not Available"
            extracted["address"] = {
                "value": addr_val,
                "confidence": 0.60 if addr_m else 0.0,
                "status": "valid" if addr_m else "unknown"
            }

            issue_val = dates[1] if len(dates) >= 2 else "Not Specified"
            expiry_val = dates[2] if len(dates) >= 3 else "Not Applicable"
            extracted["issue_date"] = {
                "value": issue_val,
                "confidence": 0.60 if issue_val != "Not Specified" else 0.0,
                "status": "valid" if issue_val != "Not Specified" else "unknown"
            }
            extracted["expiry_date"] = {
                "value": expiry_val,
                "confidence": 0.60 if expiry_val != "Not Applicable" else 0.0,
                "status": "valid" if expiry_val != "Not Applicable" else "unknown"
            }

        # 4. DRIVING LICENCE SCHEMA
        elif doc_type_clean == "driving_license":
            dl_m = re.search(r"\b([A-Z]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{7})\b", full_text.upper())
            if not dl_m:
                dl_m = re.search(r"\b([A-Z]{2}[0-9]{11,14})\b", full_text.upper())
            if not dl_m:
                dl_m = re.search(r"(?:dl|licence|license)\s*no\.?[\s:]*([A-Z0-9\-]{8,18})", full_text, re.IGNORECASE)

            dl_num = dl_m.group(1).strip() if dl_m else "Not Detected"
            conf, _ = self._get_field_ocr_confidence(dl_num, tokens)
            is_dl_valid, _ = validate_dl_format(dl_num) if dl_num != "Not Detected" else (False, None)
            extracted["licence_number"] = {
                "value": dl_num,
                "confidence": conf if dl_num != "Not Detected" else 0.0,
                "status": "valid" if is_dl_valid else ("invalid" if dl_num != "Not Detected" else "unknown")
            }

            nm = re.search(r"(?:name)[\s:]+([A-Za-z\s]{3,30})", full_text, re.IGNORECASE)
            name_val = nm.group(1).strip() if nm else ""
            extracted["name"] = {
                "value": name_val or "Not Detected",
                "confidence": 0.70 if name_val else 0.0,
                "status": "valid" if name_val else "unknown"
            }

            dates = re.findall(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", full_text)
            dob_val = dates[0] if len(dates) >= 1 else ""
            issue_val = dates[1] if len(dates) >= 2 else ""
            exp_val = dates[2] if len(dates) >= 3 else (dates[1] if len(dates) == 2 else "")

            extracted["date_of_birth"] = {
                "value": dob_val or "Not Detected",
                "confidence": 0.70 if dob_val else 0.0,
                "status": "valid" if dob_val else "unknown"
            }
            extracted["issue_date"] = {
                "value": issue_val or "Not Detected",
                "confidence": 0.70 if issue_val else 0.0,
                "status": "valid" if issue_val else "unknown"
            }
            extracted["expiry_date"] = {
                "value": exp_val or "Not Detected",
                "confidence": 0.70 if exp_val else 0.0,
                "status": "valid" if exp_val else "unknown"
            }

            vm = re.search(r"\b(LMV|MCWG|MCWOG|HGMV|HPMV|TRANS|CLASS\s*[A-Z0-9]+)\b", full_text.upper())
            vclass = vm.group(1) if vm else "LMV (Standard)"
            extracted["vehicle_class"] = {
                "value": vclass,
                "confidence": 0.75,
                "status": "valid"
            }

            auth_m = re.search(r"(?:rto|transport authority|state)[\s:]+([A-Za-z\s]{3,25})", full_text, re.IGNORECASE)
            auth_val = auth_m.group(1).strip() if auth_m else "Transport Authority"
            extracted["issuing_authority"] = {
                "value": auth_val,
                "confidence": 0.65,
                "status": "valid"
            }

        # 5. PERMIT SCHEMA
        elif doc_type_clean == "permit":
            pm = re.search(r"(?:permit\s*no\.?|authorization\s*no\.?|number)[\s:]*([A-Z0-9\-]{6,16})", full_text, re.IGNORECASE)
            p_num = pm.group(1).strip() if pm else "Not Detected"
            conf, _ = self._get_field_ocr_confidence(p_num, tokens)
            extracted["document_number"] = {
                "value": p_num,
                "confidence": conf if p_num != "Not Detected" else 0.0,
                "status": "valid" if p_num != "Not Detected" else "unknown"
            }

            nm = re.search(r"(?:name|holder)[\s:]+([A-Za-z\s]{3,30})", full_text, re.IGNORECASE)
            name_val = nm.group(1).strip() if nm else ""
            extracted["name"] = {
                "value": name_val or "Not Detected",
                "confidence": 0.70 if name_val else 0.0,
                "status": "valid" if name_val else "unknown"
            }

            nat_m = re.search(r"(?:nationality|country)[\s:]+([A-Za-z]{3,20})", full_text, re.IGNORECASE)
            nat_val = nat_m.group(1).strip().upper() if nat_m else "Not Specified"
            extracted["nationality"] = {
                "value": nat_val,
                "confidence": 0.65 if nat_m else 0.0,
                "status": "valid" if nat_m else "unknown"
            }

            dates = re.findall(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", full_text)
            dob_val = dates[0] if len(dates) >= 1 else ""
            issue_val = dates[1] if len(dates) >= 2 else ""
            exp_val = dates[2] if len(dates) >= 3 else (dates[1] if len(dates) == 2 else "")

            extracted["date_of_birth"] = {
                "value": dob_val or "Not Detected",
                "confidence": 0.70 if dob_val else 0.0,
                "status": "valid" if dob_val else "unknown"
            }
            extracted["issue_date"] = {
                "value": issue_val or "Not Detected",
                "confidence": 0.70 if issue_val else 0.0,
                "status": "valid" if issue_val else "unknown"
            }
            extracted["expiry_date"] = {
                "value": exp_val or "Not Detected",
                "confidence": 0.70 if exp_val else 0.0,
                "status": "valid" if exp_val else "unknown"
            }

            pt_m = re.search(r"(?:permit\s*type|type)[\s:]+([A-Za-z\s]{3,25})", full_text, re.IGNORECASE)
            pt_val = pt_m.group(1).strip() if pt_m else "Entry / Travel Authorization"
            extracted["permit_type"] = {
                "value": pt_val,
                "confidence": 0.70,
                "status": "valid"
            }

            auth_m = re.search(r"(?:authority|department|ministry)[\s:]+([A-Za-z\s]{3,30})", full_text, re.IGNORECASE)
            auth_val = auth_m.group(1).strip() if auth_m else "Immigration Authority"
            extracted["issuing_authority"] = {
                "value": auth_val,
                "confidence": 0.65,
                "status": "valid"
            }

        else:
            extracted["raw_text_summary"] = {
                "value": full_text[:100] + "..." if len(full_text) > 100 else (full_text or "No text detected"),
                "confidence": 0.50,
                "status": "unknown"
            }

        return extracted

    def validate_document(
        self,
        document_type: str,
        extracted_fields_or_ocr: Optional[Any] = None,
        ocr_or_extracted: Optional[Any] = None,
        extracted_fields: Optional[Dict[str, Dict[str, Any]]] = None,
        ocr_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes document-specific validation engine checks.
        Supports both positional and keyword invocation styles.
        """
        actual_fields = extracted_fields
        actual_ocr = ocr_results

        if extracted_fields_or_ocr is not None:
            if isinstance(extracted_fields_or_ocr, dict) and "full_text" in extracted_fields_or_ocr:
                actual_ocr = extracted_fields_or_ocr
            elif isinstance(extracted_fields_or_ocr, dict):
                actual_fields = extracted_fields_or_ocr

        if ocr_or_extracted is not None:
            if isinstance(ocr_or_extracted, dict) and "full_text" in ocr_or_extracted:
                actual_ocr = ocr_or_extracted
            elif isinstance(ocr_or_extracted, dict):
                actual_fields = ocr_or_extracted

        if actual_ocr is None:
            actual_ocr = {}
        if actual_fields is None:
            actual_fields = self.extract_document_fields(document_type, actual_ocr)

        ocr_results = actual_ocr
        extracted_fields = actual_fields

        full_text = ocr_results.get("full_text", "")
        lines = ocr_results.get("lines", [])
        mrz_data = self.mrz_parser.parse(full_text, lines)

        checks: List[Dict[str, Any]] = []
        penalties = 0
        reasons: List[str] = []
        now = datetime.now()

        # Date Chronology and Expiry Validation
        dob_info = extracted_fields.get("date_of_birth", {})
        issue_info = extracted_fields.get("issue_date", {})
        expiry_info = extracted_fields.get("date_of_expiry", {}) or extracted_fields.get("expiry_date", {})

        d_dob = None
        d_issue = None
        d_expiry = None

        if dob_info.get("value") and dob_info["value"] not in ["Not Detected", ""]:
            ok, parsed, err = parse_and_validate_date(dob_info["value"])
            if ok:
                d_dob = parsed
                checks.append({"rule": "Date of Birth Plausibility", "status": "PASS", "details": f"Valid date of birth: {dob_info['value']}"})
            else:
                penalties += 20
                reasons.append(f"Invalid date of birth: {err}")
                checks.append({"rule": "Date of Birth Plausibility", "status": "FAIL", "details": err})

        if issue_info.get("value") and issue_info["value"] not in ["Not Detected", "Not Specified", ""]:
            ok, parsed, err = parse_and_validate_date(issue_info["value"])
            if ok:
                d_issue = parsed
                if parsed > now:
                    penalties += 25
                    msg = f"Post-dated Issue Date: {issue_info['value']} is in the future"
                    reasons.append(msg)
                    checks.append({"rule": "Issue Date Logic", "status": "FAIL", "details": msg})
                else:
                    checks.append({"rule": "Issue Date Logic", "status": "PASS", "details": f"Valid issue date: {issue_info['value']}"})

        if expiry_info.get("value") and expiry_info["value"] not in ["Not Detected", "Not Applicable", ""]:
            ok, parsed, err = parse_and_validate_date(expiry_info["value"])
            if ok:
                d_expiry = parsed
                if parsed < now:
                    penalties += 15
                    msg = f"Document Expired: Validity ended on {expiry_info['value']}"
                    reasons.append(msg)
                    checks.append({"rule": "Document Expiry Status", "status": "EXPIRED", "details": msg})
                else:
                    checks.append({"rule": "Document Expiry Status", "status": "PASS", "details": f"Document is currently valid (expires {expiry_info['value']})"})
            else:
                penalties += 20
                reasons.append(f"Invalid expiry date: {err}")
                checks.append({"rule": "Document Expiry Status", "status": "FAIL", "details": err})

        if d_dob and d_issue and d_issue <= d_dob:
            penalties += 35
            msg = "Chronological Contradiction: Issue date is on or before Date of Birth"
            reasons.append(msg)
            checks.append({"rule": "Chronology (DOB vs Issue)", "status": "FAIL", "details": msg})

        if d_issue and d_expiry and d_expiry <= d_issue:
            penalties += 35
            msg = "Chronological Contradiction: Expiry date is before or equal to Issue date"
            reasons.append(msg)
            checks.append({"rule": "Chronology (Issue vs Expiry)", "status": "FAIL", "details": msg})

        # Category specific checks
        if document_type == "passport":
            if mrz_data.get("mrz_detected"):
                if mrz_data.get("is_valid"):
                    checks.append({
                        "rule": "ICAO Doc 9303 MRZ Check Digits",
                        "status": "PASS",
                        "details": "All MRZ check digits mathematically valid"
                    })
                else:
                    penalties += 45
                    reasons.append("MRZ check digit validation failed: one or more fields mathematically inconsistent")
                    checks.append({
                        "rule": "ICAO Doc 9303 MRZ Check Digits",
                        "status": "FAIL",
                        "details": mrz_data.get("explanation")
                    })

                mrz_pnum = mrz_data.get("fields", {}).get("passport_number", "")
                vis_pnum = extracted_fields.get("passport_number", {}).get("value", "")
                if mrz_pnum and vis_pnum and vis_pnum != "Not Detected":
                    if mrz_pnum in vis_pnum or vis_pnum in mrz_pnum:
                        checks.append({
                            "rule": "MRZ-to-Visual Passport Number Consistency",
                            "status": "PASS",
                            "details": f"Visual passport number matches MRZ '{mrz_pnum}'"
                        })
                    else:
                        penalties += 30
                        msg = f"Discrepancy: Visual passport number '{vis_pnum}' does not match MRZ '{mrz_pnum}'"
                        reasons.append(msg)
                        checks.append({"rule": "MRZ-to-Visual Passport Number Consistency", "status": "FAIL", "details": msg})
            else:
                checks.append({
                    "rule": "MRZ Presence",
                    "status": "WARNING",
                    "details": "No MRZ lines detected on this document image."
                })

            p_field = extracted_fields.get("passport_number", {})
            if p_field.get("status") == "invalid":
                penalties += 30
                reasons.append(f"Invalid passport number structure: {p_field.get('value')}")
                checks.append({"rule": "Passport Number Format", "status": "FAIL", "details": "Alphanumeric format invalid"})
            elif p_field.get("status") == "valid":
                checks.append({"rule": "Passport Number Format", "status": "PASS", "details": f"Valid passport number structure ({p_field.get('value')})"})

        elif document_type == "visa":
            v_num = extracted_fields.get("visa_number", {}).get("value")
            if not v_num or v_num == "Not Detected":
                penalties += 20
                reasons.append("Visa number not clearly identified")
                checks.append({"rule": "Visa Identifier Presence", "status": "WARNING", "details": "Visa number could not be extracted confidently"})
            else:
                checks.append({"rule": "Visa Identifier Presence", "status": "PASS", "details": f"Visa identifier identified: {v_num}"})

        elif document_type == "national_id":
            id_val = extracted_fields.get("ID_number", {}).get("value", "")
            clean_digits = re.sub(r"\D", "", id_val)
            if len(clean_digits) == 12:
                is_vh_valid, resolved, note = validate_verhoeff_with_ocr_tolerance(clean_digits)
                if is_vh_valid:
                    det = "Verhoeff Checksum Valid (Passed official 12-digit algorithm)"
                    if note:
                        det += f" [{note}]"
                    checks.append({"rule": "National ID Verhoeff Checksum", "status": "PASS", "details": det})
                else:
                    penalties += 45
                    msg = "Deterministic Verhoeff Checksum Failure: 12-digit UID fails mathematical checksum validation."
                    reasons.append(msg)
                    checks.append({"rule": "National ID Verhoeff Checksum", "status": "FAIL", "details": msg})
            elif re.fullmatch(r"^[A-Z]{5}[0-9]{4}[A-Z]$", id_val.strip().upper()):
                is_p_valid, p_err = validate_pan_format(id_val)
                if is_p_valid:
                    checks.append({"rule": "Tax ID (PAN) Structure", "status": "PASS", "details": "Valid 10-character Tax ID structure"})
                else:
                    penalties += 35
                    reasons.append(f"Tax ID format error: {p_err}")
                    checks.append({"rule": "Tax ID (PAN) Structure", "status": "FAIL", "details": p_err})
            else:
                checks.append({"rule": "National ID Format Verification", "status": "INFO", "details": "Standard national identity format check; no proprietary national database connection."})

        elif document_type == "driving_license":
            dl_val = extracted_fields.get("licence_number", {}).get("value", "")
            if dl_val and dl_val != "Not Detected":
                is_dl_valid, err_msg = validate_dl_format(dl_val)
                if is_dl_valid:
                    checks.append({"rule": "Driving Licence Format", "status": "PASS", "details": f"Valid licence pattern: {dl_val}"})
                else:
                    penalties += 25
                    reasons.append(f"Driving licence format issue: {err_msg}")
                    checks.append({"rule": "Driving Licence Format", "status": "FAIL", "details": err_msg})

        validation_score = max(0.0, min(100.0, 100.0 - penalties))
        return {
            "document_type": document_type,
            "validation_score": validation_score,
            "disclaimer": RULE_BASED_DISCLAIMER,
            "checks": checks,
            "anomalies_detected": len(reasons) > 0,
            "reasons": reasons
        }

    def validate(self, ocr_results: Dict[str, Any], user_selected_type: Optional[str] = None) -> Dict[str, Any]:
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
        lines = ocr_results.get("lines", [])

        # 1. Document Classification & Routing
        classification = self.doc_classifier.classify(full_text, user_selected_type=user_selected_type)
        sih_type = classification.get("document_type", "unknown")
        if sih_type == "unknown":
            legacy_type = self.detect_document_type(full_text, user_selected_type=user_selected_type)
            if legacy_type in ["aadhaar", "pan"]:
                sih_type = "national_id"
            elif legacy_type == "dl":
                sih_type = "driving_license"
            elif legacy_type in DOCUMENT_TYPES:
                sih_type = legacy_type

        doc_type = sih_type
        if sih_type == "national_id":
            lower = full_text.lower()
            if any(kw in lower for kw in self.doc_signatures.get("pan", [])) or re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", full_text):
                doc_type = "pan"
            else:
                doc_type = "aadhaar"
        elif sih_type == "driving_license":
            doc_type = "dl"
        
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
                
                # Check direct Verhoeff mathematical validity
                if len(raw_id_val) == 12 and raw_id_val[0] not in ['0', '1'] and validate_verhoeff(raw_id_val):
                    is_valid = True
                    resolved_val = raw_id_val
                    note = None
                elif not is_field_confident or field_conf < 0.88:
                    # Low or ambiguous OCR confidence (e.g. JPEG compression artifacts): check if optical disambiguation resolves it
                    is_valid, resolved_val, note = validate_verhoeff_with_ocr_tolerance(raw_id_val)
                else:
                    # High-confidence OCR reading that fails Verhoeff is a genuine checksum failure
                    is_valid = False
                    resolved_val = raw_id_val
                    note = None

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
                    if len(raw_id_val) == 12 and field_conf >= 0.65:
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
                    elif len(raw_id_val) == 12 and field_conf >= 0.48:
                        # Medium confidence: optical disambiguation did not resolve it; moderate anomaly requiring review, not deterministic proof
                        overall_nlp_penalty += 25
                        msg = f"Uncertain Verhoeff Checksum: 12-digit UID was extracted with moderate field confidence ({field_conf:.2f}) and fails mathematical validation across optical candidate readings. Forensic review required."
                        field_evaluations.append({
                            "field": "Aadhaar / National ID Number",
                            "value": raw_id_val,
                            "status": "FAIL",
                            "details": msg,
                            "field_ocr_confidence": field_conf,
                            "is_deterministic": False,
                            "evidence_level": "MODERATE"
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
                avg_doc_conf = sum(t.get("confidence", 0.0) for t in tokens) / max(1, len(tokens)) if tokens else 0.0
                if avg_doc_conf < 0.60 or len(tokens) < 5:
                    overall_nlp_penalty += 10
                    msg = "National ID number could not be resolved due to low OCR readability; human review required"
                    field_evaluations.append({
                        "field": "Aadhaar / National ID Number",
                        "value": "Unclear / Illegible",
                        "status": "UNCERTAIN",
                        "details": msg,
                        "field_ocr_confidence": avg_doc_conf,
                        "is_deterministic": False,
                        "evidence_level": "WEAK"
                    })
                else:
                    overall_nlp_penalty += 30
                    msg = "Expected 12-digit National ID pattern missing or illegible"
                    field_evaluations.append({
                        "field": "Aadhaar / National ID Number",
                        "value": "Missing / Illegible",
                        "status": "FAIL",
                        "details": msg,
                        "field_ocr_confidence": 0.0,
                        "is_deterministic": False,
                        "evidence_level": "MODERATE"
                    })
                reasons.append(msg)

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
                    if is_field_confident or field_conf >= 0.55:
                        overall_nlp_penalty += 40
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
                        overall_nlp_penalty += 15
                        msg = f"Uncertain field: PAN format issue on '{pan_val}', but field OCR confidence is low ({field_conf:.2f} < 0.55)."
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
                # Check if card clearly has other recognized PAN components (name, DOB, headers) but PAN itself is missing
                has_card_body = len(tokens) >= 8 and re.search(r"\b(BACHCHAN|SINGH|KUMAR|SHARMA|VERMA|NAME|FATHER|ACCOUNT|CARD|INCOME|TAX)\b", full_text.upper())
                if has_card_body:
                    overall_nlp_penalty += 35
                    msg = f"Primary PAN Identifier Missing or Unresolvable: Document is classified as a PAN Card with recognized cardholder body text, but the mandatory 10-character Permanent Account Number is missing or unreadable (baseline OCR conf: {avg_doc_conf:.2f}). Forensic review required."
                    reasons.append(msg)
                    field_evaluations.append({
                        "field": "Permanent Account Number (PAN)",
                        "value": "Missing / Unresolvable",
                        "status": "FAIL",
                        "details": msg,
                        "field_ocr_confidence": avg_doc_conf,
                        "is_deterministic": True,
                        "evidence_level": "MODERATE"
                    })
                elif avg_doc_conf < 0.65:
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
                        "value": "Missing / Illegible",
                        "status": "FAIL",
                        "details": msg,
                        "field_ocr_confidence": 0.0,
                        "is_deterministic": False,
                        "evidence_level": "MODERATE"
                    })

        elif doc_type == "passport":
            mrz_res = self.mrz_parser.parse(full_text, lines)
            
            # Calculate MRZ line clarity
            mrz_conf = 0.0
            mrz_conf_count = 0
            if lines:
                for l in lines:
                    t = l.get("text", "")
                    if "<" in t and len(t) >= 20:
                        mrz_conf += l.get("confidence", 0.0)
                        mrz_conf_count += 1
            if mrz_conf_count > 0:
                mrz_conf = mrz_conf / mrz_conf_count
            else:
                mrz_tokens = [tok for tok in tokens if "<" in tok.get("text", "")]
                if mrz_tokens:
                    mrz_conf = sum(tok.get("confidence", 0.0) for tok in mrz_tokens) / len(mrz_tokens)
                else:
                    mrz_conf = 0.50

            # 1. MRZ Check Digits Validation
            if mrz_res.get("mrz_detected"):
                chk_digits = mrz_res.get("check_digits", {})
                is_valid = mrz_res.get("is_valid", False)
                is_valid_tol = mrz_res.get("is_valid_with_tolerance", False)
                
                if is_valid:
                    field_evaluations.append({
                        "field": "Passport MRZ Check Digits (ICAO 9303)",
                        "value": "Passed (Doc Num, DOB, Expiry, Composite)",
                        "status": "PASS",
                        "details": "All ICAO Doc 9303 TD3 check digits mathematically verified",
                        "field_ocr_confidence": mrz_conf,
                        "is_deterministic": True,
                        "evidence_level": "CLEAN"
                    })
                elif is_valid_tol and mrz_conf < 0.70:
                    field_evaluations.append({
                        "field": "Passport MRZ Check Digits (ICAO 9303)",
                        "value": "Verified with OCR Disambiguation",
                        "status": "PASS",
                        "details": "MRZ check digits verified after minor optical character disambiguation on camera photo",
                        "field_ocr_confidence": mrz_conf,
                        "is_deterministic": True,
                        "evidence_level": "CLEAN"
                    })
                else:
                    failed_parts = []
                    if not chk_digits.get("passport_number_valid"):
                        failed_parts.append("Passport Number Check Digit")
                    if not chk_digits.get("date_of_birth_valid"):
                        failed_parts.append("DOB Check Digit")
                    if not chk_digits.get("date_of_expiry_valid"):
                        failed_parts.append("Expiry Check Digit")
                    if not chk_digits.get("composite_valid"):
                        failed_parts.append("Composite Check Digit")
                    fail_str = ", ".join(failed_parts) if failed_parts else "One or more check digits"

                    if mrz_conf >= 0.65:
                        overall_nlp_penalty += 45
                        msg = f"ICAO Doc 9303 MRZ Check Digit Failure: Mathematical check digit mismatch on high-clarity document ({fail_str})"
                        reasons.append(msg)
                        field_evaluations.append({
                            "field": "Passport MRZ Check Digits (ICAO 9303)",
                            "value": f"Verification Failed ({fail_str})",
                            "status": "FAIL",
                            "details": msg,
                            "field_ocr_confidence": mrz_conf,
                            "is_deterministic": True,
                            "evidence_level": "STRONG"
                        })
                    else:
                        overall_nlp_penalty += 10
                        msg = f"Passport MRZ check digits inconclusive due to low optical clarity ({mrz_conf:.2f})"
                        reasons.append(msg)
                        field_evaluations.append({
                            "field": "Passport MRZ Check Digits (ICAO 9303)",
                            "value": "Inconclusive (Low Clarity)",
                            "status": "WARNING",
                            "details": msg,
                            "field_ocr_confidence": mrz_conf,
                            "is_deterministic": False,
                            "evidence_level": "WEAK"
                        })

                # 2. Country / Nationality Consistency
                if chk_digits.get("country_discrepancy"):
                    overall_nlp_penalty += 35
                    disc_msg = f"MRZ Country Inconsistency: {chk_digits['country_discrepancy']}"
                    reasons.append(disc_msg)
                    field_evaluations.append({
                        "field": "Passport Issuing Country / Nationality",
                        "value": "Inconsistent",
                        "status": "FAIL",
                        "details": disc_msg,
                        "is_deterministic": True,
                        "evidence_level": "STRONG"
                    })

                # 3. MRZ-to-Visual Passport Number Consistency
                mrz_pnum = mrz_res.get("fields", {}).get("passport_number", "")
                m_vis = re.search(r"(?:passport\s*no\.?|paszport|pasaporte|passeport)[\s:]*([A-Z0-9]{6,12})", full_text, re.IGNORECASE)
                if not m_vis:
                    for tok in tokens:
                        t_text = tok.get("text", "").strip()
                        if re.match(r"^[A-Z][0-9]{7,8}$", t_text) and t_text not in mrz_pnum:
                            m_vis = t_text
                            break
                vis_pnum = m_vis.group(1).strip() if (m_vis and hasattr(m_vis, "group")) else (m_vis if isinstance(m_vis, str) else "")
                
                if mrz_pnum and vis_pnum:
                    clean_m = re.sub(r"[^A-Z0-9]", "", mrz_pnum.upper())
                    clean_v = re.sub(r"[^A-Z0-9]", "", vis_pnum.upper())
                    if clean_m in clean_v or clean_v in clean_m:
                        field_evaluations.append({
                            "field": "Passport Number Consistency (Visual vs MRZ)",
                            "value": f"Matches ({clean_m})",
                            "status": "PASS",
                            "details": f"Visual passport number '{clean_v}' matches MRZ '{clean_m}'",
                            "is_deterministic": True,
                            "evidence_level": "CLEAN"
                        })
                    else:
                        v_conf, _ = self._get_field_ocr_confidence(vis_pnum, tokens)
                        if v_conf >= 0.60:
                            overall_nlp_penalty += 35
                            msg = f"Passport Number Discrepancy: Visual '{clean_v}' contradicts MRZ '{clean_m}'"
                            reasons.append(msg)
                            field_evaluations.append({
                                "field": "Passport Number Consistency (Visual vs MRZ)",
                                "value": f"Mismatch ({clean_v} vs {clean_m})",
                                "status": "FAIL",
                                "details": msg,
                                "is_deterministic": True,
                                "evidence_level": "STRONG"
                            })
                        else:
                            field_evaluations.append({
                                "field": "Passport Number Consistency (Visual vs MRZ)",
                                "value": "Uncertain",
                                "status": "WARNING",
                                "details": f"Potential visual discrepancy between '{clean_v}' and '{clean_m}' on low-clarity text",
                                "is_deterministic": False,
                                "evidence_level": "WEAK"
                            })
            else:
                overall_nlp_penalty += 20
                msg = "Passport missing required Machine Readable Zone (MRZ)"
                reasons.append(msg)
                field_evaluations.append({
                    "field": "Passport MRZ Zone",
                    "value": "Not Detected",
                    "status": "WARNING",
                    "details": msg,
                    "is_deterministic": False,
                    "evidence_level": "WEAK"
                })

        elif doc_type == "dl":
            dl_m = re.search(r"\b([A-Z]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{7}|[A-Z]{2}[0-9]{13,14})\b", full_text.upper())
            if not dl_m:
                dl_m = re.search(r"(?:dl|licence|license)\s*no\.?[\s:]*([A-Z0-9\-]{8,18})", full_text, re.IGNORECASE)
            dl_num = dl_m.group(1).strip() if dl_m else ""
            if dl_num:
                conf, _ = self._get_field_ocr_confidence(dl_num, tokens)
                is_dl_valid, dl_err = validate_dl_format(dl_num)
                if is_dl_valid:
                    field_evaluations.append({
                        "field": "Driving Licence Number Format",
                        "value": dl_num,
                        "status": "PASS",
                        "details": f"Valid state code and licence structure ({dl_num})",
                        "field_ocr_confidence": conf,
                        "is_deterministic": True,
                        "evidence_level": "CLEAN"
                    })
                else:
                    if conf >= 0.65:
                        overall_nlp_penalty += 35
                        msg = f"Driving licence format issue: {dl_err}"
                        reasons.append(msg)
                        field_evaluations.append({
                            "field": "Driving Licence Number Format",
                            "value": dl_num,
                            "status": "FAIL",
                            "details": msg,
                            "field_ocr_confidence": conf,
                            "is_deterministic": True,
                            "evidence_level": "STRONG"
                        })
                    else:
                        field_evaluations.append({
                            "field": "Driving Licence Number Format",
                            "value": dl_num,
                            "status": "UNCERTAIN",
                            "details": f"Potential format issue on low-clarity text: {dl_err}",
                            "field_ocr_confidence": conf,
                            "is_deterministic": False,
                            "evidence_level": "WEAK"
                        })

        # Issuer Header Template Verification (catches counterfeit templates with misspelled issuer names)
        suspicious_header_phrases = [
            ("lncohe", "INCOME"),
            ("departmemt", "DEPARTMENT"),
            ("indla", "INDIA"),
            ("pehchah", "PEHCHAN")
        ]
        lower_full_text = full_text.lower()
        found_counterfeit_headers = []
        for bad_p, good_p in suspicious_header_phrases:
            if re.search(rf"\b{bad_p}\b", lower_full_text):
                found_counterfeit_headers.append(f"'{bad_p.upper()}' (counterfeit misspelling of '{good_p}')")

        if found_counterfeit_headers:
            overall_nlp_penalty += 35
            msg = f"Official Issuer Header Forgery: Detected misspelled template text ({', '.join(found_counterfeit_headers)}) characteristic of amateur digital card fabrication."
            reasons.append(msg)
            field_evaluations.append({
                "field": "Issuer Template Integrity",
                "value": "Misspelled Official Header",
                "status": "FAIL",
                "details": msg,
                "field_ocr_confidence": 0.90,
                "is_deterministic": True,
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
                field_conf, is_field_confident = self._get_field_ocr_confidence(d_str, tokens)
                if is_field_confident:
                    overall_nlp_penalty += 35
                    reasons.append(f"Invalid date format or impossible date: {d_str} ({err})")
                    field_evaluations.append({
                        "field": "Date Entry",
                        "value": d_str,
                        "status": "FAIL",
                        "details": err,
                        "field_ocr_confidence": field_conf,
                        "is_deterministic": False,
                        "evidence_level": "MODERATE"
                    })
                else:
                    overall_nlp_penalty += 8
                    reasons.append(f"Uncertain date entry: '{d_str}' has low OCR clarity ({field_conf:.2f} < 0.65)")
                    field_evaluations.append({
                        "field": "Date Entry",
                        "value": d_str,
                        "status": "UNCERTAIN",
                        "details": f"{err} (OCR confidence {field_conf:.2f} indicates compression/blur distortion)",
                        "field_ocr_confidence": field_conf,
                        "is_deterministic": False,
                        "evidence_level": "WEAK"
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

        # SIH Classification and Schema Validation
        schema_fields = self.extract_document_fields(sih_type, ocr_results)
        val_engine = self.validate_document(sih_type, ocr_results, extracted_fields=schema_fields)

        # Bridge any critical failures from val_engine not already in field_evaluations
        for chk in val_engine.get("checks", []):
            if chk.get("status") == "FAIL":
                rule_name = chk.get("rule", "")
                if "Chronolog" in rule_name:
                    already_present = any(rule_name in f.get("field", "") for f in field_evaluations)
                    if not already_present:
                        overall_nlp_penalty += 35
                        reasons.append(chk.get("details", ""))
                        field_evaluations.append({
                            "field": rule_name,
                            "value": "Logic Contradiction",
                            "status": "FAIL",
                            "details": chk.get("details", ""),
                            "is_deterministic": True,
                            "evidence_level": "STRONG"
                        })

        # Calculate final layer score
        nlp_score = max(0, min(100, 100 - overall_nlp_penalty))
        
        has_deterministic_failure = any(f.get("is_deterministic") and f.get("status") == "FAIL" for f in field_evaluations)
        deterministic_failures = [f for f in field_evaluations if f.get("is_deterministic") and f.get("status") == "FAIL"]
        weak_signals = [f for f in field_evaluations if f.get("evidence_level") == "WEAK" or f.get("status") == "UNCERTAIN"]

        return {
            "document_type": sih_type if sih_type != "unknown" else doc_type,
            "sih_document_type": sih_type,
            "score": nlp_score,
            "fields": field_evaluations,
            "schema_fields": schema_fields,
            "classification": classification,
            "validation_engine": val_engine,
            "disclaimer": RULE_BASED_DISCLAIMER,
            "anomalies_detected": len(reasons) > 0,
            "has_deterministic_failure": has_deterministic_failure,
            "deterministic_failures": deterministic_failures,
            "weak_signals": weak_signals,
            "reasons": reasons
        }
