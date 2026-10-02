"""
DocuShield AI — Deterministic PII Masking & Output Boundary Sanitizer

Provides deterministic, field-aware, and idempotent PII redaction for:
- Aadhaar numbers (12-digit format)
- PAN numbers (10-character alphanumeric format)
- Driving Licence numbers (state-coded Indian DL formats)
- Personal names & father's name
- Dates of birth
- Unstructured text (OCR full text, raw QR text, and validation details)

All masking functions are strictly idempotent: running them multiple times on
already-masked strings produces identical output without corrupting text.
Internal forensic analysis runs on unmasked values; this module is applied
at the serialization / response boundary.
"""

import copy
import re
from typing import Any, Dict, Optional


def mask_aadhaar(val: Any) -> Any:
    """
    Masks 12-digit Aadhaar number to standard privacy format: XXXX XXXX 1234.
    Preserves already-masked values.
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s or "XXXX" in s or "••••" in s:
        return s
    
    # 1. 12 consecutive digits or grouped as 4-4-4 (with spaces or dashes)
    return re.sub(
        r"\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b",
        r"XXXX XXXX \3",
        s
    )


def mask_pan(val: Any) -> Any:
    """
    Masks Indian PAN (5 alpha + 4 numeric + 1 alpha) to ABCDE****F format.
    Preserves already-masked values.
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s or "****" in s or "••••" in s:
        return s
    
    # Standard PAN pattern: 5 uppercase letters + 4 digits + 1 uppercase letter
    return re.sub(
        r"\b([A-Z]{5})(\d{4})([A-Z])\b",
        r"\1****\3",
        s
    )


def mask_driving_license(val: Any) -> Any:
    """
    Masks Indian Driving Licence number (e.g. BR0120190012345 or DL-1420110012345).
    Keeps state code and RTO digits, masks remaining serial digits with asterisks.
    Preserves already-masked values.
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s or "*" in s or "••••" in s:
        return s
    
    # Match standard DL format: 2-letter state code + optional dash/space + 2 digits + remaining characters
    def _repl_dl(m: re.Match) -> str:
        prefix = m.group(1)
        suffix = m.group(2)
        # Mask all alphanumeric characters in suffix with asterisks, preserving dashes/spaces
        masked_suffix = re.sub(r"[A-Za-z0-9]", "*", suffix)
        return prefix + masked_suffix

    return re.sub(
        r"\b([A-Z]{2}[-\s]?[0-9]{2})([-\s]?[0-9A-Za-z]{7,13})\b",
        _repl_dl,
        s
    )


def mask_name(val: Any) -> Any:
    """
    Masks personal name (e.g. 'Aarav Sharma' -> 'A**** S*****').
    Only applied to verified personal name fields.
    Preserves empty, unknown, non-person values, and already-masked names.
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s or s.lower() in ("not detected", "unknown", "—", "none", "n/a"):
        return s
    if "*" in s or "••••" in s:
        return s

    tokens = s.split()
    if not tokens:
        return s

    masked_tokens = []
    for tok in tokens:
        if len(tok) <= 1:
            masked_tokens.append(tok)
        else:
            masked_tokens.append(tok[0] + "*" * (len(tok) - 1))
    return " ".join(masked_tokens)


def mask_dob(val: Any) -> Any:
    """
    Masks date of birth (e.g. '15/08/1990' -> '**/**/1990').
    Preserves year for readability and age validation context, redacting day and month.
    Preserves already-masked values.
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s or s.lower() in ("not detected", "unknown", "—", "none", "n/a"):
        return s
    if "*" in s or "XX" in s or "••" in s:
        return s

    # Match DD/MM/YYYY, DD-MM-YYYY, or DD.MM.YYYY
    m = re.match(r"^(\d{1,2})([./-])(\d{1,2})\2(\d{2,4})$", s)
    if m:
        sep = m.group(2)
        year = m.group(4)
        return f"**{sep}**{sep}{year}"


    # Match YYYY/MM/DD
    m_iso = re.match(r"^(\d{4})([./-])(\d{1,2})\2(\d{1,2})$", s)
    if m_iso:
        year = m_iso.group(1)
        sep = m_iso.group(2)
        return f"{year}{sep}**{sep}**"

    return s


def mask_passport(val: Any) -> Any:
    """
    Masks passport number (e.g. 'A1234567' -> 'A*****67').
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s or "*" in s or "••••" in s:
        return s
    if len(s) >= 8 and re.match(r"^[A-Z][0-9]{7,8}$", s):
        return s[0] + "*" * (len(s) - 3) + s[-2:]
    return s


def mask_pii_text(text: Any) -> Any:
    """
    Scans freeform text (such as OCR full text, QR payloads, or explanation notes)
    and replaces Aadhaar, PAN, and DL patterns with masked representations.
    Idempotent.
    """
    if not isinstance(text, str) or not text:
        return text

    # 1. Aadhaar numbers (12-digit grouped or contiguous)
    text = re.sub(
        r"\b(\d{4})[\s-](\d{4})[\s-](\d{4})\b",
        r"XXXX XXXX \3",
        text
    )
    # Match standalone 12 digits (not preceded by word char or dash)
    text = re.sub(
        r"(?<![A-Za-z0-9])(\d{4})(\d{4})(\d{4})(?![A-Za-z0-9])",
        r"XXXX XXXX \3",
        text
    )

    # 2. PAN numbers
    text = re.sub(
        r"\b([A-Z]{5})(\d{4})([A-Z])\b",
        r"\1****\3",
        text
    )

    # 3. Driving Licence numbers
    def _repl_dl_text(m: re.Match) -> str:
        prefix = m.group(1)
        suffix = m.group(2)
        masked_suffix = re.sub(r"[A-Za-z0-9]", "*", suffix)
        return prefix + masked_suffix

    text = re.sub(
        r"\b([A-Z]{2}[-\s]?[0-9]{2})([-\s]?[0-9A-Za-z]{7,13})\b",
        _repl_dl_text,
        text
    )

    return text


def sanitize_screening_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Field-aware public response sanitizer applied at the API output boundary.
    Deep-copies the result to prevent mutating internal structures, and scrubs
    PII strictly from designated leaf fields while preserving all status codes,
    scores, metadata, and structural keys.
    """
    if not isinstance(result, dict):
        return result

    # Work on a shallow/deep hybrid or deep copy to ensure pipeline integrity
    sanitized = copy.deepcopy(result)

    # 1. Sanitize schema_fields
    if "schema_fields" in sanitized and isinstance(sanitized["schema_fields"], dict):
        for field_name, field_data in sanitized["schema_fields"].items():
            if not isinstance(field_data, dict) or "value" not in field_data:
                continue
            raw_val = field_data.get("value")
            name_lower = field_name.lower()

            if name_lower in ("aadhaar_number", "aadhaar", "id_number", "national_id"):
                field_data["value"] = mask_aadhaar(raw_val)
            elif name_lower in ("pan_number", "pan"):
                field_data["value"] = mask_pan(raw_val)
            elif name_lower in ("licence_number", "license_number", "driving_license_number", "dl_number"):
                field_data["value"] = mask_driving_license(raw_val)
            elif name_lower in ("passport_number", "passport"):
                field_data["value"] = mask_passport(raw_val)
            elif name_lower in ("name", "father_name", "mother_name", "spouse_name", "holder_name"):
                field_data["value"] = mask_name(raw_val)
            elif name_lower in ("dob", "date_of_birth", "birth_date"):
                field_data["value"] = mask_dob(raw_val)
            else:
                field_data["value"] = mask_pii_text(raw_val)

    # 2. Sanitize structural_checks.field_details
    if "structural_checks" in sanitized and isinstance(sanitized["structural_checks"], dict):
        field_details = sanitized["structural_checks"].get("field_details")
        if isinstance(field_details, list):
            for item in field_details:
                if not isinstance(item, dict):
                    continue
                val = item.get("value")
                field_label = str(item.get("field", "")).lower()
                if "aadhaar" in field_label or "national id" in field_label:
                    item["value"] = mask_aadhaar(val)
                elif "pan" in field_label:
                    item["value"] = mask_pan(val)
                elif "licence" in field_label or "license" in field_label or "dl" in field_label:
                    item["value"] = mask_driving_license(val)
                elif "name" in field_label:
                    item["value"] = mask_name(val)
                elif "dob" in field_label or "birth" in field_label:
                    item["value"] = mask_dob(val)
                else:
                    item["value"] = mask_pii_text(val)

    # 3. Sanitize signals.nlp_validation
    if "signals" in sanitized and isinstance(sanitized["signals"], dict):
        nlp_sig = sanitized["signals"].get("nlp_validation")
        if isinstance(nlp_sig, dict):
            # OCR extracted text
            if "extracted_full_text" in nlp_sig:
                nlp_sig["extracted_full_text"] = mask_pii_text(nlp_sig["extracted_full_text"])

            # Field checks
            if "field_checks" in nlp_sig and isinstance(nlp_sig["field_checks"], list):
                for chk in nlp_sig["field_checks"]:
                    if not isinstance(chk, dict):
                        continue
                    chk_field = str(chk.get("field", "")).lower()
                    chk_val = chk.get("value")
                    if "aadhaar" in chk_field or "national id" in chk_field:
                        chk["value"] = mask_aadhaar(chk_val)
                    elif "pan" in chk_field:
                        chk["value"] = mask_pan(chk_val)
                    elif "licence" in chk_field or "license" in chk_field or "dl" in chk_field:
                        chk["value"] = mask_driving_license(chk_val)
                    elif "name" in chk_field:
                        chk["value"] = mask_name(chk_val)
                    elif "dob" in chk_field or "birth" in chk_field:
                        chk["value"] = mask_dob(chk_val)
                    else:
                        chk["value"] = mask_pii_text(chk_val)

                    if "details" in chk:
                        chk["details"] = mask_pii_text(chk["details"])

            # Reasons list
            if "reasons" in nlp_sig and isinstance(nlp_sig["reasons"], list):
                nlp_sig["reasons"] = [mask_pii_text(r) for r in nlp_sig["reasons"]]

    # 4. Sanitize qr_analysis and qr_extracted_number
    if "qr_extracted_number" in sanitized and sanitized["qr_extracted_number"]:
        sanitized["qr_extracted_number"] = mask_pii_text(sanitized["qr_extracted_number"])

    if "qr_analysis" in sanitized and isinstance(sanitized["qr_analysis"], dict):
        qr = sanitized["qr_analysis"]
        if "extracted_number" in qr and qr["extracted_number"]:
            qr["extracted_number"] = mask_pii_text(qr["extracted_number"])
        if "raw_text" in qr and qr["raw_text"]:
            qr["raw_text"] = mask_pii_text(qr["raw_text"])

    # 5. Sanitize critical_mismatches and critical_triggers
    if "critical_mismatches" in sanitized and isinstance(sanitized["critical_mismatches"], list):
        sanitized["critical_mismatches"] = [mask_pii_text(m) for m in sanitized["critical_mismatches"]]

    if "critical_triggers" in sanitized and isinstance(sanitized["critical_triggers"], list):
        sanitized["critical_triggers"] = [mask_pii_text(t) for t in sanitized["critical_triggers"]]

    # 6. Sanitize field_crop_extractions and field_crop_ocr_metadata if present
    for crop_key in ("field_crop_extractions", "field_crop_ocr_metadata"):
        if crop_key in sanitized and isinstance(sanitized[crop_key], dict):
            for f_name, c_data in sanitized[crop_key].items():
                if isinstance(c_data, dict) and "text" in c_data:
                    c_data["text"] = mask_pii_text(c_data["text"])

    return sanitized
