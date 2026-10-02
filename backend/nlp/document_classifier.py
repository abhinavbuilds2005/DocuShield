"""
Document Type Classification Module for SIH26188
Classifies identity documents into one of the 5 supported categories:
1. passport
2. visa
3. national_id
4. driving_license
5. permit

Classification hierarchy:
1. User selection override (if explicitly provided and valid)
2. Structural MRZ signatures (ICAO Doc 9303 P< for passport, V< for visa)
3. Keyword, layout, and regular expression heuristic scoring
4. If classification confidence is low: returns 'unknown' with low confidence
"""

import re
from typing import Dict, Any, Optional, List

DOCUMENT_TYPES = [
    "aadhaar",
    "pan",
    "driving_license",
    "passport",
    "voter_id",
    "national_id",
    "visa",
    "permit"
]

DOCUMENT_TYPE_LABELS = {
    "passport": "Passport",
    "visa": "Visa",
    "national_id": "National Identity Card",
    "driving_license": "Driving Licence",
    "permit": "Permit / Travel Authorization",
    "voter_id": "Voter ID / EPIC Card",
    "pan": "PAN Card",
    "aadhaar": "Aadhaar Card",
    "unknown": "Unknown Document Type"
}

# Rule-based heuristic keywords for each document type
KEYWORD_SIGNATURES = {
    "passport": [
        "passport", "passeport", "republic of", "nationality", "given names",
        "surname", "place of birth", "date of expiry", "issuing authority",
        "type p", "p<", "travel document", "ministere", "foreign affairs",
        "republica", "bundesrepublik"
    ],
    "visa": [
        "visa", "v<", "type of visa", "entries", "entry visa", "valid for",
        "duration of stay", "issuing post", "visa number", "multiple entries",
        "single entry", "consulate general", "embassy", "valid until",
        "schengen", "temporary resident", "visa category"
    ],
    "national_id": [
        "unique identification", "aadhaar", "national identity", "identity card",
        "citizen id", "national id", "permanent account", "income tax", "pan card",
        "government of india", "mera aadhaar", "pehchan patra", "e-aadhaar",
        "republic of", "department of home", "identification card", "social security"
    ],
    "driving_license": [
        "driving licence", "driving license", "driver license", "driver licence",
        "motor vehicles", "transport department", "dl no", "licence no",
        "license no", "union of india", "class of vehicle", "lmv", "mcwg",
        "authorisation to drive", "rto", "transport authority"
    ],
    "permit": [
        "permit", "travel authorization", "work permit", "residence permit",
        "entry permit", "authorisation", "permit no", "stay permit",
        "employment authorization", "travel permit", "alien registration",
        "border pass", "cross-border permit", "special permit"
    ],
    "voter_id": [
        "election commission of india", "election commission", "elector",
        "electoral photo identity card", "epic no", "epic", "voter id",
        "pehchan patra", "assembly constituency", "parliamentary constituency",
        "निर्वाचन आयोग", "भारत निर्वाचन आयोग", "मतदाता पहचान पत्र", "elector's name"
    ],
    "pan": [
        "permanent account number", "income tax department", "govt. of india",
        "father's name", "pan card", "income tax", "permanent account",
        "आयकर विभाग"
    ],
    "aadhaar": [
        "unique identification authority of india", "uidai", "aadhaar",
        "मेरा आधार", "मेरी पहचान", "enrollment no", "help@uidai.gov.in",
        "www.uidai.gov.in", "government of india"
    ]
}


class DocumentClassifier:
    """
    Rule-based and layout-assisted document type classifier.
    NOTE: Transparently implemented via structural signatures and keyword heuristics.
    No fake ML claims are made; confidence scores reflect heuristic matching strength.
    """

    def __init__(self):
        self.signatures = KEYWORD_SIGNATURES

    def classify(
        self,
        full_text: str,
        user_selected_type: Optional[str] = None,
        tokens: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Classifies document type from OCR full text and optional user override.

        Returns:
            Dict containing:
                - document_type: str ("passport", "visa", "national_id", "driving_license", "permit", "unknown")
                - confidence: float (0.0 to 1.0)
                - method: str ("user_selected", "mrz_signature", "keyword_heuristics", "unknown")
                - explanation: str
                - scores: Dict[str, int]
        """
        # 1. User override takes priority if valid
        if user_selected_type and user_selected_type.lower().strip() != "auto":
            raw_type = user_selected_type.lower().strip().replace("-", "_")
            aliases = {
                "dl": "driving_license",
                "driving_licence": "driving_license",
                "driver_license": "driving_license",
                "epic": "voter_id",
                "voter": "voter_id",
                "pancard": "pan",
                "pan_card": "pan",
                "aadhaar_card": "aadhaar",
                "adhaar": "aadhaar",
                "aadhar": "aadhaar",
            }
            clean_user_type = aliases.get(raw_type, raw_type)
            if clean_user_type in DOCUMENT_TYPES:
                return {
                    "document_type": clean_user_type,
                    "confidence": 1.0,
                    "method": "user_selected",
                    "explanation": f"Explicitly specified by user as '{DOCUMENT_TYPE_LABELS.get(clean_user_type, clean_user_type)}'.",
                    "scores": {}
                }

        clean_text = full_text or ""
        lower_text = clean_text.lower()
        scores: Dict[str, int] = {dt: 0 for dt in DOCUMENT_TYPES}

        # 2. Check for ICAO Machine Readable Zone (MRZ) patterns
        # Passport MRZ: Line starting with P< followed by country code
        if re.search(r"\bP<[A-Z0-9<]{5,}", clean_text.upper()) or "P<" in clean_text.upper():
            scores["passport"] += 8

        # Visa MRZ: Line starting with V< followed by country code
        if re.search(r"\bV<[A-Z0-9<]{5,}", clean_text.upper()) or "V<" in clean_text.upper():
            scores["visa"] += 8

        # 3. Check for specific national identifiers
        # Voter ID / EPIC format (3 uppercase letters followed by 7 digits)
        if re.search(r"\b[A-Z]{3}[0-9]{7}\b", clean_text.upper()):
            scores["voter_id"] += 7
            scores["national_id"] += 3

        # 12-digit Indian Aadhaar or UID patterns
        if re.search(r"\b(\d{4}[\s-]?\d{4}[\s-]?\d{4})\b", clean_text):
            scores["aadhaar"] += 6
            scores["national_id"] += 5

        # PAN card format (5 uppercase letters, 4 digits, 1 uppercase letter)
        if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", clean_text.upper()):
            scores["pan"] += 7
            scores["national_id"] += 5

        # DL format pattern
        if re.search(r"\b[A-Z]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{7}\b", clean_text.upper()):
            scores["driving_license"] += 6
        elif re.search(r"\b[A-Z]{2}[0-9]{11,14}\b", clean_text.upper()):
            scores["driving_license"] += 5

        # 4. Keyword frequency scoring
        for doc_type, keywords in self.signatures.items():
            for kw in keywords:
                if kw in lower_text:
                    scores[doc_type] += 1

        # 5. Determine highest scoring document type
        best_type, best_score = max(scores.items(), key=lambda x: x[1])

        # Confidence calibration based on heuristic score
        if best_score >= 6:
            confidence = min(0.95, 0.70 + (best_score * 0.04))
            method = "mrz_signature" if ("<" in clean_text and best_type in ["passport", "visa"]) else "keyword_heuristics"
            return {
                "document_type": best_type,
                "confidence": round(confidence, 2),
                "method": method,
                "explanation": f"Classified as '{DOCUMENT_TYPE_LABELS[best_type]}' based on strong {method} match (score: {best_score}).",
                "scores": scores
            }
        elif best_score >= 2:
            confidence = round(0.50 + (best_score * 0.06), 2)
            return {
                "document_type": best_type,
                "confidence": confidence,
                "method": "keyword_heuristics",
                "explanation": f"Classified as '{DOCUMENT_TYPE_LABELS[best_type]}' with moderate confidence (score: {best_score}).",
                "scores": scores
            }
        else:
            return {
                "document_type": "unknown",
                "confidence": 0.20,
                "method": "unknown",
                "explanation": "Document type could not be determined confidently. Please verify or manually select document type.",
                "scores": scores
            }
