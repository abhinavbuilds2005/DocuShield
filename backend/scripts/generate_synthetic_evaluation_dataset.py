"""
DocuShield AI — Phase 3.3 Synthetic Benchmark Generation Engine
Generates 500 privacy-safe, fully fictional, deterministic synthetic evaluation samples
across 5 document classes (Aadhaar, PAN, Driving Licence, Passport, Voter ID).

STRICT SAFETY CONSTRAINTS:
- 100% fictional personae and identifiers. No real citizen data.
- Never scrape from web or use real identity cards.
- Visible disclaimer watermark on every sample:
  [SYNTHETIC TEST SAMPLE — NOT A REAL GOVERNMENT DOCUMENT]
- Zero production code or model modifications.
"""

import os
import sys
import json
import math
import random
import hashlib
import hmac
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# Generator Configuration & Pinned Metadata
# ---------------------------------------------------------------------------
GENERATOR_VERSION = "1.0.0"
MASTER_SEED = 20260918
WATERMARK_TEXT = "[SYNTHETIC TEST SAMPLE — NOT A REAL GOVERNMENT DOCUMENT]"

HMAC_SECRET = os.environ.get("DOCUSHIELD_EVAL_KEY", "fictional_synthetic_eval_key_2026").encode("utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "evaluation_synthetic"

# Windows TrueType Fonts with fallback
WIN_FONTS = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
FONT_SANS = str(WIN_FONTS / "arial.ttf") if (WIN_FONTS / "arial.ttf").exists() else None
FONT_SERIF = str(WIN_FONTS / "calibri.ttf") if (WIN_FONTS / "calibri.ttf").exists() else None
FONT_MONO = str(WIN_FONTS / "consola.ttf") if (WIN_FONTS / "consola.ttf").exists() else None

# ---------------------------------------------------------------------------
# Fictional Persona Lexicon
# ---------------------------------------------------------------------------
FIRST_NAMES_MALE = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Muhammad", "Sai",
    "Arnav", "Ayaan", "Krishna", "Ishaan", "Shaurya", "Atharv", "Advik", "Pranav",
    "Advaith", "Aaryav", "Dhruv", "Kabir", "Ritvik", "Darsh", "Kian", "Samar",
    "Rohan", "Dev", "Vikram", "Karan", "Siddharth", "Rahul", "Nikhil", "Amit"
]

FIRST_NAMES_FEMALE = [
    "Diya", "Saanvi", "Aanya", "Aadhya", "Aaradhya", "Ananya", "Pari", "Anika",
    "Navya", "Angel", "Ira", "Myra", "Sara", "Riya", "Prisha", "Ishi",
    "Advika", "Vanya", "Siya", "Avani", "Kavya", "Meera", "Pooja", "Sunita",
    "Neha", "Shreya", "Sneha", "Tanvi", "Rhea", "Priya", "Anjali", "Swati"
]

SURNAMES = [
    "Sharma", "Verma", "Patel", "Gupta", "Singh", "Kumar", "Iyer", "Nair",
    "Reddy", "Rao", "Mukherjee", "Banerjee", "Chatterjee", "Bose", "Das", "Dutta",
    "Joshi", "Kulkarni", "Deshmukh", "Patil", "Chavan", "Pawar", "Shinde", "Bhat",
    "Menon", "Pillai", "Kurian", "Choudhury", "Mehta", "Shah", "Kapoor", "Malhotra"
]

INDIAN_STATES_DL = [
    ("DL", "Delhi"), ("MH", "Maharashtra"), ("KA", "Karnataka"),
    ("TN", "Tamil Nadu"), ("UP", "Uttar Pradesh"), ("HR", "Haryana"),
    ("GJ", "Gujarat"), ("RJ", "Rajasthan"), ("WB", "West Bengal"), ("KL", "Kerala")
]

# ---------------------------------------------------------------------------
# Algorithmic Identifier Generators (Fictional)
# ---------------------------------------------------------------------------
# Verhoeff Tables for Aadhaar Generation
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

def compute_verhoeff_checksum(number_str: str) -> int:
    c = 0
    reversed_digits = [int(x) for x in reversed(number_str)]
    for i, digit in enumerate(reversed_digits):
        c = VERHOEFF_D[c][VERHOEFF_P[(i + 1) % 8][digit]]
    return VERHOEFF_INV[c]

def generate_fictional_aadhaar(rng: random.Random, valid: bool = True) -> str:
    # 12 digits: first 11 random (first digit in 2..9), 12th is Verhoeff check digit
    lead = str(rng.randint(2, 9))
    body = "".join(str(rng.randint(0, 9)) for _ in range(10))
    first11 = lead + body
    check_digit = compute_verhoeff_checksum(first11)
    if not valid:
        check_digit = (check_digit + rng.randint(1, 9)) % 10
    num = first11 + str(check_digit)
    return f"{num[:4]} {num[4:8]} {num[8:]}"

def generate_fictional_pan(rng: random.Random, valid: bool = True, surname_char: str = "S") -> str:
    # 5 uppercase letters, 4 digits, 1 letter
    # 4th char: P (individual), C (company), etc.
    # 5th char: surname initial
    chars1_3 = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3))
    entity = "P" if valid else rng.choice("0123XYZ")
    surname_letter = surname_char.upper() if valid and surname_char.isalpha() else rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    digits = f"{rng.randint(1000, 9999)}"
    last_char = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    return f"{chars1_3}{entity}{surname_letter}{digits}{last_char}"

def generate_fictional_dl(rng: random.Random, valid: bool = True) -> str:
    # Sarathi-4 standard: 2-letter state code + 2-digit RTO + 4-digit year + 7-digit sequence
    state_code = rng.choice(INDIAN_STATES_DL)[0] if valid else "ZZ"
    rto = f"{rng.randint(1, 99):02d}"
    year = rng.randint(2000, 2024)
    seq = f"{rng.randint(1, 9999999):07d}"
    sep = rng.choice(["-", " ", ""])
    return f"{state_code}{sep}{rto}{year}{seq}"

def generate_fictional_passport(rng: random.Random, valid: bool = True) -> Tuple[str, str, str]:
    # Passport Number: 1 letter + 7 digits (e.g. Z1234567)
    letter = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") if valid else "0"
    digits = f"{rng.randint(1000000, 9999999)}"
    p_num = f"{letter}{digits}"
    # Fictional TD3 MRZ line 1 & line 2
    # Line 1: P<INDLASTNAME<<FIRSTNAME<<<<<<<<<<<<<<<<<<
    # Line 2: P_NUM + check + IND + DOB + check + SEX + EXP + check + ...
    return p_num, "P<IND", "MRZ"

def generate_fictional_epic(rng: random.Random, valid: bool = True) -> str:
    # Voter ID / EPIC: 3 letters + 7 digits (e.g. ABC1234567)
    prefix = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3)) if valid else "12A"
    num = f"{rng.randint(1000000, 9999999):07d}"
    return f"{prefix}{num}"

def compute_hmac_sha256(text: str) -> str:
    normalized = "".join(c for c in str(text).upper() if c.isalnum())
    return hmac.new(HMAC_SECRET, normalized.encode("utf-8"), hashlib.sha256).hexdigest()

# ---------------------------------------------------------------------------
# Graphical Card Rendering Functions
# ---------------------------------------------------------------------------
def get_font(font_path: Optional[str], size: int) -> ImageFont.ImageFont:
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_mock_avatar(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], seed: int):
    """Draws a simple synthetic stylized portrait silhouette inside the photo box."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rng = random.Random(seed)
    
    # Background fill
    bg_color = (rng.randint(210, 235), rng.randint(220, 240), rng.randint(230, 250))
    draw.rectangle(box, fill=bg_color, outline=(120, 140, 160), width=2)
    
    # Head circle
    head_r = int(w * 0.22)
    head_cx = x1 + w // 2
    head_cy = y1 + int(h * 0.38)
    avatar_color = (rng.randint(60, 90), rng.randint(80, 110), rng.randint(120, 150))
    draw.ellipse([head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r], fill=avatar_color)
    
    # Shoulders arc
    shoulder_w = int(w * 0.70)
    draw.pieslice([head_cx - shoulder_w // 2, head_cy + head_r - 5, head_cx + shoulder_w // 2, y2 - 5],
                  start=180, end=360, fill=avatar_color)

def draw_mock_qr(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], seed: int):
    """Draws a synthetic QR code pattern."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    rng = random.Random(seed)
    draw.rectangle(box, fill=(255, 255, 255), outline=(50, 50, 50), width=1)
    grid_n = 12
    step = w // grid_n
    for r in range(grid_n):
        for c in range(grid_n):
            if (r < 3 and c < 3) or (r < 3 and c >= grid_n - 3) or (r >= grid_n - 3 and c < 3):
                draw.rectangle([x1 + c * step, y1 + r * step, x1 + (c + 1) * step, y1 + (r + 1) * step], fill=(20, 20, 20))
            elif rng.random() > 0.55:
                draw.rectangle([x1 + c * step, y1 + r * step, x1 + (c + 1) * step, y1 + (r + 1) * step], fill=(20, 20, 20))

def apply_watermark(img: Image.Image) -> Image.Image:
    """Applies the mandatory perimeter synthetic watermark."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = get_font(FONT_SANS, 14)
    # Bottom watermark bar
    draw.rectangle([0, h - 26, w, h], fill=(240, 240, 245))
    draw.line([0, h - 26, w, h - 26], fill=(180, 50, 50), width=1)
    draw.text((15, h - 21), WATERMARK_TEXT, fill=(180, 40, 40), font=font)
    return img

# ---------------------------------------------------------------------------
# Document Renderer Modules
# ---------------------------------------------------------------------------
def render_synthetic_aadhaar(rng: random.Random, persona: Dict[str, Any], is_valid: bool = True, missing_field: Optional[str] = None) -> Tuple[Image.Image, Dict[str, Any]]:
    w, h = 1000, 630
    img = Image.new("RGB", (w, h), color=(252, 252, 254))
    draw = ImageDraw.Draw(img)
    
    # Outer Card Border & Header Strip
    draw.rectangle([10, 10, w - 10, h - 10], outline=(180, 190, 205), width=2)
    draw.rectangle([12, 12, w - 12, 85], fill=(255, 140, 0)) # Saffron top bar
    draw.rectangle([12, 85, w - 12, 105], fill=(255, 255, 255))
    draw.rectangle([12, 105, w - 12, 125], fill=(34, 139, 34)) # Green bottom strip
    
    f_header = get_font(FONT_SANS, 22)
    f_sub = get_font(FONT_SANS, 15)
    f_body = get_font(FONT_SANS, 18)
    f_bold = get_font(FONT_SANS, 20)
    f_id = get_font(FONT_SANS, 26)
    
    draw.text((180, 30), "UNIQUE IDENTIFICATION AUTHORITY OF INDIA", fill=(255, 255, 255), font=f_header)
    draw.text((320, 88), "Government of India / भारत सरकार", fill=(30, 30, 30), font=f_sub)
    
    # Emblem Box
    emblem_box = [40, 25, 140, 115]
    draw.rectangle(emblem_box, fill=(245, 245, 245), outline=(150, 150, 150), width=1)
    draw.text((55, 60), "EMBLEM", fill=(100, 100, 100), font=f_sub)
    
    fields = {}
    
    # Photo
    photo_box = [50, 160, 240, 400]
    if missing_field != "photo":
        draw_mock_avatar(draw, photo_box, rng.randint(1, 999999))
        fields["photo"] = {
            "bbox_xyxy": photo_box,
            "class_id": 4,
            "synthetic_ground_truth": "[PORTRAIT_IMAGE]",
            "is_mandatory": True
        }
        
    # QR Code
    qr_box = [760, 170, 950, 360]
    if missing_field != "qr_code":
        draw_mock_qr(draw, qr_box, rng.randint(1, 999999))
        fields["qr_code"] = {
            "bbox_xyxy": qr_box,
            "class_id": 5,
            "synthetic_ground_truth": "[SECURE_QR_CODE]",
            "is_mandatory": False
        }
        
    # Text Details: Name, DOB, Gender
    name_str = f"{persona['first_name']} {persona['surname']}".upper()
    dob_str = persona["dob"]
    gender_str = persona["gender"].upper()
    
    # Name
    name_box = [270, 175, 720, 215]
    draw.text((270, 180), name_str, fill=(20, 20, 20), font=f_bold)
    fields["name"] = {
        "bbox_xyxy": name_box,
        "class_id": 1,
        "synthetic_ground_truth": name_str,
        "is_mandatory": True
    }
    
    # DOB
    dob_box = [270, 235, 600, 275]
    draw.text((270, 240), f"DOB: {dob_str}", fill=(40, 40, 40), font=f_body)
    fields["dob"] = {
        "bbox_xyxy": dob_box,
        "class_id": 2,
        "synthetic_ground_truth": dob_str,
        "is_mandatory": True
    }
    
    # Gender
    gender_box = [270, 290, 480, 330]
    draw.text((270, 295), f"{gender_str} / पुरुष", fill=(40, 40, 40), font=f_body)
    fields["gender"] = {
        "bbox_xyxy": gender_box,
        "class_id": 3,
        "synthetic_ground_truth": gender_str,
        "is_mandatory": True
    }
    
    # 12-digit Aadhaar UID
    uid_str = generate_fictional_aadhaar(rng, valid=is_valid)
    if missing_field != "id_number":
        id_box = [280, 460, 720, 520]
        draw.rectangle([id_box[0] - 15, id_box[1] - 8, id_box[2] + 15, id_box[3] + 8], fill=(255, 255, 255), outline=(210, 210, 210))
        draw.text((295, 468), uid_str, fill=(20, 20, 30), font=f_id)
        fields["aadhaar_number"] = {
            "bbox_xyxy": id_box,
            "class_id": 0,
            "synthetic_ground_truth": uid_str,
            "is_mandatory": True,
            "is_valid_verhoeff": is_valid
        }
        
    fields["emblem_header"] = {
        "bbox_xyxy": [180, 25, 900, 95],
        "class_id": 6,
        "synthetic_ground_truth": "UNIQUE IDENTIFICATION AUTHORITY OF INDIA",
        "is_mandatory": False
    }
    
    apply_watermark(img)
    return img, fields

def render_synthetic_pan(rng: random.Random, persona: Dict[str, Any], is_valid: bool = True, missing_field: Optional[str] = None) -> Tuple[Image.Image, Dict[str, Any]]:
    w, h = 980, 620
    img = Image.new("RGB", (w, h), color=(242, 248, 253))
    draw = ImageDraw.Draw(img)
    
    # Card Border & Header
    draw.rectangle([10, 10, w - 10, h - 10], outline=(70, 110, 160), width=2)
    draw.rectangle([12, 12, w - 12, 85], fill=(35, 65, 110))
    
    f_head = get_font(FONT_SANS, 21)
    f_sub = get_font(FONT_SANS, 14)
    f_label = get_font(FONT_SANS, 13)
    f_val = get_font(FONT_SANS, 19)
    f_pan = get_font(FONT_SANS, 27)
    
    draw.text((35, 24), "INCOME TAX DEPARTMENT", fill=(255, 255, 255), font=f_head)
    draw.text((35, 54), "GOVT. OF INDIA", fill=(210, 225, 245), font=f_sub)
    draw.text((640, 35), "Permanent Account Number Card", fill=(255, 255, 255), font=f_sub)
    
    fields = {}
    
    # Photo Box
    photo_box = [45, 120, 230, 350]
    if missing_field != "photo":
        draw_mock_avatar(draw, photo_box, rng.randint(1, 999999))
        fields["photo"] = {
            "bbox_xyxy": photo_box,
            "class_id": 4,
            "synthetic_ground_truth": "[PORTRAIT_IMAGE]",
            "is_mandatory": True
        }
        
    name_str = f"{persona['first_name']} {persona['surname']}".upper()
    father_str = f"{persona['father_first_name']} {persona['surname']}".upper()
    dob_str = persona["dob"]
    pan_str = generate_fictional_pan(rng, valid=is_valid, surname_char=persona["surname"][0])
    
    # Name
    draw.text((270, 120), "Name / नाम", fill=(100, 100, 110), font=f_label)
    name_box = [270, 140, 750, 180]
    draw.text((270, 145), name_str, fill=(20, 20, 20), font=f_val)
    fields["name"] = {
        "bbox_xyxy": name_box,
        "class_id": 1,
        "synthetic_ground_truth": name_str,
        "is_mandatory": True
    }
    
    # Father's Name
    draw.text((270, 195), "Father's Name / पिता का नाम", fill=(100, 100, 110), font=f_label)
    father_box = [270, 215, 750, 255]
    draw.text((270, 220), father_str, fill=(20, 20, 20), font=f_val)
    fields["father_name"] = {
        "bbox_xyxy": father_box,
        "class_id": 2,
        "synthetic_ground_truth": father_str,
        "is_mandatory": True
    }
    
    # Date of Birth
    draw.text((270, 270), "Date of Birth / जन्म की तारीख", fill=(100, 100, 110), font=f_label)
    dob_box = [270, 290, 520, 330]
    draw.text((270, 295), dob_str, fill=(20, 20, 20), font=f_val)
    fields["dob"] = {
        "bbox_xyxy": dob_box,
        "class_id": 3,
        "synthetic_ground_truth": dob_str,
        "is_mandatory": True
    }
    
    # Permanent Account Number
    draw.text((270, 360), "Permanent Account Number / स्थायी खाता संख्या", fill=(100, 100, 110), font=f_label)
    if missing_field != "id_number":
        pan_box = [270, 385, 680, 440]
        draw.text((270, 390), pan_str, fill=(15, 30, 70), font=f_pan)
        fields["pan_number"] = {
            "bbox_xyxy": pan_box,
            "class_id": 0,
            "synthetic_ground_truth": pan_str,
            "is_mandatory": True,
            "is_valid_format": is_valid
        }
        
    apply_watermark(img)
    return img, fields

def render_synthetic_dl(rng: random.Random, persona: Dict[str, Any], is_valid: bool = True, missing_field: Optional[str] = None) -> Tuple[Image.Image, Dict[str, Any]]:
    w, h = 1000, 630
    img = Image.new("RGB", (w, h), color=(253, 253, 248))
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([10, 10, w - 10, h - 10], outline=(150, 150, 120), width=2)
    draw.rectangle([12, 12, w - 12, 75], fill=(50, 80, 60))
    
    f_head = get_font(FONT_SANS, 20)
    f_sub = get_font(FONT_SANS, 14)
    f_label = get_font(FONT_SANS, 12)
    f_val = get_font(FONT_SANS, 17)
    f_dl = get_font(FONT_SANS, 23)
    
    draw.text((30, 25), "UNION OF INDIA — DRIVING LICENCE", fill=(255, 255, 255), font=f_head)
    draw.text((700, 28), "TRANSPORT DEPARTMENT", fill=(220, 240, 220), font=f_sub)
    
    fields = {}
    photo_box = [45, 110, 220, 330]
    if missing_field != "photo":
        draw_mock_avatar(draw, photo_box, rng.randint(1, 999999))
        fields["photo"] = {
            "bbox_xyxy": photo_box,
            "class_id": 5,
            "synthetic_ground_truth": "[PORTRAIT_IMAGE]",
            "is_mandatory": True
        }
        
    dl_str = generate_fictional_dl(rng, valid=is_valid)
    name_str = f"{persona['first_name']} {persona['surname']}".upper()
    dob_str = persona["dob"]
    issue_year = rng.randint(2012, 2022)
    issue_str = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{issue_year}"
    exp_year = issue_year + 20
    exp_str = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{exp_year}"
    
    # DL Number
    draw.text((260, 100), "Licence No / अनुज्ञप्ति संख्या:", fill=(80, 80, 80), font=f_label)
    if missing_field != "id_number":
        dl_box = [260, 120, 720, 165]
        draw.text((260, 125), dl_str, fill=(20, 30, 80), font=f_dl)
        fields["licence_number"] = {
            "bbox_xyxy": dl_box,
            "class_id": 0,
            "synthetic_ground_truth": dl_str,
            "is_mandatory": True,
            "is_valid_format": is_valid
        }
        
    # Name
    draw.text((260, 180), "Name / नाम:", fill=(80, 80, 80), font=f_label)
    name_box = [260, 200, 700, 235]
    draw.text((260, 202), name_str, fill=(20, 20, 20), font=f_val)
    fields["name"] = {
        "bbox_xyxy": name_box,
        "class_id": 1,
        "synthetic_ground_truth": name_str,
        "is_mandatory": True
    }
    
    # DOB
    draw.text((260, 250), "DOB / जन्म तिथि:", fill=(80, 80, 80), font=f_label)
    dob_box = [260, 270, 480, 305]
    draw.text((260, 272), dob_str, fill=(20, 20, 20), font=f_val)
    fields["dob"] = {
        "bbox_xyxy": dob_box,
        "class_id": 2,
        "synthetic_ground_truth": dob_str,
        "is_mandatory": True
    }
    
    # Issue Date
    draw.text((520, 250), "Issue Date:", fill=(80, 80, 80), font=f_label)
    iss_box = [520, 270, 720, 305]
    draw.text((520, 272), issue_str, fill=(20, 20, 20), font=f_val)
    fields["issue_date"] = {
        "bbox_xyxy": iss_box,
        "class_id": 3,
        "synthetic_ground_truth": issue_str,
        "is_mandatory": True
    }
    
    # Expiry Date
    draw.text((750, 250), "Valid Till / समाप्ति:", fill=(80, 80, 80), font=f_label)
    exp_box = [750, 270, 960, 305]
    draw.text((750, 272), exp_str, fill=(20, 20, 20), font=f_val)
    fields["expiry_date"] = {
        "bbox_xyxy": exp_box,
        "class_id": 4,
        "synthetic_ground_truth": exp_str,
        "is_mandatory": True
    }
    
    apply_watermark(img)
    return img, fields

def render_synthetic_passport(rng: random.Random, persona: Dict[str, Any], is_valid: bool = True, missing_field: Optional[str] = None) -> Tuple[Image.Image, Dict[str, Any]]:
    w, h = 1000, 680
    img = Image.new("RGB", (w, h), color=(254, 253, 249))
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([10, 10, w - 10, h - 10], outline=(120, 130, 150), width=2)
    draw.rectangle([12, 12, w - 12, 60], fill=(20, 35, 60))
    
    f_head = get_font(FONT_SANS, 20)
    f_sub = get_font(FONT_SANS, 13)
    f_label = get_font(FONT_SANS, 11)
    f_val = get_font(FONT_SANS, 16)
    f_mrz = get_font(FONT_MONO, 17)
    
    draw.text((30, 20), "REPUBLIC OF INDIA / भारत गणराज्य", fill=(255, 255, 255), font=f_head)
    draw.text((800, 22), "PASSPORT / पासपोर्ट", fill=(220, 225, 235), font=f_sub)
    
    fields = {}
    photo_box = [40, 90, 220, 320]
    if missing_field != "photo":
        draw_mock_avatar(draw, photo_box, rng.randint(1, 999999))
        fields["photo"] = {
            "bbox_xyxy": photo_box,
            "class_id": 0,
            "synthetic_ground_truth": "[PORTRAIT_IMAGE]",
            "is_mandatory": True
        }
        
    p_num, _, _ = generate_fictional_passport(rng, valid=is_valid)
    surname_str = persona["surname"].upper()
    given_str = persona["first_name"].upper()
    dob_str = persona["dob"]
    sex_str = persona["gender"][0].upper()
    exp_year = rng.randint(2028, 2034)
    exp_str = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{exp_year}"
    
    # Passport Number
    draw.text((680, 80), "Passport No / पासपोर्ट सं.:", fill=(80, 80, 80), font=f_label)
    p_box = [680, 100, 950, 135]
    draw.text((680, 102), p_num, fill=(20, 20, 80), font=f_val)
    fields["passport_number"] = {
        "bbox_xyxy": p_box,
        "class_id": 1,
        "synthetic_ground_truth": p_num,
        "is_mandatory": True,
        "is_valid_format": is_valid
    }
    
    # Surname
    draw.text((250, 90), "Surname / उपनाम:", fill=(80, 80, 80), font=f_label)
    sn_box = [250, 110, 650, 145]
    draw.text((250, 112), surname_str, fill=(20, 20, 20), font=f_val)
    fields["surname"] = {
        "bbox_xyxy": sn_box,
        "class_id": 2,
        "synthetic_ground_truth": surname_str,
        "is_mandatory": True
    }
    
    # Given Names
    draw.text((250, 150), "Given Name(s) / दिया गया नाम:", fill=(80, 80, 80), font=f_label)
    gn_box = [250, 170, 650, 205]
    draw.text((250, 172), given_str, fill=(20, 20, 20), font=f_val)
    fields["given_names"] = {
        "bbox_xyxy": gn_box,
        "class_id": 3,
        "synthetic_ground_truth": given_str,
        "is_mandatory": True
    }
    
    # DOB & Sex
    draw.text((250, 210), "Date of Birth / जन्म तिथि:", fill=(80, 80, 80), font=f_label)
    dob_box = [250, 230, 440, 265]
    draw.text((250, 232), dob_str, fill=(20, 20, 20), font=f_val)
    fields["dob"] = {
        "bbox_xyxy": dob_box,
        "class_id": 4,
        "synthetic_ground_truth": dob_str,
        "is_mandatory": True
    }
    
    # Expiry Date
    draw.text((480, 210), "Date of Expiry / समाप्ति तिथि:", fill=(80, 80, 80), font=f_label)
    exp_box = [480, 230, 680, 265]
    draw.text((480, 232), exp_str, fill=(20, 20, 20), font=f_val)
    fields["expiry_date"] = {
        "bbox_xyxy": exp_box,
        "class_id": 5,
        "synthetic_ground_truth": exp_str,
        "is_mandatory": True
    }
    
    # ICAO Doc 9303 TD3 2-line MRZ Zone
    dob_yy = dob_str.split("/")[-1][2:]
    dob_mm = dob_str.split("/")[1]
    dob_dd = dob_str.split("/")[0]
    exp_yy = str(exp_year)[2:]
    exp_mm = exp_str.split("/")[1]
    exp_dd = exp_str.split("/")[0]
    
    mrz1 = f"P<IND{surname_str}<<{given_str}"
    mrz1 = (mrz1 + "<" * 44)[:44]
    mrz2 = f"{p_num}4IND{dob_yy}{dob_mm}{dob_dd}8{sex_str}{exp_yy}{exp_mm}{exp_dd}4<<<<<<<4"
    mrz2 = (mrz2 + "<" * 44)[:44]
    
    mrz_box = [40, 490, 960, 580]
    draw.rectangle([mrz_box[0] - 10, mrz_box[1] - 8, mrz_box[2] + 10, mrz_box[3] + 8], fill=(255, 255, 255), outline=(220, 220, 220))
    draw.text((55, 500), mrz1, fill=(10, 10, 10), font=f_mrz)
    draw.text((55, 535), mrz2, fill=(10, 10, 10), font=f_mrz)
    
    fields["mrz"] = {
        "bbox_xyxy": mrz_box,
        "class_id": 6,
        "synthetic_ground_truth": f"{mrz1}\n{mrz2}",
        "is_mandatory": True
    }
    
    apply_watermark(img)
    return img, fields

def render_synthetic_voter_id(rng: random.Random, persona: Dict[str, Any], is_valid: bool = True, missing_field: Optional[str] = None) -> Tuple[Image.Image, Dict[str, Any]]:
    w, h = 950, 620
    img = Image.new("RGB", (w, h), color=(250, 252, 250))
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([10, 10, w - 10, h - 10], outline=(80, 130, 90), width=2)
    draw.rectangle([12, 12, w - 12, 75], fill=(30, 90, 50))
    
    f_head = get_font(FONT_SANS, 21)
    f_sub = get_font(FONT_SANS, 14)
    f_label = get_font(FONT_SANS, 12)
    f_val = get_font(FONT_SANS, 18)
    f_epic = get_font(FONT_SANS, 25)
    
    draw.text((30, 22), "ELECTION COMMISSION OF INDIA", fill=(255, 255, 255), font=f_head)
    draw.text((30, 52), "भारत निर्वाचन आयोग — ELECTOR PHOTO IDENTITY CARD", fill=(210, 245, 220), font=f_sub)
    
    fields = {}
    photo_box = [40, 110, 220, 340]
    if missing_field != "photo":
        draw_mock_avatar(draw, photo_box, rng.randint(1, 999999))
        fields["photo"] = {
            "bbox_xyxy": photo_box,
            "class_id": 4,
            "synthetic_ground_truth": "[PORTRAIT_IMAGE]",
            "is_mandatory": True
        }
        
    epic_str = generate_fictional_epic(rng, valid=is_valid)
    name_str = f"{persona['first_name']} {persona['surname']}".upper()
    relation_str = f"{persona['father_first_name']} {persona['surname']}".upper()
    gender_str = persona["gender"].upper()
    dob_str = persona["dob"]
    
    # EPIC Number
    draw.text((260, 100), "E.P.I.C. No. / पहचान पत्र क्र.:", fill=(80, 80, 80), font=f_label)
    if missing_field != "id_number":
        epic_box = [260, 120, 680, 165]
        draw.text((260, 125), epic_str, fill=(20, 30, 90), font=f_epic)
        fields["epic_number"] = {
            "bbox_xyxy": epic_box,
            "class_id": 0,
            "synthetic_ground_truth": epic_str,
            "is_mandatory": True,
            "is_valid_format": is_valid
        }
        
    # Elector Name
    draw.text((260, 180), "Elector's Name / निर्वाचक का नाम:", fill=(80, 80, 80), font=f_label)
    name_box = [260, 200, 720, 235]
    draw.text((260, 202), name_str, fill=(20, 20, 20), font=f_val)
    fields["elector_name"] = {
        "bbox_xyxy": name_box,
        "class_id": 1,
        "synthetic_ground_truth": name_str,
        "is_mandatory": True
    }
    
    # Father's/Husband's Name
    draw.text((260, 250), "Father's Name / पिता का नाम:", fill=(80, 80, 80), font=f_label)
    rel_box = [260, 270, 720, 305]
    draw.text((260, 272), relation_str, fill=(20, 20, 20), font=f_val)
    fields["relation_name"] = {
        "bbox_xyxy": rel_box,
        "class_id": 2,
        "synthetic_ground_truth": relation_str,
        "is_mandatory": True
    }
    
    # Sex & DOB
    draw.text((260, 320), "Sex / लिंग:", fill=(80, 80, 80), font=f_label)
    draw.text((260, 340), gender_str, fill=(20, 20, 20), font=f_val)
    draw.text((450, 320), "Date of Birth / जन्म तिथि:", fill=(80, 80, 80), font=f_label)
    draw.text((450, 340), dob_str, fill=(20, 20, 20), font=f_val)
    fields["gender"] = {
        "bbox_xyxy": [260, 335, 400, 370],
        "class_id": 3,
        "synthetic_ground_truth": gender_str,
        "is_mandatory": True
    }
    
    apply_watermark(img)
    return img, fields

# ---------------------------------------------------------------------------
# Degradation & Realism Simulation Pipeline
# ---------------------------------------------------------------------------
def apply_mobile_camera_simulation(img: Image.Image, rng: random.Random) -> Tuple[Image.Image, str]:
    """Simulates mobile camera capture: perspective keystoning, slight rotation, lighting gradient."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    
    # 1. Perspective warp (keystone)
    dx = rng.randint(15, 45)
    dy = rng.randint(10, 30)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[dx, dy], [w - dx, 0], [0, h - dy], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(arr, matrix, (w, h), borderValue=(235, 235, 235))
    
    # 2. Slight rotation (-4 to +4 deg)
    angle = rng.uniform(-4.0, 4.0)
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(warped, rot_mat, (w, h), borderValue=(235, 235, 235))
    
    # 3. Ambient lighting gradient
    gradient = np.tile(np.linspace(0.85, 1.10, w), (h, 1))
    gradient = np.stack([gradient]*3, axis=-1)
    lit = np.clip(rotated * gradient, 0, 255).astype(np.uint8)
    
    return Image.fromarray(lit), "mobile_camera_perspective_and_shadow"

def apply_compression_simulation(img: Image.Image, rng: random.Random) -> Tuple[Image.Image, str]:
    """Simulates WhatsApp/MMS compressed transmission (Q=45 to Q=65)."""
    q = rng.randint(45, 65)
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, enc = cv2.imencode(".jpg", arr, encode_param)
    dec = cv2.imdecode(enc, 1)
    return Image.fromarray(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)), f"jpeg_q{q}"

def apply_degraded_physical_scan(img: Image.Image, rng: random.Random) -> Tuple[Image.Image, str]:
    """Simulates degraded physical card: slight Gaussian blur and downscaled blur."""
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    ksize = rng.choice([3, 5])
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), 1.2)
    # Downscale and upscale to induce low-resolution pixelation
    h, w = arr.shape[:2]
    small = cv2.resize(blurred, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)), f"gaussian_blur_k{ksize}_and_resample"

# ---------------------------------------------------------------------------
# Main Generation Orchestrator
# ---------------------------------------------------------------------------
def generate_synthetic_benchmark(output_dir: Path, master_seed: int = MASTER_SEED):
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    metadata_dir = output_dir / "metadata"
    
    splits = ["dev", "val", "test"]
    for s in splits:
        (images_dir / s).mkdir(parents=True, exist_ok=True)
        (annotations_dir / s).mkdir(parents=True, exist_ok=True)
        (metadata_dir / s).mkdir(parents=True, exist_ok=True)
        
    doc_classes = [
        ("aadhaar", render_synthetic_aadhaar),
        ("pan", render_synthetic_pan),
        ("driving_license", render_synthetic_dl),
        ("passport", render_synthetic_passport),
        ("voter_id", render_synthetic_voter_id)
    ]
    
    sample_index = 0
    generated_records = []
    
    # 500 total samples: 100 per doc class
    for doc_type, render_func in doc_classes:
        for idx in range(100):
            sample_index += 1
            sample_seed = master_seed + sample_index * 1337
            rng = random.Random(sample_seed)
            
            # Split assignment: 0..59 -> dev (60), 60..69 -> val (10), 70..99 -> test (30)
            if idx < 60:
                split = "dev"
            elif idx < 70:
                split = "val"
            else:
                split = "test"
                
            sample_id = f"eval_{split}_{doc_type}_{idx + 1:04d}"
            lineage_id = f"{doc_type}_lineage_{idx + 1:04d}"
            
            # Persona generation
            is_male = rng.random() > 0.5
            fname = rng.choice(FIRST_NAMES_MALE if is_male else FIRST_NAMES_FEMALE)
            father_fname = rng.choice(FIRST_NAMES_MALE)
            sname = rng.choice(SURNAMES)
            dob_year = rng.randint(1965, 2004)
            dob_str = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{dob_year}"
            
            persona = {
                "first_name": fname,
                "father_first_name": father_fname,
                "surname": sname,
                "gender": "Male" if is_male else "Female",
                "dob": dob_str
            }
            
            # Acquisition & Degradation Category Assignment
            # Distribution per 100 samples:
            # 0..29: Pristine (30)
            # 30..54: Mobile camera (25)
            # 55..74: Compressed (20)
            # 75..89: Degraded physical (15)
            # 90..99: Negative control / Edge cases (10)
            is_valid = True
            missing_field = None
            is_neg = False
            anomaly_type = None
            
            if idx < 30:
                cat = "pristine_flatbed"
                deg_profile = "none"
            elif idx < 55:
                cat = "mobile_camera"
                deg_profile = "mobile_perspective_and_shadow"
            elif idx < 75:
                cat = "compressed_transmission"
                deg_profile = "jpeg_compression_artifacts"
            elif idx < 90:
                cat = "degraded_physical"
                deg_profile = "blur_and_resample"
            else:
                cat = "negative_control"
                is_neg = True
                sub_idx = idx - 90
                if sub_idx < 3:
                    is_valid = False # Checksum / format failure
                    anomaly_type = "invalid_checksum_or_syntax"
                elif sub_idx < 6:
                    missing_field = rng.choice(["photo", "id_number"])
                    anomaly_type = f"missing_mandatory_{missing_field}"
                elif sub_idx < 8:
                    anomaly_type = "typography_splicing_anomaly"
                else:
                    anomaly_type = "layout_confusion_edge_case"
                deg_profile = f"adversarial_{anomaly_type}"
                
            # Render Base Document
            card_img, field_annotations = render_func(rng, persona, is_valid=is_valid, missing_field=missing_field)
            
            # Apply Degradation
            if cat == "mobile_camera":
                card_img, deg_profile = apply_mobile_camera_simulation(card_img, rng)
            elif cat == "compressed_transmission":
                card_img, deg_profile = apply_compression_simulation(card_img, rng)
            elif cat == "degraded_physical":
                card_img, deg_profile = apply_degraded_physical_scan(card_img, rng)
                
            # Compute Dimensions
            w, h = card_img.size
            
            # Save Image
            img_rel_path = f"images/{split}/{sample_id}.jpg"
            img_abs_path = images_dir / split / f"{sample_id}.jpg"
            card_img.save(img_abs_path, format="JPEG", quality=90)
            
            # Save YOLO txt annotations
            txt_lines = []
            metadata_fields = {}
            for fname_key, fval in field_annotations.items():
                box = fval["bbox_xyxy"]
                # Convert to normalized YOLO coordinates
                bx_center = ((box[0] + box[2]) / 2.0) / w
                by_center = ((box[1] + box[3]) / 2.0) / h
                bw = (box[2] - box[0]) / w
                bh = (box[3] - box[1]) / h
                cid = fval["class_id"]
                txt_lines.append(f"{cid} {bx_center:.6f} {by_center:.6f} {bw:.6f} {bh:.6f}")
                
                raw_ref = fval["synthetic_ground_truth"]
                metadata_fields[fname_key] = {
                    "field_name": fname_key,
                    "class_id": cid,
                    "bbox_xyxy": box,
                    "bbox_yolo": [round(bx_center, 6), round(by_center, 6), round(bw, 6), round(bh, 6)],
                    "synthetic_ground_truth": raw_ref,
                    "ground_truth_hmac_sha256": compute_hmac_sha256(raw_ref),
                    "is_mandatory": fval.get("is_mandatory", True)
                }
                
            txt_abs_path = annotations_dir / split / f"{sample_id}.txt"
            with open(txt_abs_path, "w", encoding="utf-8") as fp:
                fp.write("\n".join(txt_lines) + "\n")
                
            # Build and save companion .eval.json metadata record
            eval_record = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "sample_id": sample_id,
                "lineage_id": lineage_id,
                "document_type": doc_type,
                "synthetic": True,
                "locked_test": (split == "test"),
                "dataset_split": split,
                "generator_version": GENERATOR_VERSION,
                "random_seed": sample_seed,
                "provenance_record": "DocuShield AI Synthetic Evaluation Generator v1.0",
                "acquisition_category": cat,
                "degradation_profile": deg_profile,
                "image_dimensions": [w, h],
                "image_file": img_rel_path,
                "watermark_present": True,
                "watermark_text": WATERMARK_TEXT,
                "negative_control": {
                    "is_negative_control": is_neg,
                    "anomaly_type": anomaly_type
                },
                "environment_fingerprint": {
                    "python_version": platform.python_version(),
                    "pil_version": Image.__version__,
                    "cv2_version": cv2.__version__,
                    "font_regular": FONT_SANS or "default"
                },
                "fields": metadata_fields
            }
            
            json_abs_path = metadata_dir / split / f"{sample_id}.eval.json"
            with open(json_abs_path, "w", encoding="utf-8") as fp:
                json.dump(eval_record, fp, indent=2)
                
            generated_records.append(eval_record)
            
    print(f"[+] Generation Complete: {len(generated_records)} synthetic samples created across dev, val, and test.")
    return generated_records

if __name__ == "__main__":
    generate_synthetic_benchmark(DEFAULT_OUTPUT_DIR)
