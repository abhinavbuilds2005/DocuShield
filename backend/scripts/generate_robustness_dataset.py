"""
Generates a held-out dataset for generalization and robustness evaluation.
Creates 11 completely unseen documents:
- 5 Genuine variations (Original, JPEG Q=70, Resized, Screenshot/photo, Perspective/lighting)
- 6 Unseen Tampering vectors (Checksum alteration, Text splice, Copy-move, Font splice, Category splice, Realistic Photo-Swap with single normal JPEG re-save)

All documents have distinct names, IDs, dates, and locations not present in the 20-card regression dataset.
"""

import os
import cv2
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter

from backend.scripts.generate_mock_dataset import (
    render_aadhaar_card, render_pan_card, render_dl_card,
    generate_synthetic_portrait, generate_qr_code,
    apply_photo_swap, apply_font_splice, apply_copy_move_tamper
)
from backend.nlp.field_validator import generate_verhoeff_checksum

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ROBUST_DIR = os.path.join(DATA_DIR, "robustness")
os.makedirs(ROBUST_DIR, exist_ok=True)


def build_robustness_dataset():
    ground_truth = {}

    # -------------------------------------------------------------
    # 1. Unseen Genuine Original
    # -------------------------------------------------------------
    d1 = {
        "name": "Suresh Narang", "seed": 642, "gender": "M", "gender_full": "Male",
        "dob": "08/11/1988", "address": "15 Heritage Enclave, Civil Lines, Mock City 110054",
        "id_number": "61529384712" + generate_verhoeff_checksum("61529384712")
    }
    c1, _ = render_aadhaar_card(d1)
    f1 = "unseen_01_genuine_original.png"
    c1.save(os.path.join(ROBUST_DIR, f1))
    ground_truth[f1] = {"label": "GENUINE", "doc_type": "aadhaar", "expected_verdict": "AUTHENTIC", "type": "original"}

    # -------------------------------------------------------------
    # 2. Unseen Genuine JPEG-Compressed (Q=70)
    # -------------------------------------------------------------
    d2 = {
        "name": "Kavita Deshmukh", "seed": 711, "gender": "F", "gender_full": "Female",
        "dob": "19/04/1993", "address": "Sector 9, Kharghar, Navi Mumbai 410210",
        "id_number": "78219463520" + generate_verhoeff_checksum("78219463520")
    }
    c2, _ = render_aadhaar_card(d2)
    f2 = "unseen_02_genuine_jpeg_compressed.jpg"
    c2.save(os.path.join(ROBUST_DIR, f2), "JPEG", quality=70)
    ground_truth[f2] = {"label": "GENUINE", "doc_type": "aadhaar", "expected_verdict": "AUTHENTIC", "type": "jpeg_compression"}

    # -------------------------------------------------------------
    # 3. Unseen Genuine Resized (Scaled down 25% then upscaled)
    # -------------------------------------------------------------
    d3 = {
        "name": "Arjun Singhal", "seed": 834, "gender": "M", "gender_full": "Male",
        "father_name": "R. K. Singhal", "dob": "27/09/1990", "pan_number": "BCHPS4821M"
    }
    c3, _ = render_pan_card(d3)
    w3, h3 = c3.size
    c3_down = c3.resize((int(w3 * 0.75), int(h3 * 0.75)), Image.Resampling.BILINEAR)
    c3_res = c3_down.resize((w3, h3), Image.Resampling.BICUBIC)
    f3 = "unseen_03_genuine_resized.png"
    c3_res.save(os.path.join(ROBUST_DIR, f3))
    ground_truth[f3] = {"label": "GENUINE", "doc_type": "pan", "expected_verdict": "AUTHENTIC", "type": "resized"}

    # -------------------------------------------------------------
    # 4. Unseen Genuine Screenshot / Simulated Screen Photo
    # -------------------------------------------------------------
    d4 = {
        "name": "Sunita Menon", "seed": 912, "gender": "F",
        "dl_number": "KL-07-2018-0091823", "dob": "03/12/1987",
        "issue_date": "14/06/2018", "expiry_date": "13/06/2038",
        "vehicle_class": "MCWG, LMV", "blood_group": "A+", "state": "Kerala"
    }
    c4, _ = render_dl_card(d4)
    # Simulate screen capture: subtle RGB gamma shift + minor pixel grid pattern
    arr4 = np.array(c4, dtype=np.float32)
    gamma = 1.06
    arr4 = 255.0 * ((arr4 / 255.0) ** gamma)
    pattern = np.array([1.0, 0.98, 1.0] * (arr4.shape[1] // 3 + 1), dtype=np.float32)[:arr4.shape[1]]
    grid = np.tile(pattern[np.newaxis, :, np.newaxis], (arr4.shape[0], 1, 3))
    arr4 = np.clip(arr4 * grid, 0, 255).astype(np.uint8)
    f4 = "unseen_04_genuine_screenshot.png"
    Image.fromarray(arr4).save(os.path.join(ROBUST_DIR, f4))
    ground_truth[f4] = {"label": "GENUINE", "doc_type": "dl", "expected_verdict": "AUTHENTIC", "type": "screenshot"}

    # -------------------------------------------------------------
    # 5. Unseen Genuine Perspective / Lighting Variation
    # -------------------------------------------------------------
    d5 = {
        "name": "Manish Tiwari", "seed": 528, "gender": "M", "gender_full": "Male",
        "dob": "15/02/1985", "address": "B-42 Anand Nagar, Bhopal 462021",
        "id_number": "39481726503" + generate_verhoeff_checksum("39481726503")
    }
    c5, _ = render_aadhaar_card(d5)
    img5 = np.array(c5)
    h5, w5 = img5.shape[:2]
    # Subtle lighting gradient (1.05 down to 0.92 across width)
    gradient = np.tile(np.linspace(1.05, 0.92, w5), (h5, 1))[:, :, np.newaxis]
    lit5 = np.clip(img5 * gradient, 0, 255).astype(np.uint8)
    # Slight perspective warp (< 1% tilt)
    pts1 = np.float32([[0, 0], [w5, 0], [0, h5], [w5, h5]])
    pts2 = np.float32([[6, 3], [w5 - 4, 5], [3, h5 - 5], [w5 - 6, h5 - 4]])
    M5 = cv2.getPerspectiveTransform(pts1, pts2)
    warped5 = cv2.warpPerspective(lit5, M5, (w5, h5))
    f5 = "unseen_05_genuine_perspective_lighting.png"
    cv2.imwrite(os.path.join(ROBUST_DIR, f5), cv2.cvtColor(warped5, cv2.COLOR_RGB2BGR))
    ground_truth[f5] = {"label": "GENUINE", "doc_type": "aadhaar", "expected_verdict": "AUTHENTIC", "type": "perspective_lighting"}

    # -------------------------------------------------------------
    # 6. Unseen Tampered: Checksum Alteration
    # -------------------------------------------------------------
    d6 = {
        "name": "Meera Joshi", "seed": 781, "gender": "F", "gender_full": "Female",
        "dob": "11/07/1995", "address": "Flat 302, Palm Springs, Pune 411001",
        "id_number": "819273645014" # Fabricated invalid checksum
    }
    c6, _ = render_aadhaar_card(d6)
    f6 = "unseen_06_tampered_checksum.png"
    c6.save(os.path.join(ROBUST_DIR, f6))
    ground_truth[f6] = {"label": "TAMPERED", "doc_type": "aadhaar", "expected_verdict": "FLAGGED / TAMPERED", "type": "checksum_alteration"}

    # -------------------------------------------------------------
    # 7. Unseen Tampered: Text Splice on DOB
    # -------------------------------------------------------------
    d7 = {
        "name": "Rajesh Iyer", "seed": 815, "gender": "M", "gender_full": "Male",
        "father_name": "S. Iyer", "dob": "18/06/1982", "pan_number": "AFYPI7291K"
    }
    c7, _ = render_pan_card(d7)
    # Splice replacement DOB with rectangular cut seam
    c7_t = apply_font_splice(c7, box=(320, 222, 450, 245), replacement_text="01/01/2000", fill_bg="#f8fafc")
    f7 = "unseen_07_tampered_textsplice.png"
    c7_t.save(os.path.join(ROBUST_DIR, f7))
    ground_truth[f7] = {"label": "TAMPERED", "doc_type": "pan", "expected_verdict": "FLAGGED / TAMPERED", "type": "text_splice"}

    # -------------------------------------------------------------
    # 8. Unseen Tampered: Copy-Move Duplication
    # -------------------------------------------------------------
    d8 = {
        "name": "Tanvi Rastogi", "seed": 664, "gender": "F", "gender_full": "Female",
        "dob": "05/10/1991", "address": "77 Mall Road, Lucknow 226001",
        "id_number": "51829374610" + generate_verhoeff_checksum("51829374610")
    }
    c8, _ = render_aadhaar_card(d8)
    # Clone QR code or emblem to another location
    c8_t = apply_copy_move_tamper(c8, src_box=(620, 180, 740, 300), dst_box=(250, 240, 370, 360))
    f8 = "unseen_08_tampered_copymove.png"
    c8_t.save(os.path.join(ROBUST_DIR, f8))
    ground_truth[f8] = {"label": "TAMPERED", "doc_type": "aadhaar", "expected_verdict": "FLAGGED / TAMPERED", "type": "copy_move"}

    # -------------------------------------------------------------
    # 9. Unseen Tampered: Font / Name Splice
    # -------------------------------------------------------------
    d9 = {
        "name": "Ananya Roy", "seed": 923, "gender": "F", "gender_full": "Female",
        "dob": "22/08/1997", "address": "Plot 12, Salt Lake, Kolkata 700091",
        "id_number": "93817264501" + generate_verhoeff_checksum("93817264501")
    }
    c9, _ = render_aadhaar_card(d9)
    c9_t = apply_font_splice(c9, box=(308, 135, 530, 165), replacement_text="SANJAY KAPOOR", fill_bg="#ffffff")
    f9 = "unseen_09_tampered_fontsplice.png"
    c9_t.save(os.path.join(ROBUST_DIR, f9))
    ground_truth[f9] = {"label": "TAMPERED", "doc_type": "aadhaar", "expected_verdict": "FLAGGED / TAMPERED", "type": "font_splice"}

    # -------------------------------------------------------------
    # 10. Unseen Tampered: Category Replacement
    # -------------------------------------------------------------
    d10 = {
        "name": "Girish Patel", "seed": 456, "gender": "M",
        "dl_number": "GJ-01-2016-0045612", "dob": "14/03/1984",
        "issue_date": "20/05/2016", "expiry_date": "19/05/2036",
        "vehicle_class": "LMV", "blood_group": "B+", "state": "Gujarat"
    }
    c10, _ = render_dl_card(d10)
    c10_t = apply_font_splice(c10, box=(375, 298, 580, 325), replacement_text="HEAVY PASSENGER VEHICLE", fill_bg="#f8fafc")
    f10 = "unseen_10_tampered_categorysplice.png"
    c10_t.save(os.path.join(ROBUST_DIR, f10))
    ground_truth[f10] = {"label": "TAMPERED", "doc_type": "dl", "expected_verdict": "FLAGGED / TAMPERED", "type": "category_splice"}

    # -------------------------------------------------------------
    # 11. Unseen Realistic Photo-Swap (Single normal JPEG re-save, NO double compression)
    # -------------------------------------------------------------
    d11 = {
        "name": "Deepak Chopra", "seed": 339, "gender": "M", "gender_full": "Male",
        "dob": "30/01/1989", "address": "90 Model Town, Delhi 110009",
        "id_number": "72918463501" + generate_verhoeff_checksum("72918463501")
    }
    c11, _ = render_aadhaar_card(d11)
    foreign_p = generate_synthetic_portrait("Unknown Intruder", seed=999, gender="M")
    # Apply photo swap with rectangular cut seam boundary
    c11_t = apply_photo_swap(c11, foreign_p, (50, 130, 200, 310))
    # CRITICAL: Save with single normal JPEG re-save at standard quality 85 (NO double compression)
    f11 = "unseen_11_tampered_realistic_photoswap.jpg"
    c11_t.save(os.path.join(ROBUST_DIR, f11), "JPEG", quality=85)
    ground_truth[f11] = {"label": "TAMPERED", "doc_type": "aadhaar", "expected_verdict": "FLAGGED / TAMPERED", "type": "realistic_photoswap"}

    # Save held-out ground truth JSON
    gt_path = os.path.join(ROBUST_DIR, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"[SUCCESS] Generated 11 held-out robustness documents in: {ROBUST_DIR}")
    print(f"Genuine: 5 | Tampered: 6 | Ground truth saved to: {gt_path}")


if __name__ == "__main__":
    build_robustness_dataset()
