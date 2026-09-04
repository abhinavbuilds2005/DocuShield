"""
Synthetic Mock Identity Document Generator (Refined & Calibrated)
Generates 20 mock cards (8 Genuine + 12 Tampered pairs)
Uses clean diagonal watermark to prevent artificial keypoint lattice patterns
Saves images, ground_truth.json, and rich OCR sidecars
"""

import os
import io
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from PIL.PngImagePlugin import PngInfo

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def draw_watermark(img: Image.Image):
    """Draws clear, prominent, non-repetitive synthetic research watermarks."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)

    # Top red safety banner
    odraw.rectangle([0, 0, w, 22], fill=(190, 18, 60, 255))
    odraw.text((w // 2, 11), "SAMPLE / MOCK — NOT A REAL GOVERNMENT DOCUMENT — SIH 2026 DEMO",
               fill=(255, 255, 255, 255), anchor="mm")
    
    # Bottom red safety banner
    odraw.rectangle([0, h - 20, w, h], fill=(190, 18, 60, 255))
    odraw.text((w // 2, h - 10), "SYNTHETIC FICTIONAL TEST DATASET — ETHICAL AI RESEARCH ONLY",
               fill=(255, 255, 255, 255), anchor="mm")

    # Single wide diagonal watermark band across the center
    odraw.line([(0, h - 70), (w, 70)], fill=(226, 232, 240, 130), width=44)
    odraw.text((w // 2, h // 2), "MOCK / SPECIMEN — NOT REAL ID",
               fill=(148, 163, 184, 180), anchor="mm")

    composited = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img.paste(composited)


def generate_synthetic_portrait(name: str, seed: int = 1, gender: str = "M") -> Image.Image:
    random.seed(seed)
    pw, ph = 150, 180
    avatar = Image.new("RGB", (pw, ph), color="#f1f5f9")
    adraw = ImageDraw.Draw(avatar)

    bg_color = (210 + (seed % 30), 225 + (seed % 20), 240 + (seed % 15))
    adraw.rectangle([0, 0, pw, ph], fill=bg_color)

    skin_tones = [(240, 200, 160), (220, 175, 135), (200, 150, 110), (235, 190, 150)]
    skin = skin_tones[seed % len(skin_tones)]
    
    shirt_colors = [(37, 99, 235), (15, 118, 110), (124, 58, 237), (180, 83, 9), (71, 85, 105)]
    shirt = shirt_colors[seed % len(shirt_colors)]
    adraw.ellipse([-20, ph - 70, pw + 20, ph + 70], fill=shirt)

    adraw.rectangle([pw // 2 - 18, ph // 2 - 10, pw // 2 + 18, ph // 2 + 35], fill=(skin[0]-15, skin[1]-15, skin[2]-15))

    head_box = [pw // 2 - 42, 30, pw // 2 + 42, 125]
    adraw.ellipse(head_box, fill=skin)

    hair_colors = [(30, 25, 20), (45, 35, 30), (60, 45, 35)]
    hair = hair_colors[seed % len(hair_colors)]
    adraw.ellipse([pw // 2 - 45, 20, pw // 2 + 45, 75], fill=hair)
    if gender == "F":
        adraw.rectangle([pw // 2 - 45, 50, pw // 2 - 35, 120], fill=hair)
        adraw.rectangle([pw // 2 + 35, 50, pw // 2 + 45, 120], fill=hair)

    adraw.ellipse([pw // 2 - 25, 68, pw // 2 - 15, 76], fill="#1e293b")
    adraw.ellipse([pw // 2 + 15, 68, pw // 2 + 25, 76], fill="#1e293b")
    adraw.line([pw // 2 - 27, 62, pw // 2 - 13, 62], fill=hair, width=2)
    adraw.line([pw // 2 + 13, 62, pw // 2 + 27, 62], fill=hair, width=2)
    adraw.arc([pw // 2 - 18, 85, pw // 2 + 18, 105], start=20, end=160, fill="#7f1d1d", width=2)
    adraw.rectangle([0, 0, pw - 1, ph - 1], outline="#94a3b8", width=1)
    return avatar


def generate_qr_code(text_payload: str) -> Image.Image:
    qw, qh = 120, 120
    qr = Image.new("RGB", (qw, qh), color="white")
    draw = ImageDraw.Draw(qr)
    grid_size = 24
    cell_w = qw / grid_size

    for corner in [(2, 2), (grid_size - 7, 2), (2, grid_size - 7)]:
        cx, cy = corner
        draw.rectangle([cx * cell_w, cy * cell_w, (cx + 5) * cell_w, (cy + 5) * cell_w], fill="black")
        draw.rectangle([(cx + 1) * cell_w, (cy + 1) * cell_w, (cx + 4) * cell_w, (cy + 4) * cell_w], fill="white")
        draw.rectangle([(cx + 2) * cell_w, (cy + 2) * cell_w, (cx + 3) * cell_w, (cy + 3) * cell_w], fill="black")

    hval = abs(hash(text_payload))
    for r in range(grid_size):
        for c in range(grid_size):
            if (r < 8 and c < 8) or (r < 8 and c >= grid_size - 8) or (r >= grid_size - 8 and c < 8):
                continue
            if ((r * 31 + c * 17 + hval) % 3) == 0:
                draw.rectangle([c * cell_w, r * cell_w, (c + 1) * cell_w, (r + 1) * cell_w], fill="black")

    return qr


# --- Template 1: Aadhaar Style ---
def render_aadhaar_card(data: dict) -> tuple:
    w, h = 800, 500
    img = Image.new("RGB", (w, h), color="#ffffff")
    draw = ImageDraw.Draw(img)

    for y in range(25, h - 25, 12):
        shade = 250 - (y % 6)
        draw.line([(0, y), (w, y)], fill=(shade, shade + 2, shade + 4), width=1)

    draw.rectangle([0, 22, w, 95], fill="#fff7ed")
    draw.rectangle([0, 93, w, 97], fill="#ea580c")

    draw.ellipse([40, 32, 90, 82], fill="#0284c7")
    draw.text((65, 57), "AG", fill="#ffffff", anchor="mm")

    draw.text((110, 42), "Republic of Mockland", fill="#9a3412")
    draw.text((110, 62), "Unique Identification Authority", fill="#1e293b")
    draw.text((110, 80), "Mock National Citizen ID Card", fill="#64748b")

    portrait = generate_synthetic_portrait(data["name"], seed=data["seed"], gender=data["gender"])
    img.paste(portrait, (50, 130))

    draw.text((230, 140), "Name:", fill="#64748b")
    draw.text((310, 138), data["name"], fill="#0f172a")

    draw.text((230, 185), "DOB:", fill="#64748b")
    draw.text((310, 183), data["dob"], fill="#0f172a")

    draw.text((230, 230), "Gender:", fill="#64748b")
    draw.text((310, 228), data["gender_full"], fill="#0f172a")

    draw.text((230, 275), "Address:", fill="#64748b")
    draw.text((310, 273), data["address"], fill="#334155")

    id_formatted = f"{data['id_number'][:4]} {data['id_number'][4:8]} {data['id_number'][8:12]}"
    draw.rectangle([210, 360, 750, 425], fill="#f8fafc", outline="#cbd5e1", width=1)
    draw.text((480, 392), id_formatted, fill="#be123c", anchor="mm")

    qr = generate_qr_code(f"MOCK-ID:{data['id_number']}|NAME:{data['name']}|DOB:{data['dob']}")
    img.paste(qr, (640, 130))

    draw_watermark(img)

    lines = [
        {"text": "Republic of Mockland Unique Identification Authority Mock National Citizen ID Card", "confidence": 0.98, "box": [110, 42, 450, 40]},
        {"text": f"Name: {data['name']}", "confidence": 0.96, "box": [230, 138, 280, 30]},
        {"text": f"DOB: {data['dob']}", "confidence": 0.97, "box": [230, 183, 180, 30]},
        {"text": f"Gender: {data['gender_full']}", "confidence": 0.95, "box": [230, 228, 160, 30]},
        {"text": f"Address: {data['address']}", "confidence": 0.92, "box": [230, 273, 400, 30]},
        {"text": id_formatted, "confidence": 0.99, "box": [210, 360, 540, 65]}
    ]
    tokens = []
    for l in lines:
        wds = l["text"].split()
        bw = l["box"][2] // max(1, len(wds))
        for i, w_str in enumerate(wds):
            tokens.append({"text": w_str, "confidence": l["confidence"], "box": [l["box"][0] + i * bw, l["box"][1], bw, l["box"][3]]})

    ocr_data = {
        "full_text": " ".join([l["text"] for l in lines]),
        "lines": lines,
        "tokens": tokens,
        "image_dims": [w, h]
    }
    return img, ocr_data


# --- Template 2: PAN Style ---
def render_pan_card(data: dict) -> tuple:
    w, h = 800, 500
    img = Image.new("RGB", (w, h), color="#eff6ff")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 22, w, 100], fill="#1e3a8a")
    draw.ellipse([35, 35, 85, 85], fill="#f59e0b")
    draw.text((60, 60), "TX", fill="#1e3a8a", anchor="mm")

    draw.text((105, 42), "INCOME TAX DEPARTMENT", fill="#ffffff")
    draw.text((105, 62), "GOVERNMENT OF MOCKLAND", fill="#93c5fd")
    draw.text((105, 82), "PERMANENT ACCOUNT NUMBER CARD", fill="#e2e8f0")

    portrait = generate_synthetic_portrait(data["name"], seed=data["seed"], gender=data["gender"])
    img.paste(portrait, (50, 125))

    draw.text((230, 130), "Cardholder's Name", fill="#475569")
    draw.text((230, 150), data["name"].upper(), fill="#0f172a")

    draw.text((230, 190), "Father's Name", fill="#475569")
    draw.text((230, 210), data["father_name"].upper(), fill="#0f172a")

    draw.text((230, 250), "Date of Birth", fill="#475569")
    draw.text((230, 270), data["dob"], fill="#0f172a")

    draw.rectangle([220, 320, 560, 385], fill="#ffffff", outline="#1d4ed8", width=2)
    draw.text((240, 335), "Permanent Account Number", fill="#64748b")
    draw.text((390, 365), data["pan_number"], fill="#1e3a8a", anchor="mm")

    draw.rectangle([220, 400, 560, 445], fill="#ffffff", outline="#94a3b8", width=1)
    draw.text((230, 410), "Signature / Signature Strip", fill="#94a3b8")
    sig_pts = [(240, 435), (280, 420), (320, 438), (370, 418), (420, 432), (480, 420), (530, 430)]
    for i in range(len(sig_pts) - 1):
        draw.line([sig_pts[i], sig_pts[i+1]], fill="#0f172a", width=2)

    qr = generate_qr_code(f"PAN:{data['pan_number']}|{data['name']}")
    img.paste(qr, (635, 140))

    draw_watermark(img)

    lines = [
        {"text": "INCOME TAX DEPARTMENT GOVERNMENT OF MOCKLAND PERMANENT ACCOUNT NUMBER CARD", "confidence": 0.98, "box": [105, 42, 500, 50]},
        {"text": f"Name {data['name'].upper()}", "confidence": 0.96, "box": [230, 130, 300, 45]},
        {"text": f"Father's Name {data['father_name'].upper()}", "confidence": 0.95, "box": [230, 190, 300, 45]},
        {"text": f"Date of Birth {data['dob']}", "confidence": 0.97, "box": [230, 250, 200, 45]},
        {"text": f"Permanent Account Number {data['pan_number']}", "confidence": 0.99, "box": [220, 320, 340, 65]}
    ]
    tokens = []
    for l in lines:
        wds = l["text"].split()
        bw = l["box"][2] // max(1, len(wds))
        for i, w_str in enumerate(wds):
            tokens.append({"text": w_str, "confidence": l["confidence"], "box": [l["box"][0] + i * bw, l["box"][1], bw, l["box"][3]]})

    ocr_data = {
        "full_text": " ".join([l["text"] for l in lines]),
        "lines": lines,
        "tokens": tokens,
        "image_dims": [w, h]
    }
    return img, ocr_data


# --- Template 3: Driving License Style ---
def render_dl_card(data: dict) -> tuple:
    w, h = 800, 500
    img = Image.new("RGB", (w, h), color="#f8fafc")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 22, w, 95], fill="#065f46")
    draw.ellipse([40, 32, 90, 82], fill="#34d399")
    draw.text((65, 57), "DL", fill="#065f46", anchor="mm")

    draw.text((110, 40), "UNION OF MOCKLAND — MOTOR VEHICLES DEPT", fill="#ffffff")
    draw.text((110, 60), "DRIVING LICENCE / PERMIT", fill="#a7f3d0")
    draw.text((110, 78), f"State Transport Authority ({data['state']})", fill="#e2e8f0")

    portrait = generate_synthetic_portrait(data["name"], seed=data["seed"], gender=data["gender"])
    img.paste(portrait, (50, 130))

    draw.rectangle([220, 120, 740, 165], fill="#ecfdf5", outline="#10b981", width=1)
    draw.text((235, 126), "Licence No:", fill="#047857")
    draw.text((380, 142), data["dl_number"], fill="#064e3b", anchor="mm")

    draw.text((235, 185), "Name:", fill="#64748b")
    draw.text((320, 185), data["name"], fill="#0f172a")

    draw.text((235, 225), "DOB:", fill="#64748b")
    draw.text((320, 225), data["dob"], fill="#0f172a")

    draw.text((235, 265), "Issue Date:", fill="#64748b")
    draw.text((320, 265), data["issue_date"], fill="#0f172a")

    draw.text((500, 265), "Valid Till:", fill="#64748b")
    draw.text((580, 265), data["expiry_date"], fill="#0f172a")

    draw.text((235, 305), "Class of Vehicles:", fill="#64748b")
    draw.text((380, 305), data["vehicle_class"], fill="#1e293b")

    draw.text((235, 345), "Blood Group:", fill="#64748b")
    draw.text((340, 345), data["blood_group"], fill="#b91c1c")

    draw.rectangle([610, 180, 730, 240], fill="#fef08a", outline="#ca8a04", width=2)
    draw.text((670, 210), "SMART CHIP", fill="#854d0e", anchor="mm")

    draw_watermark(img)

    lines = [
        {"text": "UNION OF MOCKLAND MOTOR VEHICLES DEPT DRIVING LICENCE", "confidence": 0.98, "box": [110, 40, 500, 50]},
        {"text": f"Licence No: {data['dl_number']}", "confidence": 0.99, "box": [220, 120, 520, 45]},
        {"text": f"Name: {data['name']}", "confidence": 0.96, "box": [235, 185, 250, 30]},
        {"text": f"DOB: {data['dob']}", "confidence": 0.97, "box": [235, 225, 200, 30]},
        {"text": f"Issue Date: {data['issue_date']}", "confidence": 0.95, "box": [235, 265, 200, 30]},
        {"text": f"Valid Till: {data['expiry_date']}", "confidence": 0.95, "box": [500, 265, 200, 30]},
        {"text": f"Class of Vehicles: {data['vehicle_class']}", "confidence": 0.94, "box": [235, 305, 350, 30]},
        {"text": f"Blood Group: {data['blood_group']}", "confidence": 0.93, "box": [235, 345, 180, 30]}
    ]
    tokens = []
    for l in lines:
        wds = l["text"].split()
        bw = l["box"][2] // max(1, len(wds))
        for i, w_str in enumerate(wds):
            tokens.append({"text": w_str, "confidence": l["confidence"], "box": [l["box"][0] + i * bw, l["box"][1], bw, l["box"][3]]})

    ocr_data = {
        "full_text": " ".join([l["text"] for l in lines]),
        "lines": lines,
        "tokens": tokens,
        "image_dims": [w, h]
    }
    return img, ocr_data


# --- Tampering Helpers ---

def apply_photo_swap(base_img: Image.Image, foreign_portrait: Image.Image, box: tuple = (50, 130, 200, 310)) -> Image.Image:
    tampered = base_img.copy()
    pw = box[2] - box[0]
    ph = box[3] - box[1]
    resized_foreign = foreign_portrait.resize((pw, ph))
    enhancer = ImageEnhance.Contrast(resized_foreign)
    resized_foreign = enhancer.enhance(1.4)
    tampered.paste(resized_foreign, (box[0], box[1]))
    # Draw obvious cut/splice boundary seam around photo
    tdraw = ImageDraw.Draw(tampered)
    tdraw.rectangle([box[0]-1, box[1]-1, box[2], box[3]], outline="#1e293b", width=2)
    return tampered


def apply_font_splice(base_img: Image.Image, box: tuple, replacement_text: str, fill_bg: str = "#ffffff") -> Image.Image:
    tampered = base_img.copy()
    tdraw = ImageDraw.Draw(tampered)
    tdraw.rectangle(box, fill=fill_bg)
    tdraw.text((box[0] + 5, box[1] + 3), replacement_text, fill="#000000")
    tdraw.rectangle(box, outline="#64748b", width=1)
    return tampered


def apply_copy_move_tamper(base_img: Image.Image, src_box: tuple, dst_box: tuple) -> Image.Image:
    tampered = base_img.copy()
    patch = base_img.crop(src_box)
    patch_resized = patch.resize((dst_box[2] - dst_box[0], dst_box[3] - dst_box[1]))
    tampered.paste(patch_resized, (dst_box[0], dst_box[1]))
    return tampered


def apply_ela_double_compression(base_img: Image.Image, tampered_region_box: tuple) -> Image.Image:
    # 1. Base card at quality 70
    buf1 = io.BytesIO()
    base_img.save(buf1, format="JPEG", quality=70)
    buf1.seek(0)
    low_q = Image.open(buf1).convert("RGB")

    # 2. Paste high-noise / sharp patch into low_q image
    fresh_patch = base_img.crop(tampered_region_box)
    enhancer = ImageEnhance.Sharpness(fresh_patch)
    fresh_patch = enhancer.enhance(3.0)
    low_q.paste(fresh_patch, (tampered_region_box[0], tampered_region_box[1]))

    # 3. Re-save at quality 96
    buf2 = io.BytesIO()
    low_q.save(buf2, format="JPEG", quality=96)
    buf2.seek(0)
    return Image.open(buf2).convert("RGB")


def save_card_with_sidecar(img: Image.Image, ocr_data: dict, filename: str, extra_meta: dict = None):
    img_path = os.path.join(DATA_DIR, filename)
    if filename.endswith(".jpg"):
        img.save(img_path, format="JPEG", quality=92)
    elif extra_meta and extra_meta.get("pnginfo"):
        img.save(img_path, pnginfo=extra_meta["pnginfo"])
    else:
        img.save(img_path)

    # Rebuild tokens from lines to ensure updated confidence and text are synchronized
    tokens = []
    for l in ocr_data.get("lines", []):
        wds = l["text"].split()
        bw = l["box"][2] // max(1, len(wds))
        for i, w_str in enumerate(wds):
            tokens.append({
                "text": w_str,
                "confidence": l["confidence"],
                "box": [l["box"][0] + i * bw, l["box"][1], bw, l["box"][3]]
            })
    ocr_data["tokens"] = tokens

    sidecar_fn = os.path.splitext(filename)[0] + ".ocr.json"
    sidecar_path = os.path.join(DATA_DIR, sidecar_fn)
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, indent=2)


def build_dataset():
    print("[Dataset] Building calibrated synthetic mock ID dataset...")
    from backend.nlp.field_validator import generate_verhoeff_checksum
    ground_truth = {}

    # 1. Aadhaar 1 Genuine
    a1_data = {
        "name": "Aarav Sharma", "seed": 101, "gender": "M", "gender_full": "Male",
        "dob": "14/08/1992", "address": "42 Greenfield Lane, Civil Lines, Mock City 110001"
    }
    cd1 = generate_verhoeff_checksum("54289162381")
    a1_data["id_number"] = "54289162381" + cd1
    c_a1, ocr_a1 = render_aadhaar_card(a1_data)
    f_a1 = "mock_aadhaar_01_genuine.png"
    save_card_with_sidecar(c_a1, ocr_a1, f_a1)
    ground_truth[f_a1] = {"label": "GENUINE", "doc_type": "aadhaar", "name": a1_data["name"], "id_number": a1_data["id_number"], "tampering_types": [], "expected_verdict": "AUTHENTIC"}

    # 2. Aadhaar 1 Tampered: Photo Swap & Boundary Seam
    for_p = generate_synthetic_portrait("Unknown Person", seed=999, gender="M")
    c_a1_t1 = apply_photo_swap(c_a1, for_p, (50, 130, 200, 310))
    c_a1_t1 = apply_ela_double_compression(c_a1_t1, (50, 130, 200, 310))
    f_a1_t1 = "mock_aadhaar_01_tampered_photoswap.jpg"
    save_card_with_sidecar(c_a1_t1, ocr_a1, f_a1_t1)
    ground_truth[f_a1_t1] = {"label": "TAMPERED", "doc_type": "aadhaar", "name": a1_data["name"], "id_number": a1_data["id_number"], "tampering_types": ["photo_swap", "ela_double_compression"], "tampered_boxes": [[50, 130, 150, 180]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 3. Aadhaar 1 Tampered: Verhoeff Checksum Corruption
    a1_corrupt = dict(a1_data)
    last_d = a1_corrupt["id_number"][-1]
    a1_corrupt["id_number"] = a1_corrupt["id_number"][:-1] + str((int(last_d) + 3) % 10)
    c_a1_t2, ocr_a1_t2 = render_aadhaar_card(a1_corrupt)
    f_a1_t2 = "mock_aadhaar_02_tampered_checksum.png"
    save_card_with_sidecar(c_a1_t2, ocr_a1_t2, f_a1_t2)
    ground_truth[f_a1_t2] = {"label": "TAMPERED", "doc_type": "aadhaar", "name": a1_corrupt["name"], "id_number": a1_corrupt["id_number"], "tampering_types": ["verhoeff_checksum_corruption"], "tampered_boxes": [[210, 360, 540, 65]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 4. Aadhaar 2 Genuine
    a2_data = {
        "name": "Priya Verma", "seed": 202, "gender": "F", "gender_full": "Female",
        "dob": "21/05/1996", "address": "B-12 Lotus Heights, Sector 15, Mock City 110022"
    }
    cd2 = generate_verhoeff_checksum("87421935612")
    a2_data["id_number"] = "87421935612" + cd2
    c_a2, ocr_a2 = render_aadhaar_card(a2_data)
    f_a2 = "mock_aadhaar_03_genuine.png"
    save_card_with_sidecar(c_a2, ocr_a2, f_a2)
    ground_truth[f_a2] = {"label": "GENUINE", "doc_type": "aadhaar", "name": a2_data["name"], "id_number": a2_data["id_number"], "tampering_types": [], "expected_verdict": "AUTHENTIC"}

    # 5. Aadhaar 2 Tampered: Font Splice on Name
    c_a2_t3 = apply_font_splice(c_a2, box=(308, 135, 520, 165), replacement_text="VIKRAM MALHOTRA", fill_bg="#ffffff")
    ocr_a2_t3 = dict(ocr_a2)
    ocr_a2_t3["full_text"] = ocr_a2["full_text"].replace(a2_data["name"], "VIKRAM MALHOTRA")
    ocr_a2_t3["lines"] = [dict(l) for l in ocr_a2["lines"]]
    for l in ocr_a2_t3["lines"]:
        if "Name:" in l["text"]:
            l["text"] = "Name: VIKRAM MALHOTRA"
            l["confidence"] = 0.52
    f_a2_t3 = "mock_aadhaar_04_tampered_fontsplice.png"
    save_card_with_sidecar(c_a2_t3, ocr_a2_t3, f_a2_t3)
    ground_truth[f_a2_t3] = {"label": "TAMPERED", "doc_type": "aadhaar", "name": "VIKRAM MALHOTRA", "id_number": a2_data["id_number"], "tampering_types": ["font_sharpness_splice", "name_manipulation"], "tampered_boxes": [[308, 135, 212, 30]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 6. PAN 1 Genuine
    p1_data = {
        "name": "Rohit Gupta", "father_name": "Suresh Gupta", "seed": 303,
        "gender": "M", "dob": "10/11/1988", "pan_number": "ABCPG1234F"
    }
    c_p1, ocr_p1 = render_pan_card(p1_data)
    f_p1 = "mock_pan_01_genuine.png"
    save_card_with_sidecar(c_p1, ocr_p1, f_p1)
    ground_truth[f_p1] = {"label": "GENUINE", "doc_type": "pan", "name": p1_data["name"], "pan_number": p1_data["pan_number"], "tampering_types": [], "expected_verdict": "AUTHENTIC"}

    # 7. PAN 1 Tampered: Date Splice (Invalid leap date 31/02/1988)
    c_p1_t1 = apply_font_splice(c_p1, box=(228, 268, 380, 298), replacement_text="31/02/1988", fill_bg="#eff6ff")
    ocr_p1_t1 = dict(ocr_p1)
    ocr_p1_t1["full_text"] = ocr_p1["full_text"].replace("10/11/1988", "31/02/1988")
    ocr_p1_t1["lines"] = [dict(l) for l in ocr_p1["lines"]]
    for l in ocr_p1_t1["lines"]:
        if "Date of Birth" in l["text"]:
            l["text"] = "Date of Birth 31/02/1988"
            l["confidence"] = 0.54
    f_p1_t1 = "mock_pan_02_tampered_datesplice.png"
    save_card_with_sidecar(c_p1_t1, ocr_p1_t1, f_p1_t1)
    ground_truth[f_p1_t1] = {"label": "TAMPERED", "doc_type": "pan", "name": p1_data["name"], "pan_number": p1_data["pan_number"], "tampering_types": ["invalid_date_logic", "font_anti_aliasing_mismatch"], "tampered_boxes": [[228, 268, 152, 30]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 8. PAN 1 Tampered: Copy-Move Cloned Stamp / QR
    c_p1_t2 = apply_copy_move_tamper(c_p1, src_box=(635, 140, 755, 260), dst_box=(220, 395, 340, 515))
    f_p1_t2 = "mock_pan_03_tampered_copymove.png"
    save_card_with_sidecar(c_p1_t2, ocr_p1, f_p1_t2)
    ground_truth[f_p1_t2] = {"label": "TAMPERED", "doc_type": "pan", "name": p1_data["name"], "pan_number": p1_data["pan_number"], "tampering_types": ["copy_move_cloning"], "tampered_boxes": [[220, 395, 120, 120]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 9. PAN 2 Genuine
    p2_data = {
        "name": "Ananya Sen", "father_name": "Debabrata Sen", "seed": 404,
        "gender": "F", "dob": "05/03/1994", "pan_number": "BZXPS5432K"
    }
    c_p2, ocr_p2 = render_pan_card(p2_data)
    f_p2 = "mock_pan_04_genuine.png"
    save_card_with_sidecar(c_p2, ocr_p2, f_p2)
    ground_truth[f_p2] = {"label": "GENUINE", "doc_type": "pan", "name": p2_data["name"], "pan_number": p2_data["pan_number"], "tampering_types": [], "expected_verdict": "AUTHENTIC"}

    # 10. PAN 2 Tampered: Regex Format Violation
    p2_bad = dict(p2_data)
    p2_bad["pan_number"] = "ABC9999999"
    c_p2_t3, ocr_p2_t3 = render_pan_card(p2_bad)
    f_p2_t3 = "mock_pan_05_tampered_regex.png"
    save_card_with_sidecar(c_p2_t3, ocr_p2_t3, f_p2_t3)
    ground_truth[f_p2_t3] = {"label": "TAMPERED", "doc_type": "pan", "name": p2_bad["name"], "pan_number": p2_bad["pan_number"], "tampering_types": ["regex_format_violation"], "tampered_boxes": [[220, 320, 340, 65]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 11. DL 1 Genuine
    dl1_data = {
        "name": "Karan Singhania", "seed": 505, "gender": "M", "dob": "18/07/1985",
        "state": "DL-01", "dl_number": "DL-1420110012345", "issue_date": "12/03/2011",
        "expiry_date": "11/03/2031", "vehicle_class": "MCWG, LMV", "blood_group": "O+ve"
    }
    c_dl1, ocr_dl1 = render_dl_card(dl1_data)
    f_dl1 = "mock_dl_01_genuine.png"
    save_card_with_sidecar(c_dl1, ocr_dl1, f_dl1)
    ground_truth[f_dl1] = {"label": "GENUINE", "doc_type": "dl", "name": dl1_data["name"], "dl_number": dl1_data["dl_number"], "tampering_types": [], "expected_verdict": "AUTHENTIC"}

    # 12. DL 1 Tampered: Photoshop Metadata Signature & Photo Swap
    c_dl1_t1 = apply_photo_swap(c_dl1, generate_synthetic_portrait("Suspect", 777, "M"), box=(50, 130, 200, 310))
    pi = PngInfo()
    pi.add_text("Software", "Adobe Photoshop 24.1 (Windows)")
    f_dl1_t1 = "mock_dl_02_tampered_photoshop_meta.png"
    save_card_with_sidecar(c_dl1_t1, ocr_dl1, f_dl1_t1, extra_meta={"pnginfo": pi})
    ground_truth[f_dl1_t1] = {"label": "TAMPERED", "doc_type": "dl", "name": dl1_data["name"], "dl_number": dl1_data["dl_number"], "tampering_types": ["photo_swap", "editing_software_metadata"], "tampered_boxes": [[50, 130, 150, 180]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 13. DL 1 Tampered: Vehicle Category Splice
    c_dl1_t2 = apply_font_splice(c_dl1, box=(378, 300, 580, 330), replacement_text="TRANS, HMV, HAZ", fill_bg="#f8fafc")
    ocr_dl1_t2 = dict(ocr_dl1)
    ocr_dl1_t2["full_text"] = ocr_dl1["full_text"].replace("MCWG, LMV", "TRANS, HMV, HAZ")
    ocr_dl1_t2["lines"] = [dict(l) for l in ocr_dl1["lines"]]
    for l in ocr_dl1_t2["lines"]:
        if "Class of Vehicles" in l["text"]:
            l["text"] = "Class of Vehicles: TRANS, HMV, HAZ"
            l["confidence"] = 0.51
    f_dl1_t2 = "mock_dl_03_tampered_categorysplice.png"
    save_card_with_sidecar(c_dl1_t2, ocr_dl1_t2, f_dl1_t2)
    ground_truth[f_dl1_t2] = {"label": "TAMPERED", "doc_type": "dl", "name": dl1_data["name"], "dl_number": dl1_data["dl_number"], "tampering_types": ["vehicle_category_splice", "font_discrepancy"], "tampered_boxes": [[378, 300, 202, 30]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 14. DL 2 Genuine
    dl2_data = {
        "name": "Meera Nambiar", "seed": 606, "gender": "F", "dob": "29/09/1998",
        "state": "KL-07", "dl_number": "KL-0720190054321", "issue_date": "15/10/2019",
        "expiry_date": "14/10/2039", "vehicle_class": "LMV (NT)", "blood_group": "B+ve"
    }
    c_dl2, ocr_dl2 = render_dl_card(dl2_data)
    f_dl2 = "mock_dl_04_genuine.png"
    save_card_with_sidecar(c_dl2, ocr_dl2, f_dl2)
    ground_truth[f_dl2] = {"label": "GENUINE", "doc_type": "dl", "name": dl2_data["name"], "dl_number": dl2_data["dl_number"], "tampering_types": [], "expected_verdict": "AUTHENTIC"}

    # 15. DL 2 Tampered: Illogical Expiry Prior to Issue Date
    dl2_bad = dict(dl2_data)
    dl2_bad["expiry_date"] = "14/10/2015"
    c_dl2_t3, ocr_dl2_t3 = render_dl_card(dl2_bad)
    f_dl2_t3 = "mock_dl_05_tampered_chronology.png"
    save_card_with_sidecar(c_dl2_t3, ocr_dl2_t3, f_dl2_t3)
    ground_truth[f_dl2_t3] = {"label": "TAMPERED", "doc_type": "dl", "name": dl2_bad["name"], "dl_number": dl2_bad["dl_number"], "tampering_types": ["chronological_date_contradiction"], "tampered_boxes": [[500, 265, 200, 30]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 16. Aadhaar 3 Genuine
    a3_data = {
        "name": "Devendra Joshi", "seed": 707, "gender": "M", "gender_full": "Male",
        "dob": "02/01/1954", "address": "108 Heritage Colony, Mockpur 400050"
    }
    cd3 = generate_verhoeff_checksum("31459265358")
    a3_data["id_number"] = "31459265358" + cd3
    c_a3, ocr_a3 = render_aadhaar_card(a3_data)
    f_a3 = "mock_aadhaar_05_genuine.png"
    save_card_with_sidecar(c_a3, ocr_a3, f_a3)
    ground_truth[f_a3] = {"label": "GENUINE", "doc_type": "aadhaar", "name": a3_data["name"], "id_number": a3_data["id_number"], "tampering_types": [], "expected_verdict": "AUTHENTIC"}

    # 17. Aadhaar 3 Tampered: Cloned QR over Address (Copy-Move)
    c_a3_t1 = apply_copy_move_tamper(c_a3, src_box=(640, 130, 760, 250), dst_box=(310, 270, 430, 390))
    f_a3_t1 = "mock_aadhaar_06_tampered_copymove.png"
    save_card_with_sidecar(c_a3_t1, ocr_a3, f_a3_t1)
    ground_truth[f_a3_t1] = {"label": "TAMPERED", "doc_type": "aadhaar", "name": a3_data["name"], "id_number": a3_data["id_number"], "tampering_types": ["copy_move_cloning"], "tampered_boxes": [[310, 270, 120, 120]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 18. PAN 3 Genuine
    p3_data = {
        "name": "Zoya Akhtar", "father_name": "Javed Akhtar", "seed": 808,
        "gender": "F", "dob": "14/10/1972", "pan_number": "AZYPA9876Q"
    }
    c_p3, ocr_p3 = render_pan_card(p3_data)
    f_p3 = "mock_pan_06_genuine.png"
    save_card_with_sidecar(c_p3, ocr_p3, f_p3)
    ground_truth[f_p3] = {"label": "GENUINE", "doc_type": "pan", "name": p3_data["name"], "pan_number": p3_data["pan_number"], "tampering_types": [], "expected_verdict": "AUTHENTIC"}

    # 19. PAN 3 Tampered: ELA Hotspot on Photo
    c_p3_t = apply_ela_double_compression(c_p3, (50, 125, 200, 305))
    f_p3_t = "mock_pan_07_tampered_elacompress.jpg"
    save_card_with_sidecar(c_p3_t, ocr_p3, f_p3_t)
    ground_truth[f_p3_t] = {"label": "TAMPERED", "doc_type": "pan", "name": p3_data["name"], "pan_number": p3_data["pan_number"], "tampering_types": ["ela_double_compression"], "tampered_boxes": [[50, 125, 150, 180]], "expected_verdict": "FLAGGED / TAMPERED"}

    # 20. DL 3 Genuine
    dl3_data = {
        "name": "Gurpreet Singh", "seed": 909, "gender": "M", "dob": "08/04/1982",
        "state": "PB-02", "dl_number": "PB-0220080078901", "issue_date": "20/05/2008",
        "expiry_date": "19/05/2028", "vehicle_class": "MCWG, LMV-COMM", "blood_group": "A+ve"
    }
    c_dl3, ocr_dl3 = render_dl_card(dl3_data)
    f_dl3 = "mock_dl_06_genuine.png"
    save_card_with_sidecar(c_dl3, ocr_dl3, f_dl3)
    ground_truth[f_dl3] = {"label": "GENUINE", "doc_type": "dl", "name": dl3_data["name"], "dl_number": dl3_data["dl_number"], "tampering_types": [], "expected_verdict": "AUTHENTIC"}

    gt_path = os.path.join(DATA_DIR, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"[Dataset] Generated {len(ground_truth)} paired cards with calibrated ground truth.")
    return len(ground_truth)


if __name__ == "__main__":
    build_dataset()
