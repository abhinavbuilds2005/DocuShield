"""
Tests and calibrates watermark rendering and tampering detection
"""
from PIL import Image, ImageDraw, ImageFont
import math

def draw_clean_watermark(img: Image.Image):
    """Draws an authoritative single diagonal banner watermark across the card."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)

    # Top and bottom safety banners
    odraw.rectangle([0, 0, w, 22], fill=(190, 18, 60, 255))
    odraw.text((w // 2, 11), "SAMPLE / MOCK — NOT A REAL GOVERNMENT DOCUMENT — SIH 2026 DEMO",
               fill=(255, 255, 255, 255), anchor="mm")
    
    odraw.rectangle([0, h - 20, w, h], fill=(190, 18, 60, 255))
    odraw.text((w // 2, h - 10), "SYNTHETIC FICTIONAL TEST DATASET — ETHICAL AI RESEARCH ONLY",
               fill=(255, 255, 255, 255), anchor="mm")

    # Single elegant centered diagonal watermark banner
    odraw.line([(0, h - 60), (w, 60)], fill=(226, 232, 240, 140), width=40)
    odraw.text((w // 2, h // 2), "SPECIMEN — NOT AN OFFICIAL GOVERNMENT ID",
               fill=(148, 163, 184, 180), anchor="mm")

    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
