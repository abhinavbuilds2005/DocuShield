import os
import win32com.client

ppt_path = r"d:\Document detector\DocuShield_AI_SIH2026_Final_Presentation.pptx"
pdf_path = r"d:\Document detector\DocuShield_AI_SIH2026_Final_Presentation.pdf"
out_dir = r"d:\Document detector\reports\presentation_slides"
os.makedirs(out_dir, exist_ok=True)

ppt_app = win32com.client.Dispatch("PowerPoint.Application")
presentation = ppt_app.Presentations.Open(ppt_path, False, False, False)

# Save as PDF (Format type 32 = ppSaveAsPDF)
presentation.SaveAs(pdf_path, 32)
print(f"Exported PDF: {pdf_path}")

# Export each slide as PNG
for i in range(1, presentation.Slides.Count + 1):
    slide = presentation.Slides(i)
    slide_out = os.path.join(out_dir, f"Slide_{i}.png")
    slide.Export(slide_out, "PNG", 1920, 1080)
    print(f"Exported: {slide_out}")

presentation.Close()
ppt_app.Quit()
print("All slides exported successfully.")
