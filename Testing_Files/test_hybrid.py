import fitz
from PIL import Image
import pytesseract
import io
from difflib import SequenceMatcher
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def remove_duplicate_text(native_text, ocr_text, threshold=0.85):
    native_lines = [line.strip() for line in native_text.splitlines() if line.strip()]
    ocr_lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    unique_ocr_lines = []
    for ocr_line in ocr_lines:
        is_duplicate = False
        for native_line in native_lines:
            similarity = SequenceMatcher(None,ocr_line.lower(),native_line.lower()).ratio()
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_ocr_lines.append(ocr_line)
    combined_lines = native_lines + unique_ocr_lines
    return "\n".join(combined_lines)
pdf = fitz.open("test.pdf")
full_text = ""
for page_number, page in enumerate(pdf):
    print(f"\n--- PAGE {page_number + 1} ---")
    native_text = page.get_text("text")
    print("\nNATIVE TEXT:")
    print(native_text)
    page_ocr_text = ""
    images = page.get_images(full=True)
    print("\nImages found:", len(images))
    for image_index, image_info in enumerate(images):
        xref = image_info[0]
        image_data = pdf.extract_image(xref)
        image_bytes = image_data["image"]
        image = Image.open(io.BytesIO(image_bytes))
        ocr_text = pytesseract.image_to_string(image)
        page_ocr_text += ocr_text + "\n"
        print(f"\nIMAGE {image_index + 1} OCR:")
        print(ocr_text)
    combined_page_text = remove_duplicate_text(native_text,page_ocr_text)
    full_text += combined_page_text + "\n"
print("\n============================")
print("FINAL COMBINED TEXT")
print("============================")
print(full_text)