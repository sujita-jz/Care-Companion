import os

try:
    import PyPDF2

    PYPDF_AVAILABLE = True
except ImportError:
    PyPDF2 = None
    PYPDF_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
import io


def extract_text_from_pdf(pdf_path: str) -> str:
    if not PYPDF_AVAILABLE:
        return ""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def extract_text_from_image_ocr_placeholder(image_path: str) -> str:
    # Placeholder for OCR - in production use pytesseract or Gemini Vision itself
    # Returning empty; Gemini will do vision directly
    return ""


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'))


def is_pdf_file(filename: str) -> bool:
    return filename.lower().endswith('.pdf')
