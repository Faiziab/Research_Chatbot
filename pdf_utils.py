# pdf_utils.py
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from the provided PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # Extract text from each page
    return text
