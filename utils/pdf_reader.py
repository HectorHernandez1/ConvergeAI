import hashlib
from typing import Optional
import PyPDF2
import pdfplumber
import fitz
from config import settings

def extract_text(pdf_path: str) -> str:
    text = _try_pypdf2(pdf_path)
    
    if text and len(text.strip()) >= 100:
        return text
    
    text = _try_pdfplumber(pdf_path)
    
    if text and len(text.strip()) >= 100:
        return text
    
    text = _try_pymupdf(pdf_path)
    
    if text and len(text.strip()) >= 100:
        return text
    
    raise ValueError(f"Failed to extract meaningful text from {pdf_path}. " 
                     "Ensure the PDF is not a scanned image-only document.")

def _try_pypdf2(pdf_path: str) -> Optional[str]:
    try:
        text_parts = []
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        return "\n".join(text_parts) if text_parts else None
    except Exception:
        return None

def _try_pdfplumber(pdf_path: str) -> Optional[str]:
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        return "\n".join(text_parts) if text_parts else None
    except Exception:
        return None

def _try_pymupdf(pdf_path: str) -> Optional[str]:
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts) if text_parts else None
    except Exception:
        return None

def get_pdf_hash(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
