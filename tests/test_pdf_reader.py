import pytest
import os
import tempfile
from utils.pdf_reader import extract_text, _try_pypdf2, _try_pdfplumber, _try_pymupdf

def test_extract_text_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_text("nonexistent.pdf")

def test_extract_text_no_text():
    with tempfile.NamedTemporaryFile(suffix=".pdf", mode='wb', delete=False) as f:
        f.write(b'%PDF-1.4\n%%EOF')
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError):
            extract_text(temp_path)
    finally:
        os.unlink(temp_path)

def test_text_normalization():
    from utils.comparator import _normalize_answer
    
    assert _normalize_answer("  Hello  World  ") == "hello world"
    assert _normalize_answer("Answer=42!") == "answer=42"
    assert _normalize_answer("Test\n\nAnswer") == "test answer"
