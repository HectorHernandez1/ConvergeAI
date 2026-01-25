import pytest
import os
import tempfile
from utils.pdf_reader import extract_text, _try_pypdf2, _try_pdfplumber, _try_pymupdf, _try_html_extraction

def test_extract_text_missing_file():
    with pytest.raises(ValueError):
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

def test_extract_text_from_html():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a paragraph with some text. It needs to be long enough to meet the minimum character requirement for text extraction. The minimum is one hundred characters.</p>
    <p>Another paragraph with more content. This is the second paragraph and it also contains a significant amount of text to ensure the extraction passes the validation check.</p>
    <p>Third paragraph with additional content to make sure we have enough text. This paragraph is included to ensure the total character count exceeds the minimum threshold of one hundred characters.</p>
    <script>
        var x = 5;
    </script>
    <style>
        body { color: blue; }
    </style>
</body>
</html>"""
    
    with tempfile.NamedTemporaryFile(suffix=".html", mode='w', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name
    
    try:
        text = extract_text(temp_path)
        assert "Main Heading" in text
        assert "paragraph with some text" in text
        assert "var x = 5" not in text  # Script content should be removed
        assert "color: blue" not in text  # Style content should be removed
    finally:
        os.unlink(temp_path)

def test_html_extraction_empty():
    html_content = """<!DOCTYPE html>
<html>
<body>
    <script>console.log("test");</script>
    <style>body { margin: 0; }</style>
</body>
</html>"""
    
    with tempfile.NamedTemporaryFile(suffix=".html", mode='w', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name
    
    try:
        text = _try_html_extraction(temp_path)
        # Should return None or empty string since only script/style elements exist
        assert text is None or len(text.strip()) == 0
    finally:
        os.unlink(temp_path)

def test_html_extraction_with_tables():
    html_content = """<!DOCTYPE html>
<html>
<body>
    <h1>Employee Directory</h1>
    <p>This is the employee directory table. It contains information about all employees in the organization including their names, ages, and departments. The table below provides detailed information.</p>
    <table>
        <tr><th>Name</th><th>Age</th><th>Department</th></tr>
        <tr><td>John Smith</td><td>25</td><td>Engineering</td></tr>
        <tr><td>Jane Doe</td><td>30</td><td>Marketing</td></tr>
        <tr><td>Bob Johnson</td><td>35</td><td>Sales</td></tr>
        <tr><td>Alice Brown</td><td>28</td><td>HR</td></tr>
    </table>
    <p>The table shows four employees from different departments. All of them have various ages and roles within the company.</p>
</body>
</html>"""
    
    with tempfile.NamedTemporaryFile(suffix=".html", mode='w', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name
    
    try:
        text = extract_text(temp_path)
        assert "Name" in text
        assert "Age" in text
        assert "John Smith" in text
        assert "Jane Doe" in text
        assert "25" in text
        assert "30" in text
    finally:
        os.unlink(temp_path)
