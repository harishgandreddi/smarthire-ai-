import fitz  # pymupdf
import re
import os

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text.strip()

def extract_text_from_uploaded(uploaded_file) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    text = extract_text_from_pdf(tmp_path)
    os.unlink(tmp_path)
    print(f"\n=== EXTRACTED TEXT ===\n{text[:500]}\n{'='*40}")
    return text

def parse_candidate_info(text: str) -> dict:
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    name = lines[0].strip() if lines else "Unknown"
    return {
        "name": name,
        "email": emails[0] if emails else "Not found",
        "phone": phones[0] if phones else "Not found",
        "raw_text": text
    }