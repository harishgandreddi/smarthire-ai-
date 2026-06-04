import pdfplumber
import re
import os

def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF resume."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_uploaded(uploaded_file) -> str:
    """Extract text from Streamlit uploaded file object."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    text = extract_text_from_pdf(tmp_path)
    os.unlink(tmp_path)
    return text

def parse_candidate_info(text: str) -> dict:
    """Extract basic candidate info using regex patterns."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)

    lines = text.split('\n')
    name = lines[0].strip() if lines else "Unknown"

    return {
        "name": name,
        "email": emails[0] if emails else "Not found",
        "phone": phones[0] if phones else "Not found",
        "raw_text": text
    }
