# text_protection.py
import re

PATTERNS = {
    "URL": re.compile(r"\b((https?:\/\/|www\.)[^\s<>\"']+)", re.IGNORECASE),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(\+?\d{1,3}[\s-]?)?\d{10}\b"),
    "ID": re.compile(r"\b\d{6,}\b"),
    "DATE": re.compile(
        r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"
        r"|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}"
        r"|\d{1,2}:\d{2}(\s?[APap][Mm])?)\b"
    ),
    "CURRENCY": re.compile(
        r"\b(₹|Rs\.?|INR|\$|USD)\s?\d{1,3}(,\d{3})*(\.\d+)?\b"
    ),
}

def protect_text(text: str):
    mapping = {}
    protected = text
    for key, pattern in PATTERNS.items():
        matches = list(pattern.finditer(protected))
        for idx, m in enumerate(matches):
            ph = f"<{key}_{idx}>"
            mapping[ph] = m.group(0)
            protected = protected.replace(m.group(0), ph)
    return protected, mapping

def restore_text(text: str, mapping: dict):
    for ph, original in mapping.items():
        text = text.replace(ph, original)
    return text
