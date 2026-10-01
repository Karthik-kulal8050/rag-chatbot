from app.parse import parse_pdf
from pathlib import Path
import re


def chunk_pages(pages, chunk_size=800, overlap=150):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end]
            if not is_low_value(piece):
                chunks.append({
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                })
            start += chunk_size - overlap
    return chunks

def is_low_value(text):
    low_value_patterns = [
        r"question bank",
        r"^figure \d+",
        r"^table \d+",
    ]
    text_lower = text.lower().strip()
    for pattern in low_value_patterns:
        if re.search(pattern, text_lower):
            return True
    # mostly numbers/symbols, likely a table or figure, not prose
    letters = sum(c.isalpha() for c in text)
    if len(text) > 0 and letters / len(text) < 0.5:
        return True
    return False

if __name__ == "__main__":
    all_chunks = []
    for pdf in Path("data").glob("*.pdf"):
        all_chunks.extend(chunk_pages(parse_pdf(pdf)))

    print("Total chunks:", len(all_chunks))
    print(all_chunks[10])