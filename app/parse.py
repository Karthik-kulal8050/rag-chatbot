import fitz
from pathlib import Path
import re

def parse_pdf(path):
    pages = []
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        text = clean_text(page.get_text())
        if len(text) < 30:
            continue
        pages.append({
            "text": text,
            "source": Path(path).name,
            "page": i + 1,
        })
    doc.close()
    return pages

def clean_text(text):
    text = re.sub(r"vtucircle\.com", "",text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"-\n", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


if __name__ == "__main__":
    for pdf in Path("data").glob("*.pdf"):
        pages = parse_pdf(pdf)
        print(pdf.name, "->", len(pages), "pages")