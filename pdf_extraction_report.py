# pdf_extraction_report.py
import pdfplumber
from pathlib import Path

# 👇 change to your actual file
path = r"C:\Users\Kiran Joshi\Downloads\ENVIRONMENTAL_HEALTH_and_SAFETY_POLICY_eb75053f4b.pdf"

MIN_CHARS = 80   # pages with fewer chars than this will be flagged

def analyze(pdf_path: str):
    flagged = []
    per_page = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            chars = len(text)
            words = len(text.split()) if text else 0
            if chars < MIN_CHARS:
                flagged.append(i)
            per_page.append((i, chars, words, text[:160].replace("\n", " ")))
    return per_page, flagged, len(pdf.pages)

if __name__ == "__main__":
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}")
    else:
        per_page, flagged, total = analyze(str(p))
        with_text = sum(1 for _, c, _, _ in per_page if c >= MIN_CHARS)
        print(f"\nPDF: {p.name}")
        print(f"Total pages: {total}")
        print(f"Pages with sufficient text (≥{MIN_CHARS} chars): {with_text}")
        print(f"Flagged pages (low text): {flagged or 'None'}\n")

        # show a quick table-like view
        print("Page | chars | words | snippet")
        print("-" * 70)
        for i, chars, words, snip in per_page[:10]:  # show first 10 pages; tweak as you like
            print(f"{i:>4} | {chars:>5} | {words:>5} | {snip}")
