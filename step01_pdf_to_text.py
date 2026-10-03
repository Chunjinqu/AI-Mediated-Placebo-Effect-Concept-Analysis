# -*- coding: utf-8 -*-
"""
step01: PDF -> plain text (Markdown)
Input:  data/pdfs/*.pdf (place the full-text PDFs in this folder)
Output: data/texts/<original filename>.md
Run:    python step01_pdf_to_text.py
"""
import pymupdf4llm
from config import PDF_DIR, TEXT_DIR


def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print("!! No PDF files in data/pdfs — place the full-text PDFs there first.")
        return

    for i, pdf in enumerate(pdfs, 1):
        out = TEXT_DIR / (pdf.stem + ".md")
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        try:
            md = pymupdf4llm.to_markdown(str(pdf))
            out.write_text(md, encoding="utf-8")
            # Symptom of failed extraction: extremely short text
            # (usually a scanned or encrypted PDF)
            if len(md) < 2000:
                print(f"    !! Warning: extracted text is only {len(md)} characters; "
                      f"please open {out.name} and check its completeness.")
        except Exception as e:
            print(f"    !! Failed: {e}")

    print(f"\nDone. Converted {len(pdfs)} files -> {TEXT_DIR}")
    print("Next: spot-check 2-3 .md files (beginning and end), then run step02.")


if __name__ == "__main__":
    main()
