#!/usr/bin/env python3
"""
Extract full text from a PDF file and save it as a .txt file with the same base name.

Usage:
    python extract_pdf_text.py <path_to_pdf>
"""

import sys
import os
import PyPDF2


def extract_pdf_text(pdf_path):
    """Extract all text from a PDF file, one page at a time."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        print(f"Number of pages: {num_pages}")
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                print(f"  Warning: no text extracted from page {i + 1}")
    return text


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.isfile(pdf_path):
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.lower().endswith('.pdf'):
        print(f"Warning: file does not have a .pdf extension: {pdf_path}")

    print(f"Extracting text from: {pdf_path}")
    text = extract_pdf_text(pdf_path)

    print(f"Extracted {len(text)} characters ({len(text.split())} words)")

    txt_path = os.path.splitext(pdf_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Text saved to: {txt_path}")


if __name__ == "__main__":
    main()
