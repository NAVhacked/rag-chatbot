import fitz
import os
from pathlib import Path
from typing import List, Dict, Any

def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text").strip()
            if not text:
                print(f"  [warning] page {page_index + 1} has no extractable text")
                continue
            pages.append({
                "text": text,
                "page_num": page_index + 1,
                "source": path.name,
                "char_count": len(text),
            })
    print(f"Loaded {len(pages)} pages from '{path.name}'")
    return pages

def chunk_pages(pages, chunk_size=500, chunk_overlap=50):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        chunk_index = 0
        while start < len(text):
            chunk_text = text[start:start + chunk_size].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": f"{page['source']}::p{page['page_num']}::c{chunk_index}",
                    "page_num": page["page_num"],
                    "source": page["source"],
                })
                chunk_index += 1
            start += chunk_size - chunk_overlap
    print(f"Created {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <your_file.pdf>")
        sys.exit(1)
    pages = load_pdf(sys.argv[1])
    chunks = chunk_pages(pages)
    for chunk in chunks[:2]:
        print(f"\nChunk ID : {chunk['chunk_id']}")
        print(f"Page     : {chunk['page_num']}")
        print(f"Text     : {chunk['text'][:200]}...")
