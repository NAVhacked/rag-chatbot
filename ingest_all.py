import chromadb
from sentence_transformers import SentenceTransformer
from pdf_loader import load_pdf, chunk_pages
from pathlib import Path

model      = SentenceTransformer("all-MiniLM-L6-v2")
client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_docs")

pdfs = list(Path(".").glob("*.pdf"))
print(f"Found {len(pdfs)} PDFs\n")

for pdf in pdfs:
    print(f"Processing: {pdf.name}")
    try:
        pages  = load_pdf(str(pdf))
        chunks = chunk_pages(pages)

        texts      = [c["text"]     for c in chunks]
        ids        = [c["chunk_id"] for c in chunks]
        metadatas  = [{"source": c["source"], "page": c["page_num"]} for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
        print(f"  ✓ {len(chunks)} chunks stored\n")

    except Exception as e:
        print(f"  ✗ Error: {e}\n")

total = collection.count()
print(f"Done! Total chunks in ChromaDB: {total}")
