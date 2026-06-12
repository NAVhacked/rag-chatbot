import chromadb
from sentence_transformers import SentenceTransformer
from pdf_loader import load_pdf, chunk_pages
import sys

# 1. Load and chunk the PDF
pdf_path = sys.argv[1] if len(sys.argv) > 1 else "2.Lee-3D printed programmable(2020).pdf"
print(f"\nLoading: {pdf_path}")
pages  = load_pdf(pdf_path)
chunks = chunk_pages(pages)

# 2. Load embedding model (downloads once, cached after)
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Embed all chunks
print("Embedding chunks...")
texts      = [c["text"]     for c in chunks]
ids        = [c["chunk_id"] for c in chunks]
metadatas  = [{"source": c["source"], "page": c["page_num"]} for c in chunks]

embeddings = model.encode(texts, show_progress_bar=True).tolist()

# 4. Store in ChromaDB
print("\nStoring in ChromaDB...")
client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_docs")

collection.add(
    documents  = texts,
    embeddings = embeddings,
    ids        = ids,
    metadatas  = metadatas,
)

print(f"\nDone! {len(chunks)} chunks stored in ChromaDB.")
print("Database saved to ./chroma_db/")
