import chromadb
from sentence_transformers import SentenceTransformer

model  = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_docs")

print("\nRAG Search Ready! Type your question (or 'quit' to exit)\n")

while True:
    query = input("You: ").strip()
    if query.lower() in ("quit", "exit"):
        break

    embedding = model.encode([query]).tolist()
    results   = collection.query(query_embeddings=embedding, n_results=3)

    print("\nTop 3 relevant chunks:\n")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"[{i+1}] Page {meta['page']} — {meta['source']}")
        print(f"    {doc[:300]}...")
        print()
