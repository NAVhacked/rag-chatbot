import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import sqlite3
from datetime import datetime

model      = SentenceTransformer("all-MiniLM-L6-v2")
client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_docs")

conn = sqlite3.connect("conversation.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        role      TEXT,
        message   TEXT,
        timestamp TEXT
    )
""")
conn.commit()

def save_message(role, message):
    conn.execute("INSERT INTO history (role, message, timestamp) VALUES (?, ?, ?)",
                 (role, message, datetime.now().isoformat()))
    conn.commit()

def get_recent_history(n=6):
    rows = conn.execute(
        "SELECT role, message FROM history ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return list(reversed(rows))

def search_chunks(query, k=3):
    embedding = model.encode([query]).tolist()
    results   = collection.query(query_embeddings=embedding, n_results=k)
    return results["documents"][0]

print("\nRAG Chatbot Ready! (type 'quit' to exit)\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    save_message("user", user_input)

    chunks = search_chunks(user_input, k=3)
    context = "\n\n".join(chunks)

    history = get_recent_history(n=6)
    history_text = "\n".join([f"{r}: {m}" for r, m in history[:-1]])

    prompt = f"""You are a helpful assistant. Answer the question using the context below.
If the answer is not in the context, say you don't know.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

QUESTION: {user_input}
ANSWER:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]

    save_message("assistant", answer)
    print(f"\nAssistant: {answer}\n")
