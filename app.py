import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.caption("Powered by your research papers")

@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("rag_docs")
    return model, collection

model, collection = load_models()

api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)

conn = sqlite3.connect("conversation.db", check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT, message TEXT, timestamp TEXT)""")
conn.commit()

def save_message(role, message):
    conn.execute("INSERT INTO history (role, message, timestamp) VALUES (?, ?, ?)",
                 (role, message, datetime.now().isoformat()))
    conn.commit()

def get_history():
    return conn.execute("SELECT role, message FROM history ORDER BY id").fetchall()

def search_chunks(query, k=3):
    embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=k)
    return results["documents"][0]

for role, message in get_history():
    with st.chat_message(role):
        st.write(message)

if prompt := st.chat_input("Ask anything about your research papers..."):
    with st.chat_message("user"):
        st.write(prompt)
    save_message("user", prompt)

    chunks = search_chunks(prompt, k=3)
    context = " ".join([c[:500] for c in chunks])
    history = get_history()
    history_text = " ".join([f"{r}: {m[:200]}" for r, m in history[-4:]])

    full_prompt = "You are a helpful assistant. Answer using the context below. If the answer is not in the context, say you dont know. CONTEXT: " + context + " CONVERSATION HISTORY: " + history_text + " QUESTION: " + prompt + " ANSWER:"

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": full_prompt}]
            )
            answer = response.choices[0].message.content
            st.write(answer)
            save_message("assistant", answer)
