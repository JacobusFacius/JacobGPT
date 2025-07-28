import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
import json

# =====================
# 1. Text vorbereiten
# =====================
text_chunks = [
    "Jacob Facius ist 26 Jahre alt und studiert Wirtschaftsinformatik im Master.",
    "Er arbeitet bei Duagon im Bereich Business Intelligence.",
    "Er kennt sich mit Power BI, Python, R, SQL und Machine Learning aus.",
    "Er hat einen Chatbot für interne Dokumente gebaut.",
    "Seine Masterarbeit behandelt Datenschutz bei KI im Unternehmen.",
    "Er entwickelte mit dem MIT ein KI-Modell zur Emotionserkennung bei Pferden.",
    "Er schließt sein Studium im November 2025 ab.",
    "Weitere Infos unter jacob-facius.de."
]

# =====================
# 2. Embeddings
# =====================
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(text_chunks)

# FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# =====================
# 3. Suche & Chat
# =====================
st.title("🤖 JacobGPT – Bewerbungschatbot")
query = st.text_input("Was möchtest du über Jacob wissen?")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    best_chunk = text_chunks[I[0][0]]

    # Optional: Wenn Mistral über Ollama läuft
    try:
        prompt = f"Beantworte folgende Frage basierend auf diesem Textausschnitt:\n\nText: {best_chunk}\n\nFrage: {query}\nAntwort:"
        response = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        answer = response.stdout.strip()
    except Exception:
        answer = f"(Kein Mistral gefunden – Rückgabe aus Text:)\n\n{best_chunk}"

    st.markdown(f"**Antwort:** {answer}")
