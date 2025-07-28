import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# =====================
# 1. Vorbereitung
# =====================

# API-Key aus Streamlit secrets laden
groq_api_key = st.secrets["GROQ_API_KEY"]

# Groq-Client initialisieren
client = Groq(api_key=groq_api_key)

# Text Chunks (deine Infos)
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
# 2. Embeddings & Index
# =====================
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(text_chunks)
dimension = embeddings.shape[1]

# FAISS Index erstellen und Embeddings hinzufügen
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# =====================
# 3. Streamlit UI
# =====================
st.title("🤖 JacobGPT – Bewerbungschatbot")
query = st.text_input("Was möchtest du über Jacob wissen?")

# =====================
# 4. Anfrage verarbeiten
# =====================
if query:
    # Embedding der Anfrage berechnen
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    best_chunk = text_chunks[I[0][0]]

    # Prompt für Groq vorbereiten
    prompt = f"Beantworte folgende Frage basierend auf diesem Textausschnitt:\n\nText: {best_chunk}\n\nFrage: {query}\nAntwort:"

    MODEL_NAME = "llama3-70b-8192"  # Aktualisiertes Modell

    try:
        # Chat Completion von Groq holen
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME
        )

        answer = chat_completion.choices[0].message.content.strip()

    except Exception as e:
        answer = f"Fehler bei der Anfrage an Groq (Modell {MODEL_NAME}): {e}"

    # Antwort anzeigen
    st.markdown(f"**Antwort:** {answer}")
