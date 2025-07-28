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

# Anzahl der Chunks, die bei der Antwort berücksichtigt werden
NUM_CHUNKS = 3

# Text Chunks (deine Infos)
text_chunks = [
    "Jacob Facius ist 26 Jahre alt.",
    "Er ist 26 Jahre alt.",
    "Jacob Facius studiert Wirtschaftsinformatik im Master.",
    "Er studiert Wirtschaftsinformatik im Master.",
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
# 3. Streamlit UI Setup
# =====================
st.set_page_config(page_title="JacobGPT", page_icon="🤖")
st.title("🤖 JacobGPT")

# Initialisiere Session-State für den Chat-Verlauf
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================
# 4. Chat Interface
# =====================
# Eingabefeld als Chat-Eingabe
user_input = st.chat_input("Was möchtest du über Jacob wissen?")

if user_input:
    # Zeige Nutzereingabe im Chat
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Embedding der Anfrage berechnen
    query_embedding = model.encode([user_input])
    D, I = index.search(np.array(query_embedding), k=NUM_CHUNKS)
    retrieved_chunks = [text_chunks[i] for i in I[0]]

    # Mehrere relevante Textabschnitte kombinieren
    combined_text = "\n".join(retrieved_chunks)

    # Prompt für Groq vorbereiten
    prompt = f"Beantworte folgende Frage basierend auf diesen Textausschnitten über Jacob:\n\n{combined_text}\n\nFrage: {user_input}\nAntwort:"
    MODEL_NAME = "llama3-70b-8192"

    try:
        # Modell-Antwort abrufen
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Fehler bei der Anfrage an Groq (Modell {MODEL_NAME}): {e}"

    # Modellantwort speichern
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Chatverlauf anzeigen
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
