import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import re
from typing import List

# =====================================================
# 1. Vorbereitung
# =====================================================

# API‑Key aus Streamlit secrets laden
groq_api_key = st.secrets["GROQ_API_KEY"]

# Groq‑Client initialisieren
client = Groq(api_key=groq_api_key)

# Anzahl der finalen Chunks, die an das Modell übergeben werden
NUM_FINAL_CHUNKS = 3

# Wissensbasis: Liste an Textabschnitten über Jacob Facius
text_chunks = [
    "Jacob Facius ist 26 Jahre alt und kommt aus Nürnberg.",
    "Er studiert Wirtschaftsinformatik im Master an der Universität Bamberg und wird sein Studium im November 2025 abschließen.",
    "Jacob arbeitet seit Oktober 2023 als Werkstudent im Bereich Business Intelligence bei duagon und entwickelt dort KI‑Modelle, analysiert Daten, erstellt Prognosen und fertigt Power‑BI‑Berichte an.",
    "Zuvor war er als Werkstudent bei anwalt.de tätig (Juli 2022 – September 2023) und beschäftigte sich dort mit Analysen, Prognosen und Kunden‑Insights (Python, Power BI, Excel).",
    "In seiner Bachelorarbeit untersuchte er Fairness in NLP und erhielt die Note 1,3.",
    "Jacob besitzt Kenntnisse in Power BI, Python (Pandas, NumPy, Scikit‑learn, Matplotlib), R / RStudio, SQL, Tableau, JavaScript, HTML/CSS und verwendet MS Office sicher.",
    "Er spricht fließend Deutsch und Englisch sowie etwas Spanisch.",
    "Er hat einen Chatbot für interne Dokumente gebaut und seine Masterarbeit behandelt Datenschutz bei KI im Unternehmen.",
    "Gemeinsam mit dem MIT entwickelte er ein KI‑Modell zur Emotionserkennung bei Pferden.",
    "Weitere Informationen stehen auf jacob-facius.de.",
    "Sein Name ist Jacob Facius und seine Kontaktdaten sind: Krugstraße 71, 90419 Nürnberg, E‑Mail: info@jacob-facius.de, Telefon: +49 1637 250148."
]

# =====================================================
# 2. Embeddings & Index
# =====================================================

# Embedding‑Modell wählen. "BAAI/bge-base-en-v1.5" ist ein starkes Modell; für
# deutsche Daten kann alternativ "BAAI/bge-large-de" genutzt werden (Internet
# erforderlich).
embedding_model_name = "BAAI/bge-large-de"
model = SentenceTransformer(embedding_model_name)

# Embeddings erstellen und normalisieren (Wichtig für Cosinus‑Ähnlichkeit)
embeddings = model.encode(text_chunks, normalize_embeddings=True)
dimension = embeddings.shape[1]

# FAISS‑Index mit Inner Product (Cosine Similarity) erstellen und befüllen
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# =====================================================
# 3. Hilfsfunktionen
# =====================================================

def keyword_score(chunk: str, query: str) -> int:
    """Berechnet, wie viele Wörter aus der Anfrage im Chunk vorkommen."""
    query_tokens = re.findall(r"\w+", query.lower())
    chunk_tokens = set(re.findall(r"\w+", chunk.lower()))
    return sum(1 for token in query_tokens if token in chunk_tokens)

def retrieve_best_chunks(user_query: str, top_candidates: int = 10, final_k: int = 3) -> List[str]:
    """
    Ermittelt die Top‑k relevantesten Chunks für die Anfrage.

    Zunächst werden per FAISS die `top_candidates` ähnlichsten Chunks geholt.
    Anschließend sortieren wir diese anhand der Keyword-Übereinstimmung und
    geben die `final_k` besten zurück.
    """
    query_embedding = model.encode([user_query], normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding), k=top_candidates)
    candidates = [text_chunks[i] for i in I[0]]
    ranked = sorted(candidates, key=lambda c: keyword_score(c, user_query), reverse=True)
    return ranked[:final_k]

# =====================================================
# 4. Streamlit-UI Setup
# =====================================================

st.set_page_config(page_title="JacobGPT", page_icon="🤖")
st.title("JacobGPT🤖")

# Chatverlauf in Session State speichern
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================================
# 5. Chat Interface
# =====================================================

user_input = st.chat_input("Hallo, ich bin JacobGPT. Was möchtest du über mich wissen?")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    retrieved_chunks = retrieve_best_chunks(user_input, top_candidates=10, final_k=NUM_FINAL_CHUNKS)
    conversation_history = ""
    for message in st.session_state.chat_history[-6:]:
        role = "Benutzer" if message["role"] == "user" else "JacobGPT"
        conversation_history += f"{role}: {message['content']}\n"
    prompt = (
        "Du bist JacobGPT – ein virtueller Assistent, der Jacob bei Bewerbungen unterstützt. "
        "Antworte präzise und nutze ausschließlich die unten aufgeführten Hintergrundinformationen. "
        "Sollte keine Information vorhanden sein, antworte mit 'Nicht gefunden'. \n\n"
        "=== Hintergrundinformationen ===\n"
        f"{chr(10).join(retrieved_chunks)}\n\n"
        "=== Gesprächsverlauf ===\n"
        f"{conversation_history}"
        "Frage: {user_input}\n"
        "Antwort:"
    )
    MODEL_NAME = "llama3-70b-8192"
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Du bist JacobGPT, ein hilfreicher Assistent basierend auf Jacobs Profil."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=256
        )
        answer = response.choices[0].message.content.strip()
        if keyword_score(answer, " ".join(retrieved_chunks)) == 0:
            answer = "Nicht gefunden."
    except Exception as e:
        answer = f"Fehler bei der Anfrage an Groq (Modell {MODEL_NAME}): {e}"
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# =====================================================
# 6. Chatverlauf anzeigen
# =====================================================

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
