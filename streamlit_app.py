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
    "Weitere Infos unter jacob-facius.de.",
    "Ich bin Jacob und absolviere derzeit das Masterstudium in Wirtschaftsinformatik an der Universität Bamberg.",
"Ich interessiere mich besonders für datengetriebene Themen wie SQL-Abfragen, Power BI und KI-Modelle.",
"Ich entwickle gerne innovative und effektive Lösungen mit echtem Mehrwert.",
"Ich konnte meine Datenkompetenz bereits in zahlreichen Projekten als Werkstudent einbringen.",
"Seit 10/2023: Werkstudent Business Intelligence bei duagon.",
"Bei duagon: Entwicklung von KI-Modellen, Datenanalyse, Prognosen, Power BI-Berichte.",
"Seit 10/2023: Master Wirtschaftsinformatik an der Universität Bamberg.",
"08/2023: Bachelor Wirtschaftswissenschaften mit Schwerpunkt Wirtschaftsinformatik an der FAU Erlangen-Nürnberg.",
"Bachelorarbeit über Fairness in NLP, Note: 1,3.",
"07/2022–09/2023: Werkstudent Business Intelligence bei anwalt.de.",
"Bei anwalt.de: Analysen, Prognosen, Kunden-Insights, Power BI, Python, Excel, abteilungsübergreifende Zusammenarbeit.",
"10/2020–06/2022: Werkstudent Service bei anwalt.de – Kundenbetreuung, SEO, Akquise, Support.",
"02/2020–10/2020: Customer Care Agent bei anwalt.de – Forderungsmanagement, Datenrecherche, Profilgestaltung.",
"06/2019: Reise durch Indien, Nepal und Indonesien – interkulturelle Kompetenz und Stressresistenz.",
"10/2018–06/2019: Barkeeper – Organisation, Struktur, Stressresistenz.",
"06/2018: Abitur am Johannes-Scharrer-Gymnasium Nürnberg.",
"Power BI – Erstellung interaktiver, visueller Berichte als Basis für Prognosen und Entscheidungen.",
"SQL – Komplexe Abfragen in Studium und Beruf zur Datenextraktion und Analyse.",
"Python – Erfahrung mit Pandas, NumPy, Scikit-learn, Matplotlib.",
"R / RStudio – Daten filtern, analysieren und visualisieren.",
"Tableau – Datenaufbereitung und Visualisierung im Studium.",
"JavaScript – autodidaktisch erlernt, gelegentlich im Studium verwendet.",
"HTML & CSS – Kenntnisse durch eigene Webseitenentwicklung.",
"MS Office – Excel, PowerPoint und Word sicher in Studium und Beruf genutzt.",
"Deutsch – 5 von 5.",
"Englisch – 5 von 5.",
"Spanisch – 2 von 5.",
"Name: Jacob Facius.",
"Adresse: Krugstraße 71, 90419 Nürnberg.",
"E-Mail: info@jacob-facius.de.",
"Telefon: +49 1637 250148.",
"Kontaktformular vorhanden auf Webseite."
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
user_input = st.chat_input("Hallo, ich bin JacobGPT. Was möchtest du über mich wissen?")

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
