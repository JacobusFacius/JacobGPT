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
# Diese Chunks wurden aus der offiziellen Website jacob‑facius.de zusammengestellt.
text_chunks = [
    # Leidenschaft und Studium
    "Jacob Facius studiert Wirtschaftsinformatik im Master an der Otto‑Friedrich‑Universität Bamberg. Er begeistert sich besonders für datengesteuerte Aspekte und nutzt präzise SQL‑Abfragen, aussagekräftige Power‑BI‑Berichte und moderne KI‑Modelle, um aus Daten neues Wissen zu gewinnen. Sein Ziel ist es, innovative Lösungen zu entwickeln, die Unternehmen echten Mehrwert bieten.",

    # Werkstudent bei duagon
    "Seit Oktober 2023 arbeitet Jacob als Werkstudent im Bereich Business Intelligence bei duagon. In dieser Rolle entwickelt er KI‑Modelle, beschafft und analysiert Daten, erstellt Prognosen und baut aussagekräftige Power‑BI‑Berichte. Die eigenverantwortliche Mitarbeit an Projekten ermöglicht ihm, seine Fähigkeiten stetig zu erweitern und aktiv zur technologischen Zukunft des Unternehmens beizutragen.",

    # Master‑ und Bachelorstudium
    "Parallel zu seiner Tätigkeit bei duagon begann Jacob im Oktober 2023 sein Masterstudium der Wirtschaftsinformatik in Bamberg. Davor absolvierte er im August 2023 sein Bachelorstudium der Wirtschaftswissenschaften mit Schwerpunkt Wirtschaftsinformatik an der Friedrich‑Alexander‑Universität Erlangen‑Nürnberg mit der Gesamtnote 2,1. Seine Bachelorarbeit über Fairness in Natural Language Processing wurde mit 1,3 bewertet.",

    # Werkstudent bei Anwalt.de
    "Von Juli 2022 bis September 2023 war Jacob als Werkstudent im Bereich Business Intelligence bei Anwalt.de tätig. Er führte Analysen durch, erstellte Prognosen und entwickelte Kunden‑Insights. Dabei beschaffte und bereitete er Daten auf, erstellte interaktive Power‑BI‑Berichte und führte Ad‑hoc‑Analysen mit Python oder Excel durch; Projekte koordinierte er selbstständig und arbeitete eng mit anderen Abteilungen zusammen.",

    # Service‑Erfahrung bei Anwalt.de
    "Vor seiner BI‑Tätigkeit betreute Jacob ab Oktober 2020 im Service von Anwalt.de Rechtsanwälte, führte sie in ihre Profile ein und gab SEO‑Tipps. Zu seinen Aufgaben gehörten Kundenakquise, -bindung und -rückgewinnung sowie die Pflege von Profilen und Daten. Außerdem half er Ratsuchenden, den passenden Anwalt zu finden.",

    # Weitere berufliche Erfahrungen und Reisen
    "Als Customer‑Care‑Agent kümmerte sich Jacob ab Februar 2020 um Forderungsmanagement, passte Profildesigns an und betrieb Datenrecherche. 2019 reiste er ein halbes Jahr durch Indien, Nepal und Indonesien und lernte dabei, mit ungewohnten Situationen umzugehen und einen kühlen Kopf zu bewahren.",

    # Nebentätigkeiten und Schulabschluss
    "Im Oktober 2018 arbeitete Jacob als Barkeeper in der ‚Vintage Bar‘, bereitete Schichten vor, mixte Getränke und bediente Gäste; dabei lernte er, strukturiert und organisiert unter Stress zu arbeiten. Sein Abitur legte er 2018 am Johannes‑Scharrer‑Gymnasium in Nürnberg ab.",

    # Power‑BI‑ und SQL‑Kompetenz
    "Jacob setzt Power BI professionell ein, um interaktive und visuell ansprechende Berichte zu erstellen, die als Grundlage für Prognosen und strategische Entscheidungen dienen. Er verfügt über fundierte SQL‑Kenntnisse und arbeitet sowohl im Studium als auch beruflich mit komplexen Datenbank‑Abfragen.",

    # Programmiersprachen und Datenanalyse
    "Er hat sehr gute Kenntnisse in Python, und hat bereits Erfahrung mit Pandas, NumPy, Scikit‑learn und Matplotlib gesammelt. Zudem arbeitet er mit R, um Daten zu filtern, zu analysieren und in strukturierten Grafiken darzustellen.",

    # Weitere Tools und Webtechnologien
    "Jacob besitzt Erfahrung in Tableau zur Datenaufbereitung und Visualisierung. JavaScript hat er sich selbst beigebracht und in Studienprojekten genutzt; HTML und CSS beherrscht er sicher und hat seine Kenntnisse während des Studiums vertieft.",

    # Office‑Programme und Sprachen
    "Er ist sehr versiert im Umgang mit Excel, PowerPoint und Word. Jacob spricht Deutsch und Englisch fließend und verfügt über Grundkenntnisse in Spanisch.",

    # Kontakt und persönliche Daten
    "Kontaktdaten: Jacob Facius, Krugstraße 71, 90419 Nürnberg. E‑Mail: info@jacob-facius.de. Telefon: +49 1637 250148.",

    # Vision und Motto
    "Jacob ist überzeugt, dass der Schlüssel zum Erfolg darin liegt, Daten effektiv zu nutzen und in nützliche Geschäftseinblicke umzuwandeln. Sein Motto lautet: ‚Datenanalyse und Programmieren faszinieren mich – besonders wenn es dazu beiträgt, unternehmerische Ziele zu erreichen.‘"
]

# =====================================================
# 2. Embeddings & Index
# =====================================================

# Embedding‑Modell wählen. "BAAI/bge-base-en-v1.5" ist ein starkes Modell; für
# deutsche Daten kann alternativ "BAAI/bge-large-de" genutzt werden (Internet
# erforderlich).
embedding_model_name = "all-MiniLM-L6-v2"
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
    Anschließend sortieren wir diese anhand der Keyword‑Übereinstimmung und
    geben die `final_k` besten zurück.
    """
    query_embedding = model.encode([user_query], normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding), k=top_candidates)
    candidates = [text_chunks[i] for i in I[0]]
    ranked = sorted(candidates, key=lambda c: keyword_score(c, user_query), reverse=True)
    return ranked[:final_k]

# =====================================================
# 4. Streamlit‑UI Setup
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
