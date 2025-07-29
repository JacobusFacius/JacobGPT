"""
Verbesserte Version von JacobGPT.

Diese Variante optimiert die Chunk‑Auswahl und das Halluzinations‑Handling.
Die folgenden Änderungen wurden umgesetzt:

* Die Keyword‑Bewertung nutzt nun einen deutschen Stemmer, um verschiedene Wortformen besser zu erfassen.
* Die Rangliste der Chunks kombiniert den Embedding‑Score aus FAISS mit der Keyword‑Übereinstimmung (gewichtet).
* Das Prompt wurde präzisiert, damit das Modell nur aus den bereitgestellten Hintergrundinformationen antwortet und bei fehlendem Wissen klar "Nicht gefunden" meldet.
* Nach der Generierung wird die Antwort semantisch mit dem Kontext verglichen. Wenn sie nicht hinreichend zum Kontext passt, wird "Nicht gefunden" ausgegeben.

Hinweis: Dieses Skript setzt voraus, dass `sentence_transformers`, `faiss`, `nltk` und `groq` verfügbar sind.
"""

import re
from typing import List

import numpy as np
import faiss
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# NLTK‑Stemmer importieren und downloaden, um deutsche Wortstämme zu erzeugen
import nltk
from nltk.stem.snowball import GermanStemmer

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# =====================================================
# 1. Vorbereitung
# =====================================================

groq_api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=groq_api_key)

# Anzahl der finalen Chunks, die an das Modell übergeben werden
NUM_FINAL_CHUNKS = 3

# Wissensbasis: Liste an Textabschnitten über Jacob Facius
text_chunks = [
    # Studium und Leidenschaft
    "Jacob Facius studiert Wirtschaftsinformatik im Master an der Otto‑Friedrich‑Universität Bamberg.",
    "Er begeistert sich besonders für datengesteuerte Aspekte und nutzt präzise SQL‑Abfragen, aussagekräftige Power‑BI‑Berichte und moderne KI‑Modelle, um aus Daten neues Wissen zu gewinnen.",
    "Sein Ziel ist es, innovative Lösungen zu entwickeln, die Unternehmen echten Mehrwert bieten.",

    # Werkstudent bei duagon
    "Seit Oktober 2023 arbeitet Jacob als Werkstudent im Bereich Business Intelligence bei duagon.",
    "In dieser Rolle entwickelt er KI‑Modelle, beschafft und analysiert Daten, erstellt Prognosen und baut aussagekräftige Power‑BI‑Berichte.",
    "Die eigenverantwortliche Mitarbeit an Projekten ermöglicht ihm, seine Fähigkeiten stetig zu erweitern und aktiv zur technologischen Zukunft des Unternehmens beizutragen.",

    # Master‑ und Bachelorstudium
    "Parallel zu seiner Tätigkeit bei duagon begann Jacob im Oktober 2023 sein Masterstudium der Wirtschaftsinformatik in Bamberg.",
    "Davor absolvierte er im August 2023 sein Bachelorstudium der Wirtschaftswissenschaften mit Schwerpunkt Wirtschaftsinformatik an der Friedrich‑Alexander‑Universität Erlangen‑Nürnberg mit der Gesamtnote 2,1.",
    "Seine Bachelorarbeit über Fairness in Natural Language Processing wurde mit 1,3 bewertet.",

    # Werkstudent bei Anwalt.de
    "Von Juli 2022 bis September 2023 war Jacob als Werkstudent im Bereich Business Intelligence bei Anwalt.de tätig.",
    "Er führte Analysen durch, erstellte Prognosen und entwickelte Kunden‑Insights.",
    "Dabei beschaffte und bereitete er Daten auf, erstellte interaktive Power‑BI‑Berichte und führte Ad‑hoc‑Analysen mit Python oder Excel durch.",
    "Projekte koordinierte er selbstständig und arbeitete eng mit anderen Abteilungen zusammen.",

    # Service‑Erfahrung bei Anwalt.de
    "Vor seiner BI‑Tätigkeit betreute Jacob ab Oktober 2020 im Service von Anwalt.de Rechtsanwälte.",
    "Er führte sie in ihre Profile ein und gab SEO‑Tipps.",
    "Zu seinen Aufgaben gehörten Kundenakquise, -bindung und -rückgewinnung sowie die Pflege von Profilen und Daten.",
    "Außerdem half er Ratsuchenden, den passenden Anwalt zu finden.",

    # Weitere berufliche Erfahrungen und Reisen
    "Als Customer‑Care‑Agent kümmerte sich Jacob ab Februar 2020 um Forderungsmanagement, passte Profildesigns an und betrieb Datenrecherche.",
    "2019 reiste er ein halbes Jahr durch Indien, Nepal und Indonesien und lernte dabei, mit ungewohnten Situationen umzugehen und einen kühlen Kopf zu bewahren.",

    # Nebentätigkeiten und Schulabschluss
    "Im Oktober 2018 arbeitete Jacob als Barkeeper in der ‚Vintage Bar‘, bereitete Schichten vor, mixte Getränke und bediente Gäste; dabei lernte er, strukturiert und organisiert unter Stress zu arbeiten.",
    "Sein Abitur legte er 2018 am Johannes‑Scharrer‑Gymnasium in Nürnberg ab.",

    # Power‑BI‑ und SQL‑Kompetenz
    "Jacob setzt Power BI professionell ein, um interaktive und visuell ansprechende Berichte zu erstellen, die als Grundlage für Prognosen und strategische Entscheidungen dienen.",
    "Er verfügt über fundierte SQL‑Kenntnisse und arbeitet sowohl im Studium als auch beruflich mit komplexen Datenbank‑Abfragen.",

    # Programmiersprachen und Datenanalyse
    "Er hat sehr gute Kenntnisse in Python und hat bereits Erfahrung mit Pandas, NumPy, Scikit‑learn und Matplotlib gesammelt.",
    "Zudem arbeitet er mit R, um Daten zu filtern, zu analysieren und in strukturierten Grafiken darzustellen.",

    # Weitere Tools und Webtechnologien
    "Jacob besitzt Erfahrung in Tableau zur Datenaufbereitung und Visualisierung.",
    "JavaScript hat er sich selbst beigebracht und in Studienprojekten genutzt.",
    "HTML und CSS beherrscht er sicher und hat seine Kenntnisse während des Studiums vertieft.",

    # Office‑Programme und Sprachen
    "Er ist sehr versiert im Umgang mit Excel, PowerPoint und Word.",
    "Jacob spricht Deutsch und Englisch fließend.",
    "Er verfügt über Grundkenntnisse in Spanisch.",

    # Kontakt und persönliche Daten
    "Jacob Facius wohnt in der Krugstraße 71, 90419 Nürnberg.",
    "Seine E‑Mail‑Adresse lautet info@jacob-facius.de.",
    "Telefon: +49 1637 250148.",

    # Vision und Motto
    "Jacob ist überzeugt, dass der Schlüssel zum Erfolg darin liegt, Daten effektiv zu nutzen und in nützliche Geschäftseinblicke umzuwandeln.",
    "Sein Motto lautet: ‚Datenanalyse und Programmieren faszinieren mich – besonders wenn es dazu beiträgt, unternehmerische Ziele zu erreichen.‘",

    # Motivation und Leidenschaft
    "Ich habe eine besondere Leidenschaft für datengesteuerte Arbeit und liebe es, aus Daten neues Wissen zu generieren und innovative Lösungen zu entwickeln, die Unternehmen echten Mehrwert bieten.",

    # Stärken und Verantwortungsbewusstsein
    "Zu meinen Stärken gehören analytisches Denken und die Fähigkeit, komplexe Daten zu strukturieren und in verständliche Erkenntnisse umzuwandeln.",
    "In meiner Tätigkeit bei duagon entwickle ich eigenverantwortlich KI‑Modelle, analysiere Daten, erstelle Prognosen und baue Power‑BI‑Berichte.",
    "Dabei erweitere ich kontinuierlich meine Fähigkeiten.",

    # Teamarbeit und Kommunikation
    "Ich arbeite gerne im Team und in interdisziplinären Projekten.",
    "Bei Anwalt.de stimmte ich mich regelmäßig abteilungsübergreifend ab.",
    "Ich koordinierte meine Projekte selbstständig.",
    "Diese Erfahrungen zeigen, wie wichtig Kommunikation und Zusammenarbeit sind, um erfolgreiche Ergebnisse zu erzielen.",

    # Lernbereitschaft und Weiterbildung
    "Ich bin sehr lernbereit und erweitere ständig mein Wissen.",
    "Aktuell vertiefe ich meine Kenntnisse in Python und verschiedenen Machine‑Learning‑Bibliotheken wie Pandas, NumPy, Scikit‑learn und Matplotlib.",
    "Außerdem nutze ich R und Tableau zur Datenanalyse.",
    "Ich bilde mich in Webtechnologien wie JavaScript, HTML und CSS weiter.",

    # Umgang mit Herausforderungen
    "Durch eine sechsmonatige Reise durch Indien, Nepal und Indonesien habe ich gelernt, mit ungewohnten Situationen umzugehen und auch unter schwierigen Bedingungen einen kühlen Kopf zu bewahren.",
    "Diese Fähigkeit hilft mir, mich schnell in neue Themen einzuarbeiten und Probleme strukturiert zu lösen.",

    # Zukunftsziele
    "Mein langfristiges Ziel ist es, mich in Business Intelligence und Künstlicher Intelligenz weiterzuentwickeln.",
    "Ich möchte datengestützte Strategien entwerfen, die Unternehmen helfen, bessere Entscheidungen zu treffen und die Zukunft mitzugestalten.",

    # Arbeitsweise unter Stress
    "Ich arbeite strukturiert und organisiere mich auch in stressigen Situationen.",
    "Diese Fähigkeit habe ich unter anderem in meiner Arbeit als Barkeeper entwickelt.",
    "Sie hilft mir, in analytischen Projekten den Überblick zu behalten und effizient zu arbeiten.",

    # Kundenorientierung
    "Meine Zeit im Service von Anwalt.de hat mir gezeigt, wie wichtig es ist, auf Kunden einzugehen und komplexe Informationen verständlich zu vermitteln.",
    "Diese Erfahrung ermöglicht es mir, Ergebnisse aus Datenanalysen klar zu kommunizieren und auf die Bedürfnisse verschiedener Stakeholder einzugehen.",

    # Zertifikate
    "Jacob hat das Zertifikat „Data‑Driven Decisions with PowerBI“ absolviert.",
    "Damit kann er zeigen, wie man mit PowerBI datenbasierte Entscheidungen unterstützt und aussagekräftige Dashboards erstellt.",
    "Mit dem Zertifikat „Data Analysis with Python“ hat Jacob seine Kenntnisse in der Datenanalyse mit Python vertieft.",
    "Er kann Daten mit Bibliotheken wie Pandas, NumPy und Matplotlib professionell auswerten.",
    "Durch das Zertifikat „Databases and SQL for Data Science“ beherrscht Jacob die Grundlagen relationaler Datenbanken und fortgeschrittene SQL‑Abfragen für Data‑Science‑Projekte.",
    "Als zertifizierter Professional Scrum Master kennt sich Jacob mit agilen Methoden und der Leitung von Scrum‑Teams aus.",
    "Er kann Entwicklungsprojekte effizient moderieren.",
    "Das Zertifikat „Mindreading with AI“ zeigt Jacobs Interesse an innovativen KI‑Anwendungen.",
    "Er hat gelernt, wie künstliche Intelligenz genutzt wird, um emotionale oder kognitive Zustände zu interpretieren.",

    # Hobbys
    "Zu Jacobs Hobbys gehört das Bauen und Fliegen von FPV‑Drohnen.",
    "Das Hobby erfordert sowohl Ingenieurwissen als auch Kreativität.",
    "Jacob wandert gerne und verbringt Zeit in der Natur.",
    "Wandern bietet ihm einen Ausgleich zum datengetriebenen Arbeitsalltag.",
    "Eine weitere Freizeitbeschäftigung ist das Erstellen von Webseiten.",
    "Damit lebt Jacob seine Kenntnisse in HTML, CSS und JavaScript auch privat aus.",

    # Persönliche Angaben und Studienfortschritt
    "Jacob ist 26 Jahre alt.",
    "Er befindet sich im letzten Semester seines Masterstudiums der Wirtschaftsinformatik.",
    "Er wird das Masterstudium im November 2025 abschließen.",
    "Er arbeitet nebenbei als Werkstudent im Bereich Business Intelligence bei duagon.",

    # Soft Skills und Arbeitsweise
    "Er gilt als aufgeschlossener, kommunikativer Mensch, der Verantwortung übernimmt.",
    "Er bringt kreative und lösungsorientierte Vorschläge ins Team ein.",
    "Er geht stets freundlich sowie respektvoll mit Kollegen und Mitmenschen um.",
    "Seine Motivation und Begeisterung teilt er gern mit seinem Umfeld.",

    # Aufbau einer Reporting‑Landschaft
    "In seiner beruflichen Praxis hat Jacob eine interne Reporting‑Landschaft bei duagon konzipiert und umgesetzt.",
    "Er erfasste Anforderungen mit verschiedenen Stakeholdern.",
    "Er beschaffte und bereitete Daten auf.",
    "Er visualisierte sie in Power‑BI‑Dashboards.",
    "Dabei vertiefte er seine Kenntnisse in Power BI, Python, R und SQL.",

    # Chatbot‑Projekt und Schulungen
    "Jacob beschäftigt sich intensiv mit Künstlicher Intelligenz und Machine Learning.",
    "Er entwickelte einen unternehmensinternen Chatbot, der Fragen zu firmenbezogenen Dokumenten beantwortet.",
    "Er hält englischsprachige Schulungen zur verantwortungsvollen Nutzung von KI‑Tools im Unternehmen.",

    # Akademische Erfolge und Masterarbeit
    "Das Modul Advanced Data Visualization schloss Jacob mit der Bestnote 1,0 ab.",
    "Seine Masterarbeit trägt den Titel „Determinants of Sharing Sensitive Data with AI Tools in the Workplace: A Privacy Calculus Perspective“.",
    "Die Masterarbeit untersucht, welche Faktoren die Preisgabe sensibler Daten bei der Arbeit mit KI‑Tools beeinflussen.",

    # Kooperation mit dem MIT
    "Im Rahmen eines internationalen Projekts entwickelte Jacob gemeinsam mit dem MIT ein KI‑Modell zur Emotionserkennung bei Pferden.",
    "Dieses Projekt ist ein Beispiel für seine Begeisterung für innovative KI‑Anwendungen.",
]

# =====================================================
# 2. Embeddings & Index
# =====================================================

embedding_model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(embedding_model_name)

# Embeddings erstellen und normalisieren (wichtig für Cosinus‑Ähnlichkeit)
embeddings = model.encode(text_chunks, normalize_embeddings=True)
dimension = embeddings.shape[1]

# FAISS‑Index mit Inner Product (Cosine Similarity) erstellen und befüllen
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# =====================================================
# 3. Hilfsfunktionen
# =====================================================

stemmer = GermanStemmer()

def keyword_score(chunk: str, query: str) -> int:
    """Berechnet eine robuste Keyword‑Übereinstimmung basierend auf Wortstämmen."""
    query_tokens = re.findall(r"\w+", query.lower())
    chunk_tokens = set(re.findall(r"\w+", chunk.lower()))
    # Wortstämme generieren, um verschiedene Wortformen abzudecken
    query_stems = [stemmer.stem(token) for token in query_tokens]
    chunk_stems = {stemmer.stem(token) for token in chunk_tokens}
    return sum(1 for stem in query_stems if stem in chunk_stems)

def retrieve_best_chunks(user_query: str, top_candidates: int = 10, final_k: int = 3) -> List[str]:
    query_embedding = model.encode([user_query], normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding), k=top_candidates)
    # Erstelle Liste aus Kandidaten mit ihren Indizes und Embedding‑Scores
    candidates = []
    for rank, idx in enumerate(I[0]):
        embedding_score = float(D[0][rank])
        candidates.append((idx, embedding_score))
    # Gewichtungsfaktor für Keyword‑Übereinstimmung
    weight_keyword = 0.2
    ranked = sorted(
        candidates,
        key=lambda x: x[1] + weight_keyword * keyword_score(text_chunks[x[0]], user_query),
        reverse=True
    )
    return [text_chunks[idx] for idx, _ in ranked[:final_k]]

def is_answer_supported(answer: str, retrieved_chunks: List[str], threshold: float = 0.5) -> bool:
    """
    Prüft, ob die generierte Antwort semantisch durch die bereitgestellten Chunks unterstützt wird.
    Wenn die Cosinus‑Ähnlichkeit zwischen Antwort und Kontext unter dem Schwellenwert liegt, gilt sie als nicht unterstützt.
    """
    if not answer or not retrieved_chunks:
        return False
    answer_embedding = model.encode([answer], normalize_embeddings=True)
    context_embedding = model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)
    similarity = float(np.dot(answer_embedding, context_embedding.T))
    return similarity >= threshold

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
        "Antworte ausschließlich und wörtlich basierend auf den unten aufgeführten Hintergrundinformationen. "
        "Wenn du die Frage nicht direkt mithilfe dieser Informationen beantworten kannst, antworte mit 'Nicht gefunden'. "
        "Füge keine zusätzlichen Details oder Vermutungen hinzu.\n\n"
        "=== Hintergrundinformationen ===\n"
        f"{chr(10).join(retrieved_chunks)}\n\n"
        "=== Gesprächsverlauf ===\n"
        f"{conversation_history}"
        f"Frage: {user_input}\n"
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
            temperature=0.0,
            max_tokens=256
        )
        answer = response.choices[0].message.content.strip()
        # Überprüfen, ob die Antwort semantisch durch die bereitgestellten Chunks gedeckt ist
        if not is_answer_supported(answer, retrieved_chunks):
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
