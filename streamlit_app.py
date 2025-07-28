import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ==============================
# 1. Textbasis direkt im Code
# ==============================

text = """
Sehr geehrte Damen und Herren,

mein Name ist Jacob Facius, ich bin 26 Jahre alt und befinde mich im letzten Semester meines Masterstudiums der Wirtschaftsinformatik. Neben meinem Studium arbeite ich als Werkstudent im Bereich Business Intelligence bei Duagon. 

Ich bin ein aufgeschlossener, kommunikativer Mensch, der gerne Verantwortung übernimmt und seine kreativen und lösungsorientierten Vorschläge gerne ins Team einbringt. Ein freundlicher und respektvoller Umgang mit meinen Kollegen und Mitmenschen ist für mich eine Selbstverständlichkeit. Gerne lasse ich mein Umfeld an meiner Motivation teilhaben und begeistere mich und andere für jede Herausforderung. 

In meiner beruflichen Praxis konnte ich bereits umfassende Kompetenzen in der Beschaffung, Analyse und Visualisierung von Daten aufbauen. Ich verfüge über sehr gute Kenntnisse in Power BI, Python, R und SQL. In meiner aktuellen Werkstudententätigkeit war ich maßgeblich an der Konzeption und Umsetzung der internen Reporting-Landschaft beteiligt – von der Anforderungsaufnahme mit unterschiedlichen Stakeholdern über die Datenbeschaffung und Aufbereitung bis hin zur Visualisierung in Power BI. 
Darüber hinaus beschäftige ich mich intensiv mit Künstlicher Intelligenz und Machine Learning. In meiner derzeitigen Position habe ich u.a. einen unternehmensinternen Chatbot erstellt, der Fragen zu firmenbezogenen Dokumenten beantwortet. Ergänzend dazu halte ich englischsprachige Schulungen zur verantwortungsvollen Nutzung von KI-Tools im Unternehmen. 
Meine Leidenschaft für diese Themen zeigt sich auch in meiner akademischen Laufbahn: Das Modul Advanced Data Visualization habe ich mit der Note 1,0 abgeschlossen. Meine Masterarbeit schreibe ich gerade über das Thema “Determinants of Sharing Sensitive Data with AI Tools in the Workplace: A Privacy Calculus Perspective” und ich habe im Rahmen eines internationalen Projekts in Kooperation mit dem MIT ein KI-Modell entwickelt, das Emotionen bei Pferden erkennt. Weitere Informationen zu mir und meinen Fähigkeiten finden Sie auf meiner Webseite unter: jacob-facius.de

Im November 2025 werde ich mein Masterstudium abschließen und sehe nun den richtigen Zeitpunkt gekommen, um meine Fähigkeiten in einem neuen Arbeitsumfeld unter Beweis zu stellen. Mit meinen Qualifikationen, analytischen Fähigkeiten, meiner offenen Art und meiner selbstständigen Arbeitsweise bin ich überzeugt, dass ich einen wertvollen Beitrag zu Ihrem Team leisten werde. 

Gerne stehe ich Ihnen jederzeit zur Verfügung und würde mich sehr darüber freuen, Sie in einem persönlichen Gespräch kennenzulernen und mehr über die Position und Ihr Unternehmen zu erfahren.

Ich freue mich auf Ihre Rückmeldung.

Mit freundlichen Grüßen,
Jacob Facius
"""

# ==============================
# 2. Verarbeitung & Vektoren
# ==============================

# Text in Dokumentobjekte konvertieren
doc = Document(page_content=text)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents([doc])

# Embeddings laden
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# Mistral via Ollama (lokal) verwenden
llm = Ollama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ==============================
# 3. Streamlit App
# ==============================

st.set_page_config(page_title="Jacob Chatbot", page_icon="🤖")
st.title("🤖 Chatbot zu Jacob Facius")
st.write("Stelle Fragen basierend auf dem Bewerbungstext.")

query = st.text_input("Deine Frage hier eingeben:")

if query:
    with st.spinner("Antwort wird generiert..."):
        result = qa_chain.run(query)
        st.markdown(f"**Antwort:** {result}")
