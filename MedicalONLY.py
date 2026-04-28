import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from io import BytesIO
import tempfile, os, re, hashlib
from dotenv import load_dotenv
import httpx

load_dotenv()
client = httpx.Client(verify=False)

os.environ["TIKTOKEN_CACHE_DIR"] = "token"


# =========================
# 🔥 STRONG DOCUMENT FILTER
# =========================
def is_medical_document(text):
    text = text.lower()

    medical_signals = [
        "hemoglobin", "platelet", "wbc", "rbc",
        "blood pressure", "glucose", "cholesterol",
        "diagnosis", "symptom", "prescription",
        "lab report", "pathology", "radiology",
        "patient", "clinical", "treatment",
        "hospital", "doctor", "medical", "disease",
        "report", "test", "scan"
    ]

    non_medical_signals = [
        "farming", "agriculture", "crop", "soil",
        "fertilizer", "irrigation", "harvest",
        "tractor", "livestock", "cultivation",
        "commodity", "market price", "weather report"
    ]

    pos = sum(1 for w in medical_signals if w in text)
    neg = sum(1 for w in non_medical_signals if w in text)

    # 🚫 instantly block farming/agri documents
    if neg >= 2:
        return False

    # ✅ strict medical threshold
    return pos >= 5


# =========================
# SANITIZER
# =========================
def sanitize_text(text):
    replacements = {
        r'\bfemale\b': 'individual',
        r'\bmale\b': 'individual',
        r'\bpatient\b': 'person',
        r'\bdoctor\b': 'health professional',
        r'\bprescription\b': 'recommendation',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# =========================
# MEDICAL SIMPLIFIER
# =========================
def simplify_medical_terms(text):
    replacements = {
        r'\bhypertension\b': 'high blood pressure',
        r'\bglucose\b': 'blood sugar',
        r'\bcholesterol\b': 'fat in blood',
        r'\brenal\b': 'kidney related',
        r'\bhepatic\b': 'liver related'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="AI Patient Assistant", layout="wide")
st.title("🧠 AI Patient Education Assistant")


# =========================
# MODELS
# =========================
@st.cache_resource
def load_models():
    llm = ChatOpenAI(
        openai_api_base="https://genailab.tcs.in",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key=os.getenv("API_KEY"),
        http_client=client
    )

    embeddings = OpenAIEmbeddings(
        openai_api_base="https://genailab.tcs.in",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key=os.getenv("API_KEY"),
        http_client=client
    )
    return llm, embeddings


llm, embedding_model = load_models()


# =========================
# SESSION STATE
# =========================
for key in ["vectordb", "report", "chat_history", "retriever", "file_hash"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_history" else []


# =========================
# LANGUAGE
# =========================
language = st.selectbox("🌍 Language", ["English", "Hindi", "Marathi"])


# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload PDF", type="pdf")

if uploaded_file:

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if file_hash != st.session_state.file_hash:

        st.session_state.file_hash = file_hash
        st.session_state.vectordb = None
        st.session_state.report = None
        st.session_state.chat_history = []

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            pdf_path = tmp.name

        raw_text = extract_text(pdf_path)
        raw_text = sanitize_text(raw_text or "")

        # =========================
        # 🚫 BLOCK NON-MEDICAL DOCS
        # =========================
        if not raw_text.strip():
            st.error("Empty PDF uploaded")
            st.stop()

        if not is_medical_document(raw_text):
            st.error("❌ Only MEDICAL patient reports are allowed. Non-medical document detected.")
            st.stop()

        # =========================
        # VECTOR STORE
        # =========================
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(raw_text)

        vectordb = Chroma.from_texts(chunks, embedding_model)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        st.session_state.vectordb = vectordb
        st.session_state.retriever = retriever


# =========================
# REPORT GENERATION
# =========================
if st.session_state.vectordb and st.session_state.report is None:

    retriever = st.session_state.retriever

    prompt = ChatPromptTemplate.from_template("""
You are a medical education assistant.

Explain the patient report in simple terms.

Answer in {language}

Context:
{context}
""")

    chain = create_stuff_documents_chain(llm, prompt)
    docs = retriever.get_relevant_documents("summary")

    result = chain.invoke({
        "context": docs,
        "language": language
    })

    st.session_state.report = simplify_medical_terms(str(result))


# =========================
# DISPLAY REPORT
# =========================
if st.session_state.report:
    st.subheader("📘 Patient Report")
    st.write(st.session_state.report)


# =========================
# CHAT
# =========================
if st.session_state.vectordb:

    st.divider()
    st.subheader("💬 Chat")

    for msg in st.session_state.chat_history:
        st.markdown(f"🧑 {msg['question']}")
        st.markdown(f"🤖 {msg['answer']}")
        st.divider()

    user_input = st.text_input("Ask question:")

    if st.button("Ask") and user_input.strip():

        retriever = st.session_state.retriever
        docs = retriever.get_relevant_documents(user_input)

        if not docs:
            answer = "No relevant medical information found."
        else:
            qa_prompt = ChatPromptTemplate.from_template("""
Answer ONLY from medical context.

Answer in {language}

Context:
{context}

Question:
{input}
""")

            chain = create_stuff_documents_chain(llm, qa_prompt)

            answer = chain.invoke({
                "context": docs,
                "input": user_input,
                "language": language
            })

        answer = simplify_medical_terms(str(answer))

        st.session_state.chat_history.append({
            "question": user_input,
            "answer": answer
        })

        st.rerun()


# =========================
# PDF EXPORT
# =========================
def generate_pdf(report, q=None, a=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [Paragraph("Patient Report", styles["Title"]), Spacer(1, 10)]

    for line in (report or "").split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 5))

    if q and a:
        content.append(Spacer(1, 10))
        content.append(Paragraph("Q&A", styles["Title"]))
        content.append(Paragraph(f"Q: {q}", styles["Normal"]))
        content.append(Paragraph(f"A: {a}", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer


if st.session_state.report:
    pdf = generate_pdf(
        st.session_state.report,
        None,
        None
    )

    st.download_button("📥 Download PDF", pdf, file_name="patient_report.pdf")