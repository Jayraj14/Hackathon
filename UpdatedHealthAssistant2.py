import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from io import BytesIO
import tempfile, os, re
from dotenv import load_dotenv
import httpx

# ---------------------------
# INIT
# ---------------------------
load_dotenv()
client = httpx.Client(verify=False)
os.environ["TIKTOKEN_CACHE_DIR"] = "token"

st.set_page_config(page_title="AI Patient Assistant", layout="wide")
st.title("🧠 AI Patient Education Assistant")

# ---------------------------
# LANGUAGE SELECTION
# ---------------------------
language = st.selectbox("🌍 Language", ["English", "Hindi"])

# ---------------------------
# HINDI FONT SUPPORT
# ---------------------------
FONT_PATH = "NotoSansDevanagari-Regular.ttf"

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont("Devanagari", FONT_PATH))
    HINDI_FONT = "Devanagari"
else:
    HINDI_FONT = "Helvetica"

# ---------------------------
# HELPERS
# ---------------------------
def sanitize_text(text):
    replacements = {
        r'\bfemale\b': 'individual',
        r'\bmale\b': 'individual',
        r'\bshe\b': 'the person',
        r'\bhe\b': 'the person',
        r'\bpatient\b': 'person'
    }
    for p, r in replacements.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text


def simplify_medical_terms(text):
    replacements = {
        r'\bhypertension\b': 'high blood pressure',
        r'\bglucose\b': 'blood sugar',
        r'\bcholesterol\b': 'fat in blood',
        r'\brenal\b': 'kidney related',
        r'\bhepatic\b': 'liver related'
    }
    for p, r in replacements.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text


def is_medical_document(text):
    keywords = [
        # ---------------- English medical terms ----------------
        "diagnosis", "treatment", "medication", "patient",
        "blood", "pressure", "cholesterol",
        "hypertension", "glucose", "doctor",
        "clinical", "report", "test",

        # ---------------- Hindi medical terms ----------------
        "निदान",        # diagnosis
        "उपचार",        # treatment
        "दवा",          # medicine
        "मरीज",         # patient
        "रोगी",         # patient (formal)
        "रक्त",         # blood
        "रक्तचाप",      # blood pressure
        "कोलेस्ट्रॉल",   # cholesterol
        "मधुमेह",       # diabetes
        "शुगर",         # sugar/glucose
        "डॉक्टर",       # doctor
        "चिकित्सा",     # medical
        "रिपोर्ट",       # report
        "जांच",         # test
        "लक्षण",        # symptoms
        "नैदानिक"       # clinical
    ]

    text = text.lower()

    # Count keyword matches
    score = sum(k in text for k in keywords)

    return score >= 3

# ---------------------------
# SAFE SUMMARIZATION (FIX FOR 403 + CONTROLLED CONTEXT)
# ---------------------------
def safe_summarize_docs(docs, llm, language):
    limited_docs = docs[:3]  # 🔥 prevents token overflow (IMPORTANT)

    prompt = ChatPromptTemplate.from_template("""
You are a medical assistant.

Explain the medical report in {language} in simple terms.

Rules:
- Use simple language
- Be safe and non-alarming
- Focus on:
  * Diagnosis
  * Symptoms
  * Treatment
  * Lab results
- If information is missing, say "Not available in report"

Context:
{context}
""")

    chain = create_stuff_documents_chain(llm, prompt)

    return chain.invoke({
        "context": limited_docs,
        "language": language
    })


# ---------------------------
# LAB ANALYZER
# ---------------------------
def extract_lab_values(text):
    patterns = {
        "Hemoglobin": (r"hemoglobin[:\s]*([\d.]+)", 13, 17),
        "WBC": (r"wbc[:\s]*([\d.]+)", 4000, 11000),
        "RBC": (r"rbc[:\s]*([\d.]+)", 4.5, 5.9),
        "Platelets": (r"platelet[s]*[:\s]*([\d.]+)", 150000, 450000),
        "Glucose": (r"glucose[:\s]*([\d.]+)", 70, 140),
        "Cholesterol": (r"cholesterol[:\s]*([\d.]+)", 125, 200),
    }

    results = []

    for name, (pattern, low, high) in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))

            if value < low:
                status, color = "Low", "red"
            elif value > high:
                status, color = "High", "orange"
            else:
                status, color = "Normal", "green"

            results.append({
                "name": name,
                "value": value,
                "status": status,
                "color": color,
                "range": f"{low}-{high}"
            })

    return results


# ---------------------------
# MODEL
# ---------------------------
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

# ---------------------------
# SESSION STATE
# ---------------------------
for k in ["vectordb", "report", "raw_text"]:
    if k not in st.session_state:
        st.session_state[k] = None


# ---------------------------
# UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload Medical PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    raw_text = extract_text(pdf_path) or ""
    st.session_state.raw_text = raw_text

    if not raw_text.strip():
        st.error("Empty PDF")
        st.stop()

    if not is_medical_document(raw_text):
        st.error("❌ Not a medical document")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    vectordb = Chroma.from_texts(chunks, embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    st.session_state.vectordb = vectordb

    # ---------------------------
    # PROMPT + SAFE SUMMARIZATION
    # ---------------------------
    docs = retriever.get_relevant_documents("summary")

    result = safe_summarize_docs(docs, llm, language)

    output_text = simplify_medical_terms(sanitize_text(str(result)))
    st.session_state.report = output_text


# ---------------------------
# DISPLAY REPORT
# ---------------------------
if st.session_state.report:
    st.subheader("📘 Patient Report")
    st.write(st.session_state.report)


# ---------------------------
# LAB ANALYSIS
# ---------------------------
text_for_analysis = ""

if st.session_state.vectordb:
    try:
        docs = st.session_state.vectordb._collection.get()["documents"]
        text_for_analysis = " ".join(docs)
    except:
        text_for_analysis = st.session_state.raw_text
else:
    text_for_analysis = st.session_state.raw_text

if text_for_analysis:
    lab_results = extract_lab_values(text_for_analysis)

    if lab_results:
        st.subheader("🧪 Lab Analysis")

        for lab in lab_results:
            st.markdown(
                f"**{lab['name']}**: {lab['value']} "
                f"(<span style='color:{lab['color']}'>{lab['status']}</span>) "
                f"(Normal: {lab['range']})",
                unsafe_allow_html=True
            )


# ---------------------------
# CHAT
# ---------------------------
if st.session_state.vectordb:

    st.subheader("💬 Ask Questions")

    user_input = st.text_input("Ask something")

    if st.button("Ask") and user_input:

        retriever = st.session_state.vectordb.as_retriever()
        docs = retriever.get_relevant_documents(user_input)

        qa_prompt = ChatPromptTemplate.from_template("""
Answer ONLY from context.

Context:
{context}

Question:
{input}
""")

        chain = create_stuff_documents_chain(llm, qa_prompt)

        answer = chain.invoke({
            "context": docs,
            "input": user_input
        })

        answer = simplify_medical_terms(sanitize_text(str(answer)))

        st.write("🤖", answer)


# ---------------------------
# PDF EXPORT
# ---------------------------
def generate_pdf(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    style = ParagraphStyle(
        name="Custom",
        parent=styles["Normal"],
        fontName=HINDI_FONT,
        fontSize=11,
        leading=14
    )

    content = [
        Paragraph("Patient Report", styles["Title"]),
        Spacer(1, 10)
    ]

    for line in report.split("\n"):
        if line.strip():
            content.append(Paragraph(line, style))
            content.append(Spacer(1, 5))

    doc.build(content)
    buffer.seek(0)
    return buffer


if st.session_state.report:
    pdf = generate_pdf(st.session_state.report)
    st.download_button("📥 Download PDF", pdf, file_name="patient_report.pdf")
