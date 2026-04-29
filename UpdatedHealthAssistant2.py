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

# ---------------------------
# INIT
# ---------------------------
load_dotenv()
client = httpx.Client(verify=False)
os.environ["TIKTOKEN_CACHE_DIR"] = "token"

st.set_page_config(page_title="AI Patient Assistant", layout="wide")
st.title("🧠 AI Patient Education Assistant")

# ---------------------------
# HELPERS (SAFE FOR UI ONLY)
# ---------------------------
def sanitize_text(text):
    replacements = {
        r'\bfemale\b': 'individual',
        r'\bmale\b': 'individual',
        r'\bshe\b': 'the person',
        r'\bhe\b': 'the person',
        r'\bpatient\b': 'person'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

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

def is_medical_document(text):
    keywords = [
        "diagnosis","treatment","medication","patient",
        "ICD","blood","pressure","cholesterol",
        "hypertension","glucose","mg","doctor",
        "ECG","EKG","clinical"
    ]
    text = text.lower()
    score = sum(word in text for word in keywords)
    return score >= 3

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
                status = "Low"
                color = "red"
            elif value > high:
                status = "High"
                color = "orange"
            else:
                status = "Normal"
                color = "green"

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
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
    st.session_state.report = None
    st.session_state.raw_text = None
    st.session_state.chat_history = []

# ---------------------------
# UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload Medical PDF", type="pdf")

if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name

    # ✅ RAW TEXT (DO NOT MODIFY)
    raw_text = extract_text(pdf_path) or ""
    st.session_state.raw_text = raw_text

    if not raw_text.strip():
        st.error("No readable content in PDF")
        st.stop()

    # ✅ MEDICAL CHECK
    if not is_medical_document(raw_text):
        st.error("❌ This does not appear to be a medical document")
        st.stop()

    # ---------------------------
    # VECTOR DB (RAW TEXT ONLY)
    # ---------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    vectordb = Chroma.from_texts(chunks, embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    st.session_state.vectordb = vectordb

    # ---------------------------
    # REPORT GENERATION
    # ---------------------------
    prompt = ChatPromptTemplate.from_template("""
You are an empathetic healthcare assistant.

Explain this medical report in simple terms.

Answer clearly.

Context:
{context}
""")

    chain = create_stuff_documents_chain(llm, prompt)
    docs = retriever.get_relevant_documents("summary")

    result = chain.invoke({"context": docs})

    # ✅ SANITIZE ONLY FOR DISPLAY
    final_report = simplify_medical_terms(sanitize_text(str(result)))
    st.session_state.report = final_report

# ---------------------------
# DISPLAY REPORT
# ---------------------------
if st.session_state.report:
    st.subheader("📘 Patient-Friendly Report")
    st.write(st.session_state.report)

# ---------------------------
# 🧪 LAB ANALYSIS (BEST VERSION)
# ---------------------------
text_for_analysis = ""

if st.session_state.vectordb:
    try:
        docs = st.session_state.vectordb._collection.get()['documents']
        text_for_analysis = " ".join(docs)
    except:
        text_for_analysis = st.session_state.raw_text
else:
    text_for_analysis = st.session_state.raw_text

if text_for_analysis:
    lab_results = extract_lab_values(text_for_analysis)

    if lab_results:
        st.subheader("🧪 Lab Value Analysis")

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
    user_input = st.text_input("Type your question")

    if st.button("Ask") and user_input:

        retriever = st.session_state.vectordb.as_retriever()

        docs = retriever.get_relevant_documents(user_input)

        qa_prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the context.

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

    content = [Paragraph("Patient Report", styles["Title"]), Spacer(1, 10)]

    for line in report.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 5))

    doc.build(content)
    buffer.seek(0)
    return buffer

if st.session_state.report:
    pdf = generate_pdf(st.session_state.report)
    st.download_button("📥 Download PDF", pdf, file_name="report.pdf")
