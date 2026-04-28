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

# ---------------------------
# SANITIZER
# ---------------------------
def sanitize_text(text):
    replacements = {
        r'\bfemale\b': 'individual',
        r'\bmale\b': 'individual',
        r'\bwomen\b': 'individuals',
        r'\bman\b': 'individuals',
        r'\bwoman\b': 'individuals',
        r'\bmen\b': 'individuals',
        r'\bshe\b': 'the person',
        r'\bhe\b': 'the person',
        r'\bdoctor\b': 'person treating',
        r'\banxiety\b': 'worried feelings',
        r'\bdepression\b': 'mood health issues',
        r'\btreatment\b': 'care plan',
        r'\bdiagnosis\b': 'health summary',
        r'\bdisease\b': 'condition',
        r'\bcancer\b': 'serious cellular health issue',
        r'\bmedical\b': 'health',
        r'\bmedication\b': 'health',
        r'\bmedicine\b': 'healthdrug',
        r'\bsymptoms\b': 'clue',
        r'\bpatient\b': 'person'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ---------------------------
# MEDICAL SIMPLIFIER
# ---------------------------
def simplify_medical_terms(text):
    replacements = {
        r'\bsodium\b': 'sodium (salt)',
        r'\bhypertension\b': 'hypertension (high blood pressure)',
        r'\bglucose\b': 'glucose (blood sugar)',
        r'\bcholesterol\b': 'cholesterol (fat in blood)',
        r'\brenal\b': 'renal (kidney related)',
        r'\bhepatic\b': 'hepatic (liver related)'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="AI Patient Assistant", layout="wide")
st.title("🧠 AI Patient Education Assistant")


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
# SESSION STATE INIT
# ---------------------------
defaults = {
    "vectordb": None,
    "report": None,
    "chat_history": [],
    "current_query": None,
    "input_counter": 0,
    "last_q": None,
    "last_a": None,
    "retriever": None,
    "file_hash": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------
# LANGUAGE
# ---------------------------
language = st.selectbox("🌍 Language", ["English", "Hindi", "Marathi"])


# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload PDF", type="pdf")

if uploaded_file:

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # RESET ALWAYS ON NEW FILE
    if file_hash != st.session_state.file_hash:

        st.session_state.file_hash = file_hash
        st.session_state.vectordb = None
        st.session_state.report = None
        st.session_state.chat_history = []
        st.session_state.current_query = None
        st.session_state.retriever = None
        st.session_state.last_q = None
        st.session_state.last_a = None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            pdf_path = tmp.name

        raw_text = extract_text(pdf_path)
        raw_text = sanitize_text(raw_text or "")

        if not raw_text.strip():
            st.warning("⚠️ Uploaded PDF has no readable content.")
            st.session_state.report = "No readable content found in this PDF."
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(raw_text)

            vectordb = Chroma.from_texts(chunks, embedding_model)
            retriever = vectordb.as_retriever(search_kwargs={"k": 4})

            st.session_state.vectordb = vectordb
            st.session_state.retriever = retriever


# ---------------------------
# REPORT GENERATION
# ---------------------------
if st.session_state.vectordb and st.session_state.report is None:

    retriever = st.session_state.retriever

    prompt = ChatPromptTemplate.from_template("""
You are an Empathetic Patient Education Specialist.

Explain clearly and simply.

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


# ---------------------------
# DISPLAY REPORT
# ---------------------------
if st.session_state.report is not None:
    st.subheader("📘 Patient Report")
    st.write(st.session_state.report)


# ---------------------------
# CHAT
# ---------------------------
if st.session_state.vectordb:

    st.divider()
    st.subheader("💬 Chat")

    for msg in st.session_state.chat_history:
        st.markdown(f"🧑 {msg['question']}")
        st.markdown(f"🤖 {msg['answer']}")
        st.divider()

    user_input = st.text_input("Ask question:", key=f"input_{st.session_state.input_counter}")

    if st.button("Ask") and user_input.strip():
        st.session_state.current_query = user_input

    if st.session_state.current_query:

        query = sanitize_text(st.session_state.current_query)
        retriever = st.session_state.retriever

        docs = retriever.get_relevant_documents(query) if retriever else []

        if not docs:
            answer = "I don't know based on the document."
        else:
            qa_prompt = ChatPromptTemplate.from_template("""
You are a healthcare assistant.

Answer ONLY from context.

Answer in {language}

Context:
{context}

Question:
{input}
""")

            chain = create_stuff_documents_chain(llm, qa_prompt)

            answer = chain.invoke({
                "context": docs,
                "input": query,
                "language": language
            })

            answer = simplify_medical_terms(str(answer))

        st.session_state.chat_history.append({
            "question": query,
            "answer": answer
        })

        st.session_state.last_q = query
        st.session_state.last_a = answer
        st.session_state.current_query = None
        st.session_state.input_counter += 1

        st.rerun()


# ---------------------------
# PDF EXPORT
# ---------------------------
def generate_pdf(report, q=None, a=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [Paragraph("Patient Report", styles["Title"]), Spacer(1, 10)]

    for line in (report or "").split("\n"):
        if line.strip():
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


if st.session_state.report is not None:
    pdf = generate_pdf(
        st.session_state.report,
        st.session_state.last_q,
        st.session_state.last_a
    )

    st.download_button("📥 Download PDF", pdf, file_name="report.pdf")