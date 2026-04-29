import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
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

st.set_page_config(page_title="AI Insurance Policy Assistant", layout="wide")
st.title("📄 AI Insurance Policy Summarizer + Comparator")

# ---------------------------
# MODE SELECTOR
# ---------------------------
mode = st.radio("Select Mode", ["Single PDF", "Multiple PDFs"])

# ---------------------------
# LANGUAGE
# ---------------------------
language = st.selectbox("🌍 Language", ["English", "Hindi"])

# ---------------------------
# HINDI FONT
# ---------------------------
FONT_PATH = "NotoSansDevanagari-Regular.ttf"

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont("Devanagari", FONT_PATH))
    HINDI_FONT = "Devanagari"
else:
    HINDI_FONT = "Helvetica"

# ---------------------------
# SESSION STATE
# ---------------------------
if "policies" not in st.session_state:
    st.session_state.policies = []

if "report" not in st.session_state:
    st.session_state.report = None

if "skipped_files" not in st.session_state:
    st.session_state.skipped_files = []

# ---------------------------
# SANITIZATION
# ---------------------------
def sanitize_text(text):
    replacements = {
        r'\bfemale\b': 'individual',
        r'\bmale\b': 'individual',
        r'\bshe\b': 'the person',
        r'\bhe\b': 'the person',
        r'\bpatient\b': 'person',
        r'\btreatment\b': 'care',
        r'\bsurgery\b': 'clinical procedure',
        r'\billness\b': 'sickness',
        r'\banxiety\b': 'unease',
    }

    for p, r in replacements.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)

    return text


def sanitize_input_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,:/()-]', '', text)
    return text[:8000]


def build_safe_context(docs):
    cleaned_docs = []
    for d in docs[:3]:
        cleaned_docs.append(
            Document(
                page_content=sanitize_input_text(d.page_content),
                metadata=d.metadata
            )
        )
    return cleaned_docs


def simplify_policy_terms(text):
    replacements = {
        r'\bexclusion\b': 'not covered condition',
        r'\bpremium\b': 'payment for insurance',
        r'\bdeductible\b': 'amount you pay before insurance starts',
        r'\bclaim\b': 'request for insurance payment',
        r'\bcoverage\b': 'what is included in insurance',
    }

    for p, r in replacements.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)

    return text


# ---------------------------
# FIXED CLASSIFIER (IMPORTANT)
# ---------------------------
def is_insurance_document_llm(text, llm):
    prompt = f"""
Return ONLY YES or NO.

Is this an insurance policy document?

Text:
{text[:2500]}
"""

    result = llm.invoke(prompt).content.strip().lower()
    return result.startswith("yes")


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
# PROMPT
# ---------------------------
summary_prompt = ChatPromptTemplate.from_template("""
You are an Insurance Policy Expert.

Explain clearly in {language}.

Focus on:
- Coverage
- Exclusions
- Premium
- Claim process

Context:
{context}
""")

summary_chain = create_stuff_documents_chain(llm, summary_prompt)

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_files = st.file_uploader(
    "📤 Upload Insurance Policy PDF(s)",
    type="pdf",
    accept_multiple_files=(mode == "Multiple PDFs")
)

# ---------------------------
# SINGLE PDF
# ---------------------------
if mode == "Single PDF" and uploaded_files:

    file = uploaded_files

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        path = tmp.name

    raw_text = extract_text(path) or ""

    if raw_text.strip():

        raw_text = sanitize_text(raw_text)

        if is_insurance_document_llm(raw_text, llm):

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_text(raw_text)

            vectordb = Chroma.from_texts(chunks, embedding_model)
            retriever = vectordb.as_retriever(search_kwargs={"k": 4})

            docs = retriever.get_relevant_documents("summary")
            safe_docs = build_safe_context(docs)

            result = summary_chain.invoke({
                "context": safe_docs,
                "language": language
            })

            st.session_state.report = simplify_policy_terms(sanitize_text(str(result)))

# ---------------------------
# MULTIPLE PDFS (FIXED)
# ---------------------------
if mode == "Multiple PDFs" and uploaded_files:

    st.session_state.policies = []
    st.session_state.skipped_files = []

    for file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            path = tmp.name

        raw_text = extract_text(path) or ""

        if not raw_text.strip():
            st.session_state.skipped_files.append(f"{file.name} → Empty file")
            continue

        raw_text = sanitize_text(raw_text)

        if not is_insurance_document_llm(raw_text, llm):
            st.session_state.skipped_files.append(f"{file.name} → Not an insurance policy")
            continue

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_text(raw_text)

        vectordb = Chroma.from_texts(chunks, embedding_model)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        docs = retriever.get_relevant_documents("summary")
        safe_docs = build_safe_context(docs)

        result = summary_chain.invoke({
            "context": safe_docs,
            "language": language
        })

        summary = simplify_policy_terms(sanitize_text(str(result)))

        st.session_state.policies.append({
            "name": file.name,
            "summary": summary
        })

# ---------------------------
# DISPLAY SINGLE
# ---------------------------
if mode == "Single PDF" and st.session_state.report:
    st.subheader("📘 Policy Summary")
    st.write(st.session_state.report)

# ---------------------------
# DISPLAY MULTI
# ---------------------------
if mode == "Multiple PDFs" and st.session_state.policies:

    st.subheader("📘 Policy Summaries")

    for p in st.session_state.policies:
        st.markdown(f"### 📄 {p['name']}")
        st.write(p["summary"])

# ---------------------------
# SHOW SKIPPED FILES
# ---------------------------
if mode == "Multiple PDFs" and st.session_state.skipped_files:

    st.subheader("⚠️ Skipped Files")

    for msg in st.session_state.skipped_files:
        st.warning(msg)

# ---------------------------
# DOWNLOAD SINGLE
# ---------------------------
if mode == "Single PDF" and st.session_state.report:

    def generate_pdf(text):
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
            Paragraph("Insurance Summary", styles["Title"]),
            Spacer(1, 10)
        ]

        for line in text.split("\n"):
            if line.strip():
                content.append(Paragraph(line, style))
                content.append(Spacer(1, 5))

        doc.build(content)
        buffer.seek(0)
        return buffer

    pdf = generate_pdf(st.session_state.report)

    st.download_button(
        "📥 Download Summary",
        pdf,
        file_name="policy_summary.pdf"
    )