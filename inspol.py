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

st.set_page_config(page_title="AI Insurance Policy Assistant", layout="wide")
st.title("📄 AI Insurance Policy Summarizer")

# ---------------------------
# LANGUAGE SELECTION
# ---------------------------
language = st.selectbox("🌍 Language", ["English", "Hindi"])

# ---------------------------
# REGISTER HINDI FONT (IMPORTANT)
# ---------------------------
FONT_PATH = "NotoSansDevanagari-Regular.ttf"

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont("Devanagari", FONT_PATH))
    HINDI_FONT = "Devanagari"
else:
    HINDI_FONT = "Helvetica"

# ---------------------------
# POLICY SIMPLIFIER
# ---------------------------
def simplify_policy_terms(text):
    replacements = {
        r'\bexclusion\b': 'not covered condition',
        r'\bpremium\b': 'payment for insurance',
        r'\bdeductible\b': 'amount you pay before insurance starts',
        r'\bclaim\b': 'request for insurance payment',
        r'\bcoverage\b': 'what is included in insurance',
        r'\bbeneficiary\b': 'person receiving benefits',
        r'\bliability\b': 'legal responsibility',
        r'\brider\b': 'extra coverage option'
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


# ---------------------------
# POLICY VALIDATION
# ---------------------------
# ---------------------------
# POLICY VALIDATION (LLM-BASED)
# ---------------------------
def is_insurance_document_llm(text, llm):
    prompt = f"""
You are a document classifier.

Return ONLY YES or NO.

Is this an insurance policy document?

Text:
{text[:3000]}
"""

    result = llm.invoke(prompt).content.lower()
    return "yes" in result


# ---------------------------
# POLICY INSIGHTS (LAB ANALYZER REPLACEMENT)
# ---------------------------
def extract_policy_clauses(text):
    clauses = {
        "Exclusions": ["not covered", "exclusion", "does not cover"],
        "Waiting Period": ["waiting period", "after 30 days", "after 90 days"],
        "Coverage": ["covered", "benefits include"],
        "Deductible": ["deductible", "out of pocket"],
        "Claim Process": ["claim", "documents", "submit"]
    }

    text = text.lower()
    return [c for c, keys in clauses.items() if any(k in text for k in keys)]


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
uploaded_file = st.file_uploader("📤 Upload Insurance Policy PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    raw_text = extract_text(pdf_path) or ""
    st.session_state.raw_text = raw_text

    if not raw_text.strip():
        st.error("No readable content found")
        st.stop()

    if not is_insurance_document_llm(raw_text, llm):
        st.error("❌ Not a valid insurance policy document")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    vectordb = Chroma.from_texts(chunks, embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    st.session_state.vectordb = vectordb

    # ---------------------------
    # PROMPT (HINDI ENABLED)
    # ---------------------------
    prompt = ChatPromptTemplate.from_template("""
You are an Insurance Policy Expert.

Explain the policy clearly in {language}.

Focus on:
- Coverage
- Exclusions
- Premium
- Claim process

Context:
{context}
""")

    chain = create_stuff_documents_chain(llm, prompt)
    docs = retriever.get_relevant_documents("summary")

    result = chain.invoke({
        "context": docs,
        "language": language
    })

    st.session_state.report = simplify_policy_terms(str(result))


# ---------------------------
# DISPLAY REPORT
# ---------------------------
if st.session_state.report:
    st.subheader("📘 Policy Summary")
    st.write(st.session_state.report)


# ---------------------------
# POLICY INSIGHTS
# ---------------------------
if st.session_state.raw_text:
    clauses = extract_policy_clauses(st.session_state.raw_text)

    if clauses:
        st.subheader("🧾 Key Policy Sections")
        for c in clauses:
            st.markdown(f"- 📌 {c}")


# ---------------------------
# CHAT
# ---------------------------
if st.session_state.vectordb:

    st.subheader("💬 Ask About Policy")

    user_input = st.text_input("Ask question")

    if st.button("Ask") and user_input:

        retriever = st.session_state.vectordb.as_retriever()
        docs = retriever.get_relevant_documents(user_input)

        qa_prompt = ChatPromptTemplate.from_template("""
You are an insurance expert.

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

        st.write("🤖", simplify_policy_terms(str(answer)))


# ---------------------------
# PDF EXPORT (WITH HINDI SUPPORT)
# ---------------------------
def generate_pdf(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    custom_style = ParagraphStyle(
        name="Custom",
        parent=styles["Normal"],
        fontName=HINDI_FONT,
        fontSize=11,
        leading=14
    )

    content = [
        Paragraph("Insurance Policy Summary", styles["Title"]),
        Spacer(1, 10)
    ]

    for line in report.split("\n"):
        if line.strip():
            content.append(Paragraph(line, custom_style))
            content.append(Spacer(1, 5))

    doc.build(content)
    buffer.seek(0)
    return buffer


if st.session_state.report:
    pdf = generate_pdf(st.session_state.report)
    st.download_button("📥 Download Summary", pdf, file_name="policy_summary.pdf")
