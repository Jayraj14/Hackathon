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
# INSURANCE TEXT SIMPLIFIER
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
# POLICY CLAUSE EXTRACTOR (REPLACES LAB ANALYZER)
# ---------------------------
def extract_policy_clauses(text):
    clauses = {
        "Exclusions": ["not covered", "exclusion", "does not cover", "excluded"],
        "Waiting Period": ["waiting period", "after 30 days", "after 90 days"],
        "Coverage": ["covered", "coverage includes", "benefits include"],
        "Deductible": ["deductible", "out of pocket"],
        "Claim Rules": ["claim", "claim process", "submit documents"]
    }

    results = []

    text_lower = text.lower()

    for clause, keywords in clauses.items():
        if any(k in text_lower for k in keywords):
            results.append(clause)

    return results


# ---------------------------
# SIMPLE POLICY CHECK (instead of medical check)
# ---------------------------
def is_insurance_document(text):
    keywords = [
        "policy", "insurance", "premium", "coverage",
        "claim", "benefit", "exclusion", "deductible",
        "insured", "sum assured"
    ]
    text = text.lower()
    score = sum(word in text for word in keywords)
    return score >= 3


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
# UPLOAD PDF
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

    # ---------------------------
    # POLICY VALIDATION
    # ---------------------------
    if not is_insurance_document(raw_text):
        st.error("❌ This does not look like an insurance policy document")
        st.stop()

    # ---------------------------
    # VECTOR DB
    # ---------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    vectordb = Chroma.from_texts(chunks, embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    st.session_state.vectordb = vectordb

    # ---------------------------
    # POLICY SUMMARY
    # ---------------------------
    prompt = ChatPromptTemplate.from_template("""
You are an Insurance Policy Expert.

Explain this insurance policy in simple terms.

Focus on:
- Coverage
- Exclusions
- Premium
- Claim process
- Hidden conditions

Context:
{context}
""")

    chain = create_stuff_documents_chain(llm, prompt)
    docs = retriever.get_relevant_documents("summary")

    result = chain.invoke({"context": docs})

    st.session_state.report = simplify_policy_terms(str(result))

# ---------------------------
# DISPLAY REPORT
# ---------------------------
if st.session_state.report:
    st.subheader("📘 Policy Summary")
    st.write(st.session_state.report)

# ---------------------------
# 🧾 POLICY INSIGHTS (REPLACES LAB ANALYZER)
# ---------------------------
if st.session_state.raw_text:
    clauses = extract_policy_clauses(st.session_state.raw_text)

    if clauses:
        st.subheader("🧾 Key Policy Sections Found")

        for c in clauses:
            st.markdown(f"- 📌 {c}")

# ---------------------------
# CHAT
# ---------------------------
if st.session_state.vectordb:

    st.subheader("💬 Ask About Policy")

    user_input = st.text_input("Ask your question")

    if st.button("Ask") and user_input:

        retriever = st.session_state.vectordb.as_retriever()
        docs = retriever.get_relevant_documents(user_input)

        qa_prompt = ChatPromptTemplate.from_template("""
You are an insurance expert.

Answer ONLY from the policy context.

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

        answer = simplify_policy_terms(str(answer))

        st.write("🤖", answer)

# ---------------------------
# PDF EXPORT
# ---------------------------
def generate_pdf(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Insurance Policy Summary", styles["Title"]),
        Spacer(1, 10)
    ]

    for line in report.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 5))

    doc.build(content)
    buffer.seek(0)
    return buffer


if st.session_state.report:
    pdf = generate_pdf(st.session_state.report)
    st.download_button("📥 Download Summary", pdf, file_name="policy_summary.pdf")