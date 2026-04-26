To enable LLM API access:
=========================
    - Visit link: https://genailab.tcs.in
    - Authorize your API key
      -AWS_Key:saojnsuan



Optional steps for Python Libraries:
====================================
pip install langchain langchain-openai langchain-community tiktoken httpx
pip install pypdf langchain_community langchain-openai langchain-chroma

pip install langchain-openai
pip install langchain-chroma


Solve tiktoken SSL issue steps:
================================
1. Create folder in project called "token" which you are using in the code
2. Download tiktoken manually from https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
3. Copy the downloaded tiktoken in the folder "token" created above.
4. Rename the tiktoken file to its SHA1 hash value: 9b5ad71b2ce5302211f9c61530b329a4922fc6a4
  and remove the .token extension.
5. Add following code to your program:
tiktoken_cache_dir = "token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir


Others:
=======

streamlit UI Application
import streamlit as st
# The above imports Streamlit (web UI framework) and aliases it st. You use this to build the simple web app UI.

streamlit run .\your_program.py --debug


pdfminer.high_level  reads data from PDF 
extract_text from pdfminer.high_level — function that extracts plain text from a PDF file path.

RecursiveCharacterTextSplitter — a LangChain text-splitting utility that splits long text into chunks while trying to keep boundaries sensible (characters-based splitter with overlap).

ChatOpenAI, OpenAIEmbeddings — wrappers that create an LLM client and an embeddings client (your code is using a custom provider endpoint).

Chroma — the vectorstore implementation (stores embeddings and allows similarity search). It is DB 

tempfile — standard library module for creating temporary files (used to save uploaded PDFs).

os — standard library OS utilities (used for env vars).

httpx — an HTTP client used to supply a custom HTTP client to the LLM/embeddings clients and to set verify=False