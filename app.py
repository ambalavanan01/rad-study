import streamlit as st
import os
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------------
# Modern Streamlit UI Setup
# --------------------------
st.set_page_config(page_title="📘 Smart Document Search", layout="centered")
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📘 Study Material Search Assistant")
st.caption("Upload a PDF or PPTX file and ask questions. The app retrieves relevant parts of your document without using any AI model.")

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("📤 Upload your study material (PDF or PPTX)", type=["pdf", "pptx"])
retriever = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load document
    with st.spinner("🔍 Reading your file..."):
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredPowerPointLoader(file_path)
        documents = loader.load()

    # Split and embed
    with st.spinner("📚 Processing and indexing document..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("✅ Document indexed. Ask your question below:")

# --------------------------
# Search and Display Result
# --------------------------
if retriever:
    query = st.text_input("💬 Ask a question related to the document:")
    if query:
        with st.spinner("🔎 Searching..."):
            docs = retriever.get_relevant_documents(query)

        st.markdown("### 🔎 Top Matching Snippets")
        for i, doc in enumerate(docs):
            st.markdown(f"**Result {i+1}:**")
            st.info(doc.page_content.strip()[:500] + "...")
else:
    st.info("Upload a document to begin.")
