import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import traceback

# =============================
# STREAMLIT CONFIG (MUST BE FIRST)
# =============================
st.set_page_config(page_title="RAG App", layout="wide")

# =============================
# LOAD ENV
# =============================
load_dotenv(override=True)

# =============================
# LangChain imports
# =============================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# =============================
# CONFIG
# =============================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVAL = 4

# =============================
# UI HEADER
# =============================
st.title("🤖 RAG Document Q&A")

# =============================
# API KEY CHECK
# =============================
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ API Key not found. Please add it in .env file")
    st.stop()

# =============================
# FUNCTIONS
# =============================

def load_pdf(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(docs)
        return chunks

    except Exception as e:
        st.error(f"❌ Error loading PDF: {str(e)}")
        traceback.print_exc()
        return None


def create_vector_store(chunks):
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key
        )
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None


def query_rag(vector_store, query):
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=api_key
        )

        prompt = PromptTemplate(
            template="""
Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:
""",
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": K_RETRIEVAL}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        result = qa_chain({"query": query})
        return result["result"], result["source_documents"]

    except Exception as e:
        st.error(f"❌ Error generating answer: {str(e)}")
        return None, None


# =============================
# SESSION STATE
# =============================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =============================
# FILE UPLOAD
# =============================
st.header("📂 Upload PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("📄 Processing PDF..."):
        chunks = load_pdf(uploaded_file)

        if chunks:
            st.session_state.vector_store = create_vector_store(chunks)
            st.success("✅ Document processed successfully!")

# =============================
# QUERY SECTION
# =============================
if st.session_state.vector_store:
    st.header("❓ Ask a Question")

    query = st.text_input("Enter your question:")

    if st.button("🔍 Get Answer"):
        if query.strip():
            with st.spinner("🤖 Generating answer..."):
                answer, docs = query_rag(
                    st.session_state.vector_store, query
                )

                if answer:
                    st.subheader("📝 Answer")
                    st.write(answer)

                    st.subheader("📚 Source Chunks")
                    for i, d in enumerate(docs):
                        with st.expander(f"Chunk {i+1}"):
                            st.write(d.page_content)
        else:
            st.warning("⚠️ Please enter a question.")

else:
    st.info("👆 Upload a PDF to start asking questions.")

# =============================
# CLEAR CACHE
# =============================
if st.button("🗑️ Clear Cache"):
    st.session_state.vector_store = None
    st.success("✅ Cache cleared! Upload a new file.")