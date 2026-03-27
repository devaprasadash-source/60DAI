import streamlit as st
import os
import tempfile
import traceback

# =============================
# STREAMLIT CONFIG (FIRST LINE)
# =============================
st.set_page_config(page_title="RAG App", layout="wide")

# =============================
# GET API KEY (STREAMLIT CLOUD SAFE)
# =============================
api_key = st.secrets.get("OPENAI_API_KEY", None)

if not api_key:
    st.error("❌ API Key not found. Add it in Streamlit Secrets.")
    st.stop()

# =============================
# LANGCHAIN IMPORTS (CLEAN)
# =============================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# =============================
# CONFIG
# =============================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVAL = 4

# =============================
# UI HEADER
# =============================
st.title("📄 AI Document Assistant")
st.caption("Upload a PDF and ask questions using AI")

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

        retriever = vector_store.as_retriever(
            search_kwargs={"k": K_RETRIEVAL}
        )

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

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

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(query)
        docs = retriever.invoke(query)

        return answer, docs

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