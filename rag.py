import os
import tempfile
from tkinter.ttk import Separator

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Page Setup
st.set_page_config(page_title="PDF RAg AGENT", page_icon="💻", layout="wide")
st.title("📃 PDF RAG AGENT")
st.caption("Upload a PDF, then ask Questions - answers come only from documents")

#Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    value=os.getenv("OPENAI_API_KEY", "")
)                   
    model_name = st.selectbox("chat Model", [ "gpt-4o-mini", "gpt-4o"])
    chunk_size = st.slider("Chunk Size", 500,2000,1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 150, step=50)
    top_k = st.slider("Retrieved chunks", 2,10,4)

    if not api_key:
        st.warning("Please enter your OpenAI API Key to continue.")
        st.stop() 

    os.environ["OPENAI_API_KEY"] = api_key
    #PDF Upload + indexing
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
def build_vectorstore(pdf_bytes:bytes, size:int, overlap:int) -> FAISS:
    """Load PDF, split into chunk, embed, and store in FAISS"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name
            try:
                 docs=PyPDFLoader(temp_file_path).load()
            finally:
                 os.unlink(temp_file_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            return FAISS.from_documents(chunks,embeddings)
    if uploaded_file is not None:
        file_sig = (
        uploaded_file.name,
        uploaded_file.size,
        chunk_size,
        chunk_overlap,
    )

    if st.session_state.get("file_sig") != file_sig:
        with st.spinner("Reading and Indexing PDF"):
            st.session_state.vectorstore = build_vectorstore(
                uploaded_file.getvalue(),
                chunk_size,
                chunk_overlap,
            )

            st.session_state.file_sig = file_sig
            st.session_state.messages = []

        st.success(f"Indexed **{uploaded_file.name}** and ask away")
# RAG chain---------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions strictly using the "
     "provided context from a PDF document.\n"
     "Rules:\n"
     "1. Answer ONLY from the context below.\n"
     "2. If the answer is not in the context, say: "
     "\"I couldn't find that in the document.\"\n"
     "3. Cite the page number(s) you used, e.g. (p. 3).\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    """Join retrieved chunks, tagging each with its page number."""
    return "\n\n".join(
        f"[Page {d.metadata.get('page', '?') + 1}]\n{d.page_content}"
        for d in docs
    )


def get_chain(vectorstore: FAISS, model: str, k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=0)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    ), retriever

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if "vectorstore" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the PDF…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        chain, retriever = get_chain(
            st.session_state.vectorstore, model_name, top_k
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = chain.invoke(question)
                st.markdown(answer)

                # Show the retrieved chunks for transparency
                with st.expander("🔍 Sources (retrieved chunks)"):
                    for doc in retriever.invoke(question):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Page {page + 1 if isinstance(page, int) else page}**")
                        st.text(doc.page_content[:500])
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload a PDF to get started.")               