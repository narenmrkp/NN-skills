    # RAG Chatbot (from PDFs) with Groq + Streamlit (Modular way)   (Ref Link: https://github.com/snsupratim/RagBot/tree/main)
# Features
Upload and process multiple PDF files
Store document embeddings in Chroma (in-memory)
Query using GROQ LLM with Retrieval-Augmented Generation (RAG)
Inspect vector store chunks from the sidebar
Modular and clean code structure

create Env:
python -m venv myenv --> myenv\Scripts\activate --> source myenv/bin/activate (Mac/Linux) --> pip install -r requirements.txt
# Project-Structure:
groq_rag_chatbot/
|
├── index.py              # Main Streamlit app
├── requirements.txt      # Python dependencies
|
├── moduless/             # All modular logic
|   ├── pdf_handler.py        # PDF upload and loading logic
|   ├── vectorstore.py        # Chroma in-memory vector store setup
|   ├── llm.py                # GROQ LLM and RetrievalQA chain
|   ├── chat.py               # Chat interaction logic (input/output)
|   ├── chroma_inspector.py   # View vector store chunks from sidebar

1. requirements.txt file:
streamlit
langchain_groq 
langchain_community 
pypdf 
sentence_transformers
python-dotenv
chromadb
transformers
-----------------------------------------------------
2. index.py file:
import warnings
import logging
import streamlit as st
# Local modules
from modules.chat import display_chat_history, handle_user_input, download_chat_history
from modules.pdf_handler import upload_pdfs
from modules.vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.chroma_inspector import inspect_chroma

# Silence noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
st.set_page_config(
    page_title="RagBot!",     
)

# App title
st.title("Ask Ragbot! ")
# Step 1: Upload PDFs + wait for submit
uploaded_files, submitted = upload_pdfs()
# Step 2: If user clicks submit, update vectorstore
if submitted and uploaded_files:
    with st.spinner(" Updating vector database..."):
        vectorstore = load_vectorstore(uploaded_files)
        st.session_state.vectorstore = vectorstore
# Step 3: Display vectorstore inspector (Sidebar)
if "vectorstore" in st.session_state:
    inspect_chroma(st.session_state.vectorstore)
# Step 4: Display old chat messages
display_chat_history()
# Step 5: Handle new prompt input
if "vectorstore" in st.session_state:
    handle_user_input(get_llm_chain(st.session_state.vectorstore))
# Step 6: Chat history export
download_chat_history()
--------------------------------------------------
3. modules/chat.py file:
import streamlit as st

def display_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

def handle_user_input(chain):
    user_input = st.chat_input("Pass your prompt here")
    if not user_input:
        return
    
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append( {"role": "user", "content": user_input} )

    try:
        result = chain({"query": user_input})
        response = result["result"]
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append( {"role": "assistant", "content": response} )
    except Exception as e:
        st.error(f"Error: {str(e)}")

def download_chat_history():
    if st.session_state.get("messages"):
        content = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("💾 Download Chat History", content, file_name="chat_history.txt", mime="text/plain")
-----------------------------------------------------------------
4. modules/chroma_inspector.py file:
import streamlit as st
from langchain.vectorstores import Chroma

def inspect_chroma(vectorstore):
    st.sidebar.markdown("🧪 **ChromaDB Inspector**")
    
    # Show basic info
    try:
        doc_count = vectorstore._collection.count()
        st.sidebar.success(f"🔎 {doc_count} documents stored in ChromaDB.")
    except Exception as e:
        st.sidebar.error("Could not fetch document count.")
        st.sidebar.code(str(e))

    # Search inside the vectorstore
    query = st.sidebar.text_input("🔍 Test a query against ChromaDB")

    if query:
        try:
            results = vectorstore.similarity_search(query, k=3)
            st.sidebar.markdown("### Top Matching Chunks:")
            for i, doc in enumerate(results):
                st.sidebar.markdown(f"**Result {i+1}:**")
                st.sidebar.markdown(doc.page_content[:300] + "...")
                st.sidebar.markdown("---")
        except Exception as e:
            st.sidebar.error("Error querying ChromaDB")
            st.sidebar.code(str(e))
------------------------------------------------------------------------
5. modules/llm.py file:
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def get_llm_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
--------------------------------------------------------------------------
6. modules/pdf_handler.py file:
import streamlit as st
import tempfile

def upload_pdfs():
    with st.sidebar:
        st.header("📁 Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        submit = st.button(" Submit to DB")
    return uploaded_files, submit

def save_uploaded_files(uploaded_files):
    file_paths = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            file_paths.append(tmp.name)
    return file_paths
---------------------------------------------------------------------------
7. modules/vectorstore.py file:
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.pdf_handler import save_uploaded_files
import os

PERSIST_DIR = "./chroma_store"

def load_vectorstore(uploaded_files):
    paths = save_uploaded_files(uploaded_files)

    docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # Append to existing
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        vectorstore.add_documents(texts)
        vectorstore.persist()
    else:
        # Create new
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()

    return vectorstore
---------------------------------------------------------------------------------------------------------------
