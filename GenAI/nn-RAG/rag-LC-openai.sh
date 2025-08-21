    # RAG (pdf/doc) with Langchain + OpenAI (siddhardhan, in colab)
!pip install transformers sentence-transformers langchain langchain-community langchain-openai faiss-cpu unstructured unstructured[pdf]

import os
from langchain_openai import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxx"
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# load the document
loader = UnstructuredFileLoader("/content/attention_is_all_you_need.pdf")
documents = loader.load()
type(documents[0])

# create text chunks
text_splitter = CharacterTextSplitter(separator='/n', chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(documents)

# loading the vector embedding model
embeddings = HuggingFaceEmbeddings()
# vector embedding for text chunks
knowledge_base = FAISS.from_documents(text_chunks, embeddings)

# chain for QA retrieval
qa_chain = RetrievalQA.from_chain_type(llm, retriever=knowledge_base.as_retriever())

question = "What is this document about?"
response = qa_chain.invoke({"query": question})
print(response["result"])

question = "What are the applications of attention in our Model?"
response = qa_chain.invoke({"query": question})
print(response["result"])
----------------------------------------------------------------------------------------------


