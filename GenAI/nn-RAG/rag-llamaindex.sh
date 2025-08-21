    # RAG with Llamaindex with No API key   (Ref Link: https://github.com/krishnaik06/Llamindex-Projects/tree/main/Basic%20Rag)
!pip install pypdf
!pip install -q transformers einops accelerate langchain bitsandbytes
!pip install install sentence_transformers
!pip install llama_index

from llama_index import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt


documents=SimpleDirectoryReader("/content/data").load_data()
documents

system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
!huggingface-cli login

import torch
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs = {"torch_dtype": torch.float16 , "load_in_8bit":True}
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding
embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)
index=VectorStoreIndex.from_documents(documents,service_context=service_context)
query_engine=index.as_query_engine()
response=query_engine.query("what is YOLO?") --> print(response)
---------------------------------------------------------------------------------------------------------
    # RAG with Llamaindex with OpenAI API key   (Ref Link: https://github.com/krishnaik06/Llamindex-Projects/tree/main/Basic%20Rag)
create data folder with our pdf files and create .env file with our openai API key.
1. requirements.txt file:
llama-index
openai
pypdf
python-dotenv

2. test.ipynb file:
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
from llama_index import VectorStoreIndex,SimpleDirectoryReader
documents=SimpleDirectoryReader("data").load_data()
index=VectorStoreIndex.from_documents(documents,show_progress=True)
query_engine=index.as_query_engine()
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

retriever=VectorIndexRetriever(index=index,similarity_top_k=4)
postprocessor=SimilarityPostprocessor(similarity_cutoff=0.80)
query_engine=RetrieverQueryEngine(retriever=retriever,
                                  node_postprocessors=[postprocessor])
response=query_engine.query("What is attention is all yopu need?")
from llama_index.response.pprint_utils import pprint_response
pprint_response(response,show_source=True)
print(response)
# the above code with persistent memory:
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What are transformers?")
print(response)
-----------------------------------------------------------------------------------------------------------------
