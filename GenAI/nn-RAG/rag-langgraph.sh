    # Build RAG with LangGraph
!pip install langchain langchain-core langchain_community langgraph langchain-huggingface transformers torch unstructured langchain_chroma
from langchain_community.document_loaders import UnstructuredURLLoader

urls = ['https://langchain-ai.github.io/langgraph/tutorials/introduction/']
loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print( "Total number of documents: ", len(all_splits) )
all_splits[7]

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
vector = embeddings.embed_query("hello, world!")
vector[:5]

from langchain_chroma import Chroma
from langchain_core.documents import Document
vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#model_id = "meta-llama/Meta-Llama-3-8B"
model_id = "tiiuae/falcon-7b"
# text_generation_pipeline = pipeline(
#     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, max_new_tokens=400, device=0)
text_generation_pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs = {"torch_dtype": torch.bfloat16},
    max_new_tokens=200,
    device=0,
    temperature=0.7,  #  (lower values = more deterministic)
    top_k=50,  # Filters out low-probability tokens
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# from langchain_core.prompts import PromptTemplate
# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum and keep the answer as concise as possible.
# Always say "thanks for asking!" at the end of the answer.
# {context}
# Question: {question}
# Helpful Answer:"""
# prompt = PromptTemplate.from_template(template)

from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

from typing_extensions import List, TypedDict
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"], k=1)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    #return {"answer": response.content}
    return {"answer": response}

from langgraph.graph import START, StateGraph
# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

response = graph.invoke({"question": "what is langgraph?"})
print(response["answer"])
----------------------------------------------------------------------------------------------------------------

