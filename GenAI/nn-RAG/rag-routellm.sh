    # RAG with llm-Router with Groq (ipynb file)
!pip install llama-index==0.12.9 llama-index-llms-ollama==0.5.0 llama-index-embeddings-huggingface==0.4.0 llama-index-llms-groq==0.3.1
import os
import nest_asyncio
nest_asyncio.apply()
from llama_index.core import SimpleDirectoryReader
# load teh document
documents = SimpleDirectoryReader(input_files=["attention_is_all_you_need.pdf"]).load_data()
len(documents)
from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)
from llama_index.core import Settings
#from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#Settings.llm = Ollama(model="gemma:2b")
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
Settings.llm = Groq(model="llama-3.1-8b-instant")
Settings.embed_model = HuggingFaceEmbedding()
from llama_index.core import SummaryIndex, VectorStoreIndex
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True
)
vector_query_engine = vector_index.as_query_engine()
from llama_index.core.tools import QueryEngineTool
summary_tool = QueryEngineTool.from_defaults(
    query_engine = summary_query_engine,
    description = (
        "Useful for summarization related to the  given context"
    )
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine = vector_query_engine,
    description = (
        "Useful for retrieving specific context from the given context based on the given question"
    )
)
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
query_engine = RouterQueryEngine(
    selector = LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool
    ],
    verbose=True
)
response = query_engine.query("Summarize the given document")
print(response)
------------------------------------------------------------------------------------------------------------
