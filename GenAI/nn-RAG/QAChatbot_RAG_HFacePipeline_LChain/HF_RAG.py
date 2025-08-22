from langchain_community.document_loaders import UnstructuredURLLoader
urls = ['https://www.livemint.com/economy/budget-2024-key-highlights-live-updates-nirmala-sitharaman-infrastructure-defence-income-tax-modi-budget-23-july-11721654502862.html',
        'https://cleartax.in/s/budget-2024-highlights',
        'https://www.hindustantimes.com/budget',
        'https://economictimes.indiatimes.com/news/economy/policy/budget-2024-highlights-india-nirmala-sitharaman-capex-fiscal-deficit-tax-slab-key-announcement-in-union-budget-2024-25/articleshow/111942707.cms?from=mdr']
loader = UnstructuredURLLoader(urls=urls)
data = loader.load() 
len(data)

from langchain.text_splitter import RecursiveCharacterTextSplitter
# split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)
print("Total number of documents: ",len(docs))

# Embedding models: https://python.langchain.com/v0.1/docs/integrations/text_embedding/
# Let's load the Hugging Face Embedding class.  sentence_transformers
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
vector = embeddings.embed_query("hello, world!")
vector[:5]

from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=docs, embedding=HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("Budget highlights")
len(retrieved_docs)
print(retrieved_docs[2].page_content)

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#model_id = "meta-llama/Meta-Llama-3-8B"
model_id = "tiiuae/falcon-7b"

text_generation_pipeline = pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, max_new_tokens=400, device=0)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:
{context}

</s>
<|user|>
{question}
</s>

<|assistant|>
 """
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
llm_chain = prompt | llm | StrOutputParser()

from langchain_core.runnables import RunnablePassthrough
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
question = "2024 Budget Highlights"
rag_chain.invoke(question)

question = "2024 Budget Highlights"
response = rag_chain.invoke(question)
# Making the response readable
response = response.replace("</s>", "").strip()
print("Response:", response)

question = "What is the Union Budget?"
response = rag_chain.invoke(question)
# Making the response readable
response = response.replace("</s>", "").strip()
print("Response:", response)

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:
{context}
</s>
<|user|>
{question}
</s>
<|assistant|>
 """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
llm_chain = prompt | llm | StrOutputParser()
from langchain_core.runnables import RunnablePassthrough
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

question = "What is the Union Budget?"
response = rag_chain.invoke(question)
# Making the response readable
response = response.replace("</s>", "").strip()
print("Response:", response)
