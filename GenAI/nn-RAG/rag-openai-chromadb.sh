Ref link: https://github.com/ThomasJanssen-tech/Retrieval-Augmented-Generation
create data folder with our pdf file inside it.
create requirements.txt, ask.py, fill_db.py files with code as below.
Creating Virtual Env:
python -m venv venv --> venv\Scripts\Activate (for Windows) --> source venv/bin/activate (for Mac)
create .env file with our openai API key
python fill_db.py --> python ask.py

# ask.py file:
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# setting the environment
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="growing_vegetables")

user_query = input("What do you want to know about growing vegetables?\n\n")
results = collection.query(
    query_texts=[user_query],
    n_results=4
)
print(results['documents'])
#print(results['metadatas'])
client = OpenAI()
system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make thins up.
If you don't know the answer, just say: I don't know
--------------------
The data:
"""+str(results['documents'])+"""
"""
#print(system_prompt)
response = client.chat.completions.create(
    model="gpt-4o",
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_query}    
    ]
)
print("\n\n---------------------\n\n")
print(response.choices[0].message.content)

# fill_db.py file:
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# setting the environment
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="growing_vegetables")

# loading the document
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)

# preparing to be added in chromadb
documents = []
metadata = []
ids = []

i = 0
for chunk in chunks:
    documents.append(chunk.page_content)
    ids.append("ID"+str(i))
    metadata.append(chunk.metadata)

    i += 1

# adding to chromadb
collection.upsert(
    documents=documents,
    metadatas=metadata,
    ids=ids
)

# requirements.txt file (please refer github repo file, bcoz its too lengthy to write here)
pip install -r requirements.txt
