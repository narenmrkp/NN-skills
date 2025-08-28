1 File-based Loaders
Load documents from local file formats.

Text Files → TextLoader, UnstructuredFileLoader
CSV Files → CSVLoader, UnstructuredCSVLoader
JSON Files → JSONLoader
PDF Files → PyPDFLoader, PDFPlumberLoader, PyMuPDFLoader
Word Documents → Docx2txtLoader, UnstructuredWordDocumentLoader
Excel Files → UnstructuredExcelLoader, PandasExcelLoader
HTML/XML → UnstructuredHTMLLoader, UnstructuredXMLLoader, BSHTMLLoader
----------------------------------------------------------------------------------
1. Text Loaders
pip install langchain_openai
pip install langchain_community
pip install ipykernel
from langchain_community.document_loaders import TextLoader
loader = TextLoader("files/sample.txt")
doc = loader.load()
doc
doc[0].metadata
print(doc[0].page_content)
----------------------------------------------------------------------------------
2. PDF Loaders
pip install pypdf
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('files/sample.pdf')
doc = loader.load()
print(doc[0].page_content)
--------------------------------------------------------------------------------
3. docx loaders
pip install docx2txt
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("files/sample.docx")
documents = loader.load()
print(documents[0].page_content)
-----------------------------------------------------------------------------
4. CSV files Loaders
pip install "unstructured[all-docs]" or pip install "unstructured[all]"
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path="files/test_data.csv")
doc = loader.load()
print(doc[0].page_content)
-----------------------------------------------------------------------------
5. Excel Files Loaders
pip install networkx
pip install msoffcrypto-tool
pip install openpyxl
pip install pandas
from langchain_community.document_loaders import UnstructuredExcelLoader
loader = UnstructuredExcelLoader("files/test_data.xlsx")
doc = loader.load()
print(doc[0].page_content)
------------------------------------------------------------------------------
6. JSon Files Loaders
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="files/test_data.json",
    jq_schema=".[]",
    text_content=False,
)
docs = loader.load()

for doc in docs:
    print(doc.page_content)
--------------------------------------------------------------------------------
7. HTML/XML files Loaders
from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredXMLLoader
# Unstructured HTML Loader
loader = UnstructuredHTMLLoader("files/test_data.html")
docs = loader.load()
print(docs[0].page_content)

# Unstructured XML Loader
loader = UnstructuredXMLLoader("files/test_data.xml")
docs = loader.load()
print(docs[0].page_content)
---------------------------------------------------------------------------------------
