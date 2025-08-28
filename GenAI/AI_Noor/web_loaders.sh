1. Web Based URL Loaders
pip install -qU langchain-community beautifulsoup4
import os

# Set the USER_AGENT environment variable
"""os.environ["USER_AGENT"] = "..."

Many websites block requests from scripts/bots if they don’t look like a real browser.

USER_AGENT is a string that identifies your browser type to the server (e.g., Chrome on Windows 10).

By setting it, you pretend to be a normal browser, which helps bypass “bot-blocking” filters."""

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                          "AppleWebKit/537.36 (KHTML, like Gecko) " \
                          "Chrome/120.0.0.0 Safari/537.36"

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://vtohal.medium.com/mastering-mcqs-the-ultimate-guide-26a180865cd2")
docs = loader.load()
print(docs[0].page_content)
----------------------------------------------------------------------------------------------
2. Unstructured URL Loaders
from langchain_community.document_loaders import UnstructuredURLLoader
urls = ["https://python.langchain.com/docs/integrations/document_loaders/"]
loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()
print(f"Number of docs: {len(docs)}")
print(docs[0].page_content)
----------------------------------------------------------------------------------------------
3. Selenium URL LOaders
from langchain_community.document_loaders import SeleniumURLLoader
urls = ["https://www.investopedia.com/articles/active-trading/111115/why-all-worlds-top-10-companies-are-american.asp"]
loader = SeleniumURLLoader(urls=urls,headless=True, browser="chrome")
docs = loader.load()
print(docs[0].page_content)
---------------------------------------------------------------------------------------------
