# with Google API key
pip install langchain-google-genai python-dotenv ipykernel
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from dotenv import load_dotenv

# load env
load_dotenv()

# load model 
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=os.getenv('GOOGLE_API_KEY'))

# user query 
user_query = "what is black hole"
output = model.invoke(user_query).content
print(output)
----------------------------------------------------------------------------
# with OpenAI API key
pip install langchain_openai
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
result = model.invoke("What is black hole")
print(result.content)
--------------------------------------------------------------------------------
# With Anthropic API key
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')
result = model.invoke('What is black hole')
-----------------------------------------------------------------------------------
print(result.content)
