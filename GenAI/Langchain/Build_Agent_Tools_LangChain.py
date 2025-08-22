from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

search_tool = SerpAPIWrapper()
llm = OpenAI(temperature=0.5)

tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool to perform web searches."
    )
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
response = agent.run("What’s the weather in Delhi?")
print(response)
