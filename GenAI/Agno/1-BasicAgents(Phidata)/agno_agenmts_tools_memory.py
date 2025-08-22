!pip install -U -q agno
!pip install python-dotenv

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My Drive/

from dotenv import load_dotenv
import os
# Load API key from .env
#load_dotenv()
load_dotenv(dotenv_path="/content/drive/My Drive/agentic_ai_tutorials/.env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

from agno.agent import Agent
from agno.models.openai import OpenAIChat
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You’re a cheerful AI pal who loves a good chat!",
    markdown=True
)
agent.print_response("Summarize the story of 'The Lion King.", stream=True)

!pip install -U googlesearch-python pycountry
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.googlesearch import GoogleSearchTools
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[
        GoogleSearchTools(
            stop_after_tool_call_tools=["google_search"],
            show_result_tools=["google_search"],
        )
    ],
    show_tool_calls=True,
)
agent.print_response("What are the trending AI tools in 2025?", stream=True)

from agno.agent import Agent
from agno.tools.python import PythonTools
agent = Agent(tools=[PythonTools()], show_tool_calls=True)
agent.print_response("Write a function to reverse a string without using slicing")


from datetime import datetime
from agno.agent import Agent

def get_today_date() -> str:
    """
    Returns today's date
    """
    today = datetime.now().strftime("Today is %B %d, %Y")
    return today
get_today_date()
agent = Agent(
    tools=[get_today_date],
    show_tool_calls=True,
    markdown=True
)
# Ask a question to trigger the tool
agent.print_response("What is today’s date?", stream=True)

import httpx
from agno.agent import Agent

def search_wikipedia(topic: str = "Machine learning") -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '%20')}"
    response = httpx.get(url, timeout=5.0)  # Make the HTTP request. We send a GET request using httpx.
    data = response.json()

    # Extract the summary from response - extract contains a short summary of the topic.We also get the page’s title from the response.
    if "extract" in data:
        title = data.get("title", topic)
        extract = data["extract"]
        return f"**{title}**:\n{extract}"
    else:
        return "Sorry, I couldn't find anything on that topic."
agent = Agent(
    tools=[search_wikipedia],
    show_tool_calls=True,
    markdown=True
)
# Example prompt
agent.print_response("Explain quantum computing in simple words", stream=True)

# Agno with Memory
from agno.agent import Agent
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.storage.sqlite import SqliteStorage
from rich.pretty import pprint

db_file = "tmp/agent_storage.db"
user_id = "Aarohi"
memory = Memory(
    db=SqliteMemoryDb(table_name="user_memories", db_file=db_file),
)
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You’re an AI with a memory!",
    memory=memory,
    storage=SqliteStorage(table_name="agent_sessions", db_file=db_file),
    enable_user_memories=True,
    add_history_to_messages=True,
    num_history_runs=3,
    session_id="my_chat_session",
    markdown=True,
)
agent.print_response("I love South Indian food. What’s your favorite cuisine?", user_id=user_id, stream=True)
print("\n Current **memories** about the user:")
user_memories = memory.get_user_memories(user_id=user_id)
pprint([{"memory": m.memory, "topics": m.topics} for m in user_memories])

agent.print_response("What did I just say I love and also I love hill stations", user_id=user_id)
print("\n Current **memories** about the user:")
user_memories = memory.get_user_memories(user_id=user_id)
pprint([{"memory": m.memory, "topics": m.topics} for m in user_memories])

