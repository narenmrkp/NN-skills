
!pip install langgraph-supervisor langchain-openai
from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent


# Initialize OpenAI model
model = ChatOpenAI(temperature=0, api_key="sk-pro*********************QHrtvM7", model="gpt-4o-mini")
!pip install -U duckduckgo-search
!pip install langchain langchain_community langgraph
from langchain_community.tools import DuckDuckGoSearchRun

def search_duckduckgo(query: str):
    """Searches DuckDuckGo using LangChain's DuckDuckGoSearchRun tool."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

# Example usage
# result = search_duckduckgo("what are AI agent")
# print(result)

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)
research_agent = create_react_agent(
    model=model,
    tools=[search_duckduckgo],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)
# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)
# Compile and run
app = workflow.compile()

result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what is quantum computing?"
        }
    ]
})
result
for m in result["messages"]:
    m.pretty_print()
  
# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what is the weather in delhi today. Multiply it by 2 and add 5"
        }
    ]
})

for m in result["messages"]:
    m.pretty_print()
