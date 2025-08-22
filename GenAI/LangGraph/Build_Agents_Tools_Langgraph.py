!pip install --upgrade langchain langchain-community langgraph  openai langchain_openai
     
from IPython.display import Image,display
from langgraph.graph import StateGraph,START
from langchain_openai import ChatOpenAI
import requests
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
     
class State(TypedDict):
    messages: Annotated[list, add_messages]
     
!pip install -U duckduckgo-search
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
search.invoke("Obama's first name?")
from langchain_community.tools import DuckDuckGoSearchRun

def search_duckduckgo(query: str):
    """Searches DuckDuckGo using LangChain's DuckDuckGoSearchRun tool."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

# Example usage
result = search_duckduckgo("what are AI agent")
print(result)

def multiply(a:int,b:int) -> int:
    """
    Multiply a and b
    """
    return a* b

def add(a:int,b:int) -> int:
    """
    Adds a and b
    """
    return a + b
     
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0, api_key='sk-proj-************************QHrtvM7', model="gpt-4o-mini")
llm.invoke('hello').content
tools = [search_duckduckgo, add, multiply]
llm_with_tools = llm.bind_tools(tools)
     
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

from langgraph.prebuilt import ToolNode, tools_condition
graph_builder = StateGraph(State)
# Define nodes
graph_builder.add_node("assistant",chatbot)
graph_builder.add_node("tools",ToolNode(tools))

#define edges
graph_builder.add_edge(START,"assistant")
graph_builder.add_conditional_edges("assistant",tools_condition)
graph_builder.add_edge("tools","assistant")

react_graph=graph_builder.compile()
     
# To see the graph’s connection visually
display(Image(react_graph.get_graph().draw_mermaid_png()))

response = react_graph.invoke({"messages": [HumanMessage(content="what is the weather in delhi. Multiply it by 2 and add 5.")]})
print(response["messages"])

