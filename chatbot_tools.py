from typing import Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv

load_dotenv()

# Tools
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

tools = [wikipedia_tool, arxiv_tool]


# State
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]


# Graph
from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(State)


# LLM
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition

grok_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=grok_api_key
)

llm_with_tools = llm.bind_tools(tools)


# Chatbot Node
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)


# Tool Node
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


# Graph Flow
graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph_builder.add_edge("tools", "chatbot")

graph_builder.add_edge("chatbot", END)


# Compile
graph = graph_builder.compile()


# Run chatbot
while True:
    user_input = input("User: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )

    print("Assistant:", result["messages"][-1].content)