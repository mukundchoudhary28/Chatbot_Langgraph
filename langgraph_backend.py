from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import requests
import random

load_dotenv()

#Tools

# web_search = TavilySearch(max_results=3)
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


tools = [search_tool, calculator, get_stock_price]

model = ChatGroq(model="llama-3.3-70b-versatile")

model_with_tools = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat(state: AgentState):
    message = state['messages']
    response = model_with_tools.invoke(message)
    return {"messages": [response]}

graph = StateGraph(AgentState)

graph.add_node("chat",chat)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools","chat")

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
memory = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=memory)

def retrieve_threads():
    all_threads = set()
    for checkpoint in memory.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)
