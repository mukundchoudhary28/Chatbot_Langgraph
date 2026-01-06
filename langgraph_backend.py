from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat(state: AgentState):
    message = state['messages']
    response = model.invoke(message)
    return {"messages": [response]}

graph = StateGraph(AgentState)

graph.add_node("chat",chat)
graph.add_edge(START, "chat")
graph.add_edge("chat",END)

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
memory = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=memory)

def retrieve_threads():
    all_threads = set()
    for checkpoint in memory.list(None):
        print(checkpoint)
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)

