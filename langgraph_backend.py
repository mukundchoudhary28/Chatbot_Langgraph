from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

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

memory = InMemorySaver()

chatbot = graph.compile(checkpointer=memory)

# CONFIG = {"configurable":{"thread_id":"1"}}
# initial_state = {"messages":[HumanMessage(content="Hi")]}
# response = chatbot.invoke(initial_state, CONFIG)['messages'][-1].content
# print(response)

