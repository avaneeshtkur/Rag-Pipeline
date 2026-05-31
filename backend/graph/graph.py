from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .nodes import retrieve_context, generate_answer
from .state import GraphState

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve_context)
    builder.add_node("generate", generate_answer)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    # MemorySaver stores conversation state in memory
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

GRAPH_APP = build_graph()
