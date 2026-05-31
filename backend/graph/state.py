from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class GraphState(TypedDict):
    session_id: str
    messages: Annotated[List[BaseMessage], operator.add]  # chat history
    question: str
    retrieved_docs: list
    answer: str
    video_a_metadata: dict
    video_b_metadata: dict
