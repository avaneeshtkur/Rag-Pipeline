from .state import GraphState
from services.embedder import get_embedder
from services.vector_store import get_retriever
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import os

SYSTEM = """
You are a social media content analyst specializing in engagement optimization.
You have access to transcripts and metadata from two social media videos:

VIDEO A:
  Creator: {video_a_creator} | Followers: {video_a_followers:,}
  Views: {video_a_views:,} | Likes: {video_a_likes:,} | Comments: {video_a_comments:,}
  Engagement Rate: {video_a_er:.4f}%
  Duration: {video_a_duration}s | Upload Date: {video_a_date}
  Hashtags: {video_a_hashtags}

VIDEO B:
  Creator: {video_b_creator} | Followers: {video_b_followers:,}
  Views: {video_b_views:,} | Likes: {video_b_likes:,} | Comments: {video_b_comments:,}
  Engagement Rate: {video_b_er:.4f}%
  Duration: {video_b_duration}s | Upload Date: {video_b_date}
  Hashtags: {video_b_hashtags}

Rules:
1. Always cite which video a piece of information comes from (e.g., "[Video A]").
2. When using transcript content, reference it as: [Video A, Chunk N].
3. Be specific and data-driven. Use actual numbers.
4. For improvement suggestions, be actionable and concrete.
5. If asked about metadata (views, likes, etc.), answer directly from the
   stats above — do NOT search the transcript for these.
6. If the retrieved context is insufficient, say so clearly.
"""

def retrieve_context(state: GraphState):
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    
    # Simple keyword matching to filter
    video_filter = None
    if "Video A" in question or "video a" in question.lower():
        video_filter = "A"
    elif "Video B" in question or "video b" in question.lower():
        video_filter = "B"
        
    embedder = get_embedder()
    retriever = get_retriever(embedder, session_id, video_filter)
    docs = retriever.invoke(question)
    
    return {"retrieved_docs": docs}

async def generate_answer(state: GraphState):
    question = state.get("question", "")
    retrieved_docs = state.get("retrieved_docs", [])
    video_a_meta = state.get("video_a_metadata", {})
    video_b_meta = state.get("video_b_metadata", {})
    messages = state.get("messages", [])
    
    # Format system prompt
    formatted_system = SYSTEM.format(
        video_a_creator=video_a_meta.get("creator", ""),
        video_a_followers=video_a_meta.get("followers", 0),
        video_a_views=video_a_meta.get("views", 0),
        video_a_likes=video_a_meta.get("likes", 0),
        video_a_comments=video_a_meta.get("comments", 0),
        video_a_er=video_a_meta.get("engagement_rate", 0.0),
        video_a_duration=video_a_meta.get("duration", 0),
        video_a_date=video_a_meta.get("upload_date", ""),
        video_a_hashtags=", ".join(video_a_meta.get("hashtags", [])),
        
        video_b_creator=video_b_meta.get("creator", ""),
        video_b_followers=video_b_meta.get("followers", 0),
        video_b_views=video_b_meta.get("views", 0),
        video_b_likes=video_b_meta.get("likes", 0),
        video_b_comments=video_b_meta.get("comments", 0),
        video_b_er=video_b_meta.get("engagement_rate", 0.0),
        video_b_duration=video_b_meta.get("duration", 0),
        video_b_date=video_b_meta.get("upload_date", ""),
        video_b_hashtags=", ".join(video_b_meta.get("hashtags", []))
    )
    
    # Format retrieved context
    context_str = "\n".join([f"[Video {doc.metadata.get('video_id', '?')}, Chunk {i}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
    user_prompt = f"Context from transcript search:\n{context_str}\n\nQuestion: {question}"
    
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        streaming=True,
        temperature=0.3,
    )
    
    sys_msg = SystemMessage(content=formatted_system)
    user_msg = HumanMessage(content=user_prompt)
    
    # Keep last 6 messages in context window
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    
    input_messages = [sys_msg] + recent_messages + [user_msg]
    
    response = await llm.ainvoke(input_messages)
    
    return {"answer": response.content, "messages": [user_msg, response]}
