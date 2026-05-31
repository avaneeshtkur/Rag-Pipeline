from dotenv import load_dotenv
load_dotenv()  # Load .env before any other imports that need API keys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import asyncio, uuid, json, os
import httpx
from collections import defaultdict
from typing import AsyncGenerator

# In-memory conversation history per session
# Structure: { session_id: [ {"role": "user", "content": "..."}, ... ] }
conversation_store: dict = defaultdict(list)

from services.youtube_service import fetch_youtube_data
from services.instagram_service import fetch_instagram_data
from utils.engagement import compute_engagement_rate
from services.chunker import chunk_transcript
from services.embedder import get_embedder
from services.vector_store import ingest_chunks
from graph.graph import GRAPH_APP

app = FastAPI(title="RAG Video Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],      # Required for SSE headers to reach the browser
)

class IngestRequest(BaseModel):
    youtube_url: str
    instagram_url: str

class ChatRequest(BaseModel):
    session_id: str
    question: str
    video_a_metadata: dict
    video_b_metadata: dict

@app.post("/api/ingest")
async def ingest_videos(req: IngestRequest):
    session_id = str(uuid.uuid4())
    
    video_a_data = fetch_youtube_data(req.youtube_url)
    video_b_data = fetch_instagram_data(req.instagram_url)
    
    video_a_data["engagement_rate"] = compute_engagement_rate(
        video_a_data["likes"], video_a_data["comments"], video_a_data["views"]
    )
    video_b_data["engagement_rate"] = compute_engagement_rate(
        video_b_data["likes"], video_b_data["comments"], video_b_data["views"]
    )
    
    chunks_a = chunk_transcript(video_a_data["transcript"], "A", video_a_data)
    chunks_b = chunk_transcript(video_b_data["transcript"], "B", video_b_data)
    
    embedder = get_embedder()
    all_chunks = chunks_a + chunks_b
    ingest_chunks(all_chunks, embedder, session_id)
    
    return {
        "session_id": session_id,
        "video_a": video_a_data,
        "video_b": video_b_data
    }

def format_sources(docs):
    return [
        {
            "video_id": doc.metadata.get("video_id", "?"),
            "chunk_index": i,
            "text_preview": doc.page_content[:100] + "...",
            "creator": doc.metadata.get("creator", "")
        }
        for i, doc in enumerate(docs)
    ]

def detect_video_filter(question: str):
    """Returns 'A', 'B', or None based on which video the question mentions."""
    q = question.lower()
    mentions_a = "video a" in q
    mentions_b = "video b" in q
    if mentions_a and not mentions_b:
        return "A"
    if mentions_b and not mentions_a:
        return "B"
    return None  # search both

async def stream_ollama(messages: list) -> AsyncGenerator[str, None]:
    """
    Streams tokens directly from Ollama's /api/chat endpoint.
    messages: list of {"role": "...", "content": "..."} dicts.
    Yields one string token at a time.
    """
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024
        }
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{ollama_url}/api/chat", json=payload) as response:
            if response.status_code != 200:
                yield f"[Error: Ollama returned status {response.status_code}]"
                return
            async for raw_line in response.aiter_lines():
                if not raw_line.strip():
                    continue
                try:
                    data = json.loads(raw_line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

from services.vector_store import retrieve_chunks

@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def generate():
        session_id = req.session_id
        question = req.question
        video_a_meta = req.video_a_metadata
        video_b_meta = req.video_b_metadata

        # --- STEP 1: Retrieve relevant chunks from ChromaDB ---
        video_filter = detect_video_filter(question)
        try:
            docs = retrieve_chunks(session_id, question, video_filter, k=5)
        except Exception as e:
            yield {"data": json.dumps({"type": "token", "content": f"[Retrieval error: {str(e)}]"})}
            yield {"data": json.dumps({"type": "done"})}
            return

        # --- STEP 2: Build context string from retrieved chunks ---
        context_parts = []
        for i, doc in enumerate(docs):
            vid = doc.metadata.get("video_id", "?")
            context_parts.append(f"[Video {vid}, Chunk {i+1}]:\n{doc.page_content}")
        context_text = "\n\n".join(context_parts) if context_parts else "No relevant transcript chunks found."

        # --- STEP 3: Build system prompt with full metadata ---
        def fmt(n):
            try:
                return f"{int(n):,}"
            except Exception:
                return str(n) if n else "N/A"

        system_prompt = f"""You are a social media content analyst specializing in engagement optimization.
You have full access to transcripts and metadata for two videos.

VIDEO A:
  Creator: {video_a_meta.get('creator', 'Unknown')} | Followers: {fmt(video_a_meta.get('followers', 0))}
  Views: {fmt(video_a_meta.get('views', 0))} | Likes: {fmt(video_a_meta.get('likes', 0))} | Comments: {fmt(video_a_meta.get('comments', 0))}
  Engagement Rate: {video_a_meta.get('engagement_rate', 0):.4f}%
  Duration: {video_a_meta.get('duration', 'N/A')}s | Uploaded: {video_a_meta.get('upload_date', 'N/A')}
  Hashtags: {video_a_meta.get('hashtags', '')}

VIDEO B:
  Creator: {video_b_meta.get('creator', 'Unknown')} | Followers: {fmt(video_b_meta.get('followers', 0))}
  Views: {fmt(video_b_meta.get('views', 0))} | Likes: {fmt(video_b_meta.get('likes', 0))} | Comments: {fmt(video_b_meta.get('comments', 0))}
  Engagement Rate: {video_b_meta.get('engagement_rate', 0):.4f}%
  Duration: {video_b_meta.get('duration', 'N/A')}s | Uploaded: {video_b_meta.get('upload_date', 'N/A')}
  Hashtags: {video_b_meta.get('hashtags', '')}

Rules:
- Cite the video when referencing it: [Video A] or [Video B].
- Cite transcript chunks as: [Video A, Chunk N].
- Use exact numbers from the metadata above when answering stats questions.
- Be specific and actionable. Do not be vague.
- If context is insufficient, say so clearly rather than guessing."""

        # --- STEP 4: Build message list with memory ---
        history = conversation_store[session_id]  # previous turns
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history[-6:]  # last 3 turns (6 messages)
            + [{"role": "user", "content": f"Transcript context:\n{context_text}\n\nQuestion: {question}"}]
        )

        # --- STEP 5: Stream tokens from Ollama directly ---
        full_response = ""
        try:
            async for token in stream_ollama(messages):
                full_response += token
                yield {"data": json.dumps({"type": "token", "content": token})}
        except Exception as e:
            yield {"data": json.dumps({"type": "token", "content": f"\n[Stream error: {str(e)}]"})}

        # --- STEP 6: Save to conversation memory ---
        conversation_store[session_id].append({"role": "user", "content": question})
        conversation_store[session_id].append({"role": "assistant", "content": full_response})

        # --- STEP 7: Send sources ---
        if docs:
            sources = [
                {
                    "video_id": doc.metadata.get("video_id"),
                    "chunk_index": i,
                    "text_preview": doc.page_content[:100] + "...",
                    "creator": doc.metadata.get("creator", "")
                }
                for i, doc in enumerate(docs)
            ]
            yield {"data": json.dumps({"type": "sources", "content": sources})}

        # --- STEP 8: Signal done ---
        yield {"data": json.dumps({"type": "done"})}

    return EventSourceResponse(generate())

