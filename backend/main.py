from dotenv import load_dotenv
load_dotenv()  # Load .env before any other imports that need API keys

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import asyncio, uuid, json, os, re
import httpx
from collections import defaultdict
from typing import AsyncGenerator

# ── Prompt-injection defence ─────────────────────────────────────────────────
# Regex matches the most common injection trigger phrases.
_INJECTION_RE = re.compile(
    r"("
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?"
    r"|disregard\s+(all\s+)?(previous|above|prior)\s+instructions?"
    r"|forget\s+(everything|all|what\s+i\s+said)"
    r"|you\s+are\s+now\s+"
    r"|act\s+as\s+if\s+you\s+are"
    r"|pretend\s+(to\s+be|you\s+are)"
    r"|your\s+new\s+instructions?"
    r"|override\s+(the\s+)?system"
    r"|jailbreak|DAN\s+mode|developer\s+mode"
    r")",
    re.IGNORECASE,
)
_MAX_QUESTION_LEN = 500  # hard cap — prevents token-flooding attacks

def sanitize_input(text: str) -> str:
    """
    Sanitises user-supplied text before it is sent to the LLM:
    1. Truncates to _MAX_QUESTION_LEN characters.
    2. Replaces known injection trigger phrases with '[blocked]'.
    User content is *also* wrapped in XML delimiters at call-site
    so the model receives it as data, never as instruction.
    """
    text = text.strip()[:_MAX_QUESTION_LEN]
    text = _INJECTION_RE.sub("[blocked]", text)
    return text

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

# ── Preloaded singletons (set during lifespan startup) ──────────────────────
_embedder = None
_whisper_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload heavy models once at startup so requests pay no cold-start cost."""
    global _embedder, _whisper_model
    loop = asyncio.get_event_loop()

    whisper_model_name = os.getenv("WHISPER_MODEL", "tiny")

    print(f"[startup] Loading embedding model…")
    _embedder = await loop.run_in_executor(None, get_embedder)
    print(f"[startup] Embedding model ready.")

    print(f"[startup] Loading Whisper '{whisper_model_name}' model…")
    from faster_whisper import WhisperModel as _WM
    _whisper_model = await loop.run_in_executor(
        None,
        lambda: _WM(whisper_model_name, device="cpu", compute_type="int8")
    )
    print(f"[startup] Whisper model ready.")

    yield  # ← app runs here

    print("[shutdown] Cleanup complete.")

app = FastAPI(title="RAG Video Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],      # Required for SSE headers to reach the browser
)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "ViralLens RAG Video Analyzer API is running.",
        "docs": "/docs",
        "health": "/api/health"
    }


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
    loop = asyncio.get_event_loop()

    # ── Fetch both videos in parallel ───────────────────────────────────────
    video_a_data, video_b_data = await asyncio.gather(
        loop.run_in_executor(None, fetch_youtube_data, req.youtube_url),
        loop.run_in_executor(None, fetch_instagram_data, req.instagram_url),
    )
    
    video_a_data["engagement_rate"] = compute_engagement_rate(
        video_a_data["likes"], video_a_data["comments"], video_a_data["views"]
    )
    video_b_data["engagement_rate"] = compute_engagement_rate(
        video_b_data["likes"], video_b_data["comments"], video_b_data["views"]
    )
    
    chunks_a = chunk_transcript(video_a_data["transcript"], "A", video_a_data)
    chunks_b = chunk_transcript(video_b_data["transcript"], "B", video_b_data)
    
    # Use the preloaded embedder singleton (falls back to get_embedder() on cold start)
    embedder = _embedder if _embedder is not None else get_embedder()
    all_chunks = chunks_a + chunks_b
    await loop.run_in_executor(None, ingest_chunks, all_chunks, embedder, session_id)
    
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

def classify_question(question: str) -> str:
    """Classifies the user question to guide LLM response style."""
    q = question.lower()
    if any(w in q for w in ["similar", "same", "both", "in common",
                              "alike", "shared", "neither"]):
        return "SIMILARITY"
    if any(w in q for w in ["different", "difference", "better",
                              "worse", "compare", "outperform",
                              "more than", "less than"]):
        return "COMPARISON"
    if any(w in q for w in ["engagement rate", "views", "likes",
                              "comments", "followers", "how many",
                              "what is the", "what are the"]):
        return "STATS"
    if any(w in q for w in ["improve", "suggest", "fix", "change",
                              "should", "recommendation", "advice"]):
        return "IMPROVEMENT"
    return "GENERAL"


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
    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.55,
            "num_predict": 512,
            "repeat_penalty": 1.3
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
            # Pass the preloaded singleton so HuggingFaceEmbeddings is never rebuilt per-request
            docs = retrieve_chunks(session_id, question, video_filter, k=5, embedder=_embedder)
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

        system_prompt = f"""You are ViralLens, a social media content analyst. 
Answer questions about two videos using their metadata and transcripts.

VIDEO A:
  Creator: {video_a_meta.get('creator', 'Unknown')}
  Followers: {fmt(video_a_meta.get('followers', 0))}
  Views: {fmt(video_a_meta.get('views', 0))}
  Likes: {fmt(video_a_meta.get('likes', 0))}
  Comments: {fmt(video_a_meta.get('comments', 0))}
  Engagement Rate: {video_a_meta.get('engagement_rate', 0):.4f}%
  Duration: {video_a_meta.get('duration', 'N/A')}s
  Upload Date: {video_a_meta.get('upload_date', 'N/A')}
  Hashtags: {video_a_meta.get('hashtags', '')}

VIDEO B:
  Creator: {video_b_meta.get('creator', 'Unknown')}
  Followers: {fmt(video_b_meta.get('followers', 0))}
  Views: {fmt(video_b_meta.get('views', 0))}
  Likes: {fmt(video_b_meta.get('likes', 0))}
  Comments: {fmt(video_b_meta.get('comments', 0))}
  Engagement Rate: {video_b_meta.get('engagement_rate', 0):.4f}%
  Duration: {video_b_meta.get('duration', 'N/A')}s
  Upload Date: {video_b_meta.get('upload_date', 'N/A')}
  Hashtags: {video_b_meta.get('hashtags', '')}

When answering any question:
- Lead with metadata facts first (views, likes, followers, 
  engagement rate, creator, duration, upload date)
- Use transcript chunks only for questions about spoken 
  content, hooks, tone, or pacing
- Cite sources inline like this: [Video A] or [Video B, Chunk 2]
- Give the final answer only. Never explain your reasoning 
  process. Never number your thoughts. Just answer.
- Keep answers between 3 and 8 sentences"""

        # --- STEP 4: Classify question and build message list with memory ---
        # Sanitise the raw question — strips injection triggers, caps length.
        # The sanitised version is also what gets stored in memory.
        safe_question = sanitize_input(question)
        question_type = classify_question(safe_question)

        # Build the human message with context
        user_message = f"""
  Transcript context:
  {context_text}

  Question: {safe_question}
  """

        # --- STEP 4b: Filter conversation history based on question type ---
        history = conversation_store[session_id]  # previous turns
        if question_type in ("SIMILARITY", "STATS"):
            # Only send user turns from history, not assistant turns.
            # This prevents the model from anchoring on previous answers.
            filtered_history = [
                m for m in history[-6:]
                if m["role"] == "user"
            ]
        else:
            filtered_history = history[-6:]

        messages = (
            [{"role": "system", "content": system_prompt}]
            + filtered_history
            + [{"role": "user", "content": user_message}]
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
        # Store safe_question (sanitised) — never the raw input — so injected
        # content cannot persist into future turns via the history window.
        conversation_store[session_id].append({"role": "user", "content": safe_question})
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

@app.get("/api/health")
async def health():
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{ollama_url}/api/tags")
            ollama_ok = r.status_code == 200
            # Check if the required model is actually pulled
            models = [m["name"] for m in r.json().get("models", [])]
            model_ready = any(model in m for m in models)
    except Exception:
        ollama_ok = False
        model_ready = False

    return {
        "status": "ok" if (ollama_ok and model_ready) else "degraded",
        "ollama_running": ollama_ok,
        "model_ready": model_ready,
        "model": model,
        "message": (
            "All systems ready." if (ollama_ok and model_ready)
            else f"Ollama running but model '{model}' not found. Run: ollama pull {model}" if ollama_ok
            else "Ollama not running. Start it with: ollama serve"
        )
    }
