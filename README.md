# ViralLens — RAG Video Engagement Analyzer

Take two video URLs — one YouTube, one Instagram Reel — and chat with their content. Ask why one performed better, compare the hooks, get the engagement numbers, request specific improvements. The whole thing runs locally. No OpenAI. No API keys. No monthly bill.

I built this in 5 days as part of a technical screening. The spec said to use LangChain/LangGraph, embeddings, and a vector DB — so that's what's here. But I also made a few architectural calls that differed from the obvious path, and I'll explain those below because I think they're the more interesting part.

---

## How It Works

Paste a YouTube URL and a public Instagram Reel URL. The backend:

1. Fetches the YouTube transcript via `youtube-transcript-api` (free, no key, instant). For Instagram, it downloads the audio with `yt-dlp` and transcribes it locally with Whisper.
2. Pulls metadata for both — views, likes, comments, follower count, upload date, duration, hashtags — and computes engagement rate as `(likes + comments) / views × 100`.
3. Splits both transcripts into 500-character chunks with a 50-character overlap and embeds them using a local `sentence-transformers` model (`all-MiniLM-L6-v2`). Stores everything in ChromaDB on disk, tagged with `session_id` and `video_id` so sessions don't bleed into each other.
4. When you ask a question, it embeds the question locally, pulls the top-5 most relevant chunks from ChromaDB, builds a prompt with the full metadata context + retrieved chunks + conversation history, and streams a response from a locally running Ollama LLM.
5. Every answer cites which video and which chunk it came from. The conversation remembers previous turns in the session.

---

## Tech Stack

### LLM: Ollama (`llama3.2:3b` default)

The spec said GPT-4o or Claude were fine. I went with Ollama instead because running this at 1,000 creators a day on OpenAI would cost around ₹25,000/month just in LLM calls. Running it locally on a GPU server costs maybe ₹5,000/month total — for everything. For a product that's going to scale, that's not a minor difference.

`llama3.2:3b` runs on 6GB RAM with no GPU. For better quality, swap to `mistral:7b` in the `.env` — it needs 16GB RAM but the answers are noticeably sharper on complex comparative questions.

### Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

22MB. Runs on CPU. No API key. Cached after first download. I looked at `BAAI/bge-small-en-v1.5` too — slightly better retrieval quality in benchmarks, but 33% larger and the difference wasn't meaningful for transcript content where the semantic signal is pretty surface-level anyway.

OpenAI's `text-embedding-3-small` would cost around ₹1.70 per million tokens — not much, but zero is better than not-much when you're making the same call thousands of times a day.

### Vector DB: ChromaDB (local, persistent to disk)

I considered Qdrant. Qdrant is the better database — faster filtering, cleaner API, proper horizontal scaling. But ChromaDB requires zero infrastructure for a demo and early production. I didn't want to introduce a Docker dependency or a cloud signup just to show the thing working. The migration path to Qdrant is a 20-line change in `vector_store.py` when it's needed.

At around 40,000 vectors per day (1,000 creators × 2 videos × ~20 chunks each), ChromaDB on a decent disk starts to feel it after a couple of weeks. That's when you move to Qdrant.

### Transcription: Local Whisper (`openai-whisper` library, not the API)

The Whisper API charges about ₹0.50 per minute of audio. Instagram Reels average maybe 45 seconds — that's ₹0.37 per video, ₹370 per day at 1,000 reels. The local library uses the same model weights, runs in your process, and costs nothing. The trade-off is it takes 20–40 seconds per minute of audio on CPU (versus ~2 seconds on the API). I think that's acceptable for an async ingestion flow where the user expects to wait.

### Streaming: Direct Ollama `/api/chat` HTTP, not LangGraph `astream_events`

This one took me a day to figure out. LangGraph's `astream_events` works cleanly with OpenAI's event format. With Ollama via `ChatOllama`, the `on_chat_model_stream` events either don't fire or arrive malformed depending on the LangGraph version. I spent more time than I'd like to admit trying to make that work before just calling Ollama's streaming endpoint directly via `httpx`. It's 20 lines, it works first try, and it's far easier to debug.

LangGraph is still used for the retrieval graph and state management. Just not for streaming.

---

## Setup

You need Ollama running before the backend starts. That's the one external process.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh   # macOS/Linux
# Windows: https://ollama.com/download

# Pull the model (2GB download, happens once)
ollama pull llama3.2:3b

# Start Ollama — keep this running
ollama serve
```

Then:

```bash
git clone https://github.com/your-username/rag-video-analyzer
cd rag-video-analyzer

# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install && npm run dev
```

App runs at `http://localhost:(frontend port)`. Backend at `http://localhost:(backend port)`.

---

## Environment Configuration

No API keys. The `.env` is just local configuration (cheapest and most efficient cost free setup):
