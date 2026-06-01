# ViralLens-Rag pipeline
RAG Video Engagement Analyzer
Take two video URLs — one YouTube, one Instagram Reel — and chat with their content. Ask why one performed better, compare the hooks, get the engagement numbers, request specific improvements. The whole thing runs locally. No OpenAI. No API keys. No monthly bill.
I built this in 5 days as part of a technical screening. The spec said to use LangChain/LangGraph, embeddings, and a vector DB — so that's what's here. But I also made a few architectural calls that differed from the obvious path, and I'll explain those below because I think they're the more interesting part.

How it works
Paste a YouTube URL and a public Instagram Reel URL. The backend:

1.Fetches the YouTube transcript via youtube-transcript-api (free, no key, instant). For Instagram, it downloads the audio with yt-dlp and transcribes it locally with Whisper.
2.Pulls metadata for both — views, likes, comments, follower count, upload date, duration, hashtags — and computes engagement rate as (likes + comments) / views × 100.
3>Splits both transcripts into 500-character chunks with a 50-character overlap and embeds them using a local sentence-transformers model (all-MiniLM-L6-v2). Stores everything in ChromaDB on disk, tagged with session_id and video_id so sessions don't bleed into each other.
4.When you ask a question, it embeds the question locally, pulls the top-5 most relevant chunks from ChromaDB, builds a prompt with the full metadata context + retrieved chunks + conversation history, and streams a response from a locally running Ollama LLM.
5.Every answer cites which video and which chunk it came from. The conversation remembers previous turns in the session.
