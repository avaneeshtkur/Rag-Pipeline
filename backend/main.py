from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from services.youtube_service import fetch_youtube_data
from services.instagram_service import fetch_instagram_data
from utils.engagement import compute_engagement_rate
from services.chunker import chunk_transcript
from services.embedder import get_embedder
from services.vector_store import ingest_chunks

app = FastAPI(title="RAG Video Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class IngestRequest(BaseModel):
    youtube_url: str
    instagram_url: str

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
