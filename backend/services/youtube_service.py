import re
import os
import uuid
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

def get_video_id(url: str) -> str:
    """Extracts the YouTube video ID from a URL."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid YouTube URL: {url}")

def _whisper_fallback(video_id: str) -> str:
    """Downloads audio and transcribes with local Whisper when captions are unavailable."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    audio_path = f"temp_yt_audio_{uuid.uuid4().hex}.m4a"
    ydl_opts = {
        'quiet': True,
        'format': 'bestaudio/best',
        'outtmpl': audio_path
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path)
        return " ".join([seg.text for seg in segments]).strip()
    except Exception as e:
        print(f"Warning: Whisper fallback failed for {video_id}: {e}")
        return ""
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

def get_transcript(video_id: str) -> str:
    """Fetches the transcript for a given video ID. Falls back to Whisper if no captions."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([chunk['text'] for chunk in transcript_list])
        return full_transcript
    except Exception as e:
        print(f"Warning: Could not fetch transcript for {video_id}: {e}")
        print(f"Attempting Whisper fallback transcription...")
        return _whisper_fallback(video_id)

def get_metadata(url: str) -> dict:
    """Fetches metadata using yt-dlp."""
    ydl_opts = { 'quiet': True, 'skip_download': True, 'dumpjson': True }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info

def fetch_youtube_data(url: str) -> dict:
    """Fetches transcript and metadata for a YouTube video."""
    video_id = get_video_id(url)
    transcript = get_transcript(video_id)
    metadata = get_metadata(url)
    
    return {
        "url": url,
        "video_id": video_id,
        "transcript": transcript,
        "title": metadata.get("title", ""),
        "creator": metadata.get("uploader", ""),
        "views": metadata.get("view_count", 0),
        "likes": metadata.get("like_count", 0),
        "comments": metadata.get("comment_count", 0),
        "followers": metadata.get("channel_follower_count", 0),
        "hashtags": metadata.get("tags", []) or [],
        "upload_date": metadata.get("upload_date", ""),
        "duration": metadata.get("duration", 0)
    }
