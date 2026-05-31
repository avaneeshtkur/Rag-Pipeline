import yt_dlp
from faster_whisper import WhisperModel
import os
import instaloader
import uuid
from dotenv import load_dotenv

load_dotenv()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

def get_reel_metadata(url: str) -> dict:
    """Extracts metadata from an Instagram reel using yt-dlp."""
    ydl_opts = { 'quiet': True, 'skip_download': True, 'dumpjson': True }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        raise ValueError(f"This reel is not publicly accessible or valid: {url}. Error: {e}")

def transcribe_audio(url: str) -> str:
    """Downloads audio of the reel and transcribes it using Whisper."""
    audio_path = f"temp_reel_audio_{uuid.uuid4().hex}.m4a"
    ydl_opts = { 
        'quiet': True, 
        'format': 'bestaudio/best',
        'outtmpl': audio_path
    }
    
    try:
        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        # Transcribe with faster-whisper (int8, CPU-optimised)
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path)
        transcript = " ".join([seg.text for seg in segments]).strip()

        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return transcript
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        print(f"Warning: Could not transcribe Instagram reel: {e}")
        return ""

def fetch_instagram_data(url: str) -> dict:
    """Fetches full data and transcript for an Instagram reel."""
    metadata = get_reel_metadata(url)
    transcript = transcribe_audio(url)
    
    # Try to fetch followers using Instaloader (might fail for unauthenticated requests)
    followers = 0
    uploader = metadata.get("uploader", "")
    if uploader:
        try:
            L = instaloader.Instaloader()
            profile = instaloader.Profile.from_username(L.context, uploader)
            followers = profile.followers
        except Exception as e:
            print(f"Warning: Could not fetch followers for {uploader}: {e}")
    
    return {
        "url": url,
        "video_id": metadata.get("id", ""),
        "transcript": transcript,
        "title": metadata.get("title", metadata.get("description", "")),
        "creator": uploader,
        "views": metadata.get("view_count", 0) or 0,
        "likes": metadata.get("like_count", 0) or 0,
        "comments": metadata.get("comment_count", 0) or 0,
        "followers": followers,
        "hashtags": metadata.get("tags", []) or [],
        "upload_date": metadata.get("upload_date", ""),
        "duration": metadata.get("duration", 0)
    }
