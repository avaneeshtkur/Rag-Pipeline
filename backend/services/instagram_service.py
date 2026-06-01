import yt_dlp
from faster_whisper import WhisperModel
import os
import http.cookiejar
import instaloader
import uuid
from dotenv import load_dotenv

load_dotenv()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

def _get_cookie_path() -> str:
    """Returns the cookie path if it exists, else empty string."""
    path = os.getenv("INSTAGRAM_COOKIES_PATH", "")
    if path and os.path.exists(path):
        return path
    return ""

def get_reel_metadata(url: str) -> dict:
    """Extracts metadata from an Instagram reel using yt-dlp."""
    cookie_path = _get_cookie_path()
    ydl_opts = {'quiet': True, 'skip_download': True, 'dumpjson': True}
    if cookie_path:
        ydl_opts['cookiefile'] = cookie_path
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        raise ValueError(f"This reel is not publicly accessible or valid: {url}. Error: {e}")

def transcribe_audio(url: str) -> str:
    """Downloads audio of the reel and transcribes it using Whisper."""
    audio_path = f"temp_reel_audio_{uuid.uuid4().hex}.m4a"
    cookie_path = _get_cookie_path()
    ydl_opts = {
        'quiet': True,
        'format': 'bestaudio/best',
        'outtmpl': audio_path
    }
    if cookie_path:
        ydl_opts['cookiefile'] = cookie_path

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
    # Check cookie availability and warn early
    cookie_path = _get_cookie_path()
    cookies_loaded = bool(cookie_path)

    if not cookies_loaded:
        print(
            "Warning: No Instagram cookie file found at INSTAGRAM_COOKIES_PATH. "
            "Stats (views, likes, followers) will show as zero. "
            "To fix: export Instagram cookies using Cookie-Editor browser extension "
            "and set INSTAGRAM_COOKIES_PATH in your .env file."
        )

    metadata = get_reel_metadata(url)
    transcript = transcribe_audio(url)

    # Try to fetch followers using Instaloader, authenticated via cookie file
    followers = 0
    uploader = metadata.get("uploader", "")
    if uploader:
        try:
            L = instaloader.Instaloader()

            # Set a mobile user-agent to bypass some bot detection
            L.context._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15'
            })

            # Load session from exported cookie file if it exists
            if cookies_loaded:
                try:
                    cookie_jar = http.cookiejar.MozillaCookieJar()
                    cookie_jar.load(cookie_path, ignore_discard=True, ignore_expires=True)
                    for cookie in cookie_jar:
                        if 'instagram.com' in cookie.domain:
                            L.context._session.cookies.set(
                                cookie.name,
                                cookie.value,
                                domain=cookie.domain
                            )
                except Exception as e:
                    print(f"Warning: Could not load Instagram cookies into instaloader: {e}")

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
