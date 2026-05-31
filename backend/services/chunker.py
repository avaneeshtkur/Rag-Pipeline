from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_transcript(transcript: str, video_id: str, metadata: dict) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    # Handle empty transcript nicely
    text_to_chunk = transcript if transcript.strip() else "No transcript available."
    
    chunks = splitter.create_documents(
        texts=[text_to_chunk],
        metadatas=[{
            "video_id": video_id,          # "A" or "B"
            "creator": metadata.get("creator", ""),
            "title": metadata.get("title", ""),
            "views": metadata.get("views", 0),
            "likes": metadata.get("likes", 0),
            "comments": metadata.get("comments", 0),
            "engagement_rate": metadata.get("engagement_rate", 0.0),
            "followers": metadata.get("followers", 0),
            "upload_date": metadata.get("upload_date", ""),
            "duration": metadata.get("duration", 0),
            "hashtags": ", ".join(metadata.get("hashtags", [])),
            "source_url": metadata.get("url", "")
        }]
    )
    return chunks
