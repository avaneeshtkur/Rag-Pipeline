import chromadb
from langchain_community.vectorstores import Chroma
import os

CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "video_transcripts"

# ── Module-level cache: reuse the same Chroma store across all requests ──────
_store_cache: dict = {}  # keyed by id(embedder) so each embedder gets one store

def get_vector_store(embedder):
    cache_key = id(embedder)
    if cache_key not in _store_cache:
        _store_cache[cache_key] = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedder,
            persist_directory=CHROMA_PATH
        )
    return _store_cache[cache_key]

def ingest_chunks(chunks: list, embedder, session_id: str):
    """
    Each ingestion session gets a unique namespace using session_id
    so multiple users don't pollute each other's vector space.
    Filter by session_id in retrieval.
    """
    store = get_vector_store(embedder)
    # Add session_id to each chunk's metadata
    for chunk in chunks:
        chunk.metadata["session_id"] = session_id
    store.add_documents(chunks)
    return store

def retrieve_chunks(session_id: str, question: str, video_filter: str = None, k: int = 5, embedder=None) -> list:
    """
    Retrieves top-k chunks from ChromaDB for a question.
    Accepts a preloaded embedder to avoid re-initialising on every request.
    Falls back to get_embedder() when no embedder is supplied.
    """
    if embedder is None:
        from services.embedder import get_embedder  # local import to avoid circular deps
        embedder = get_embedder()
    store = get_vector_store(embedder)
    where_filter = {"session_id": session_id}
    if video_filter:
        where_filter["video_id"] = video_filter

    results = store.similarity_search(
        query=question,
        k=k,
        filter=where_filter
    )
    return results

def get_retriever(embedder, session_id: str, video_filter: str = None):
    """
    video_filter: "A", "B", or None (both)
    """
    store = get_vector_store(embedder)
    where_filter = {"session_id": session_id}
    if video_filter:
        where_filter["video_id"] = video_filter
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "filter": where_filter
        }
    )

