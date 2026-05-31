import chromadb
from langchain_community.vectorstores import Chroma
import os

CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "video_transcripts"

def get_vector_store(embedder):
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=CHROMA_PATH
    )

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

def retrieve_chunks(session_id: str, question: str, video_filter: str = None, k: int = 5) -> list:
    """
    Directly retrieves top-k chunks from ChromaDB for a question.
    No LangGraph involved. Plain similarity search.
    """
    from services.embedder import get_embedder # local import to avoid circular dependencies if any
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
