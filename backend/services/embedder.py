from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embedder():
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
