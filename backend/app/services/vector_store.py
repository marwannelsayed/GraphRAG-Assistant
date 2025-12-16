"""
Vector store utilities for retrieval using Chroma and LangChain retrievers.
"""
import os
import logging
from typing import Optional

from langchain_community.vectorstores import Chroma
from app.services.embeddings import get_embeddings

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "./chroma_db"


def get_retriever(collection_name: str, k: int = 4, persist_directory: Optional[str] = None):
    """
    Build a LangChain retriever backed by Chroma.

    Args:
        collection_name: Name of the Chroma collection to query.
        k: Number of results to return.
        persist_directory: Optional override for Chroma persistence path.

    Returns:
        A LangChain retriever object.
    """
    persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    
    # Use the same embedding function as ingestion
    embeddings = get_embeddings()

    logger.info(
        f"Initializing Chroma retriever for collection='{collection_name}', "
        f"persist_dir='{persist_dir}', k={k}"
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})


def get_vector_store(collection_name: str, persist_directory: Optional[str] = None) -> Chroma:
    """
    Get a direct Chroma vector store instance.

    Args:
        collection_name: Name of the Chroma collection.
        persist_directory: Optional override for Chroma persistence path.

    Returns:
        A Chroma vector store instance.
    """
    persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    embeddings = get_embeddings()
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


"""
Vector store service for storing and retrieving document embeddings.
"""
from typing import List, Dict, Optional


class VectorStore:
    """Vector store for document embeddings."""
    
    def __init__(self):
        """Initialize vector store."""
        # TODO: Initialize vector database (e.g., ChromaDB, Pinecone, Weaviate)
        pass
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadata: List[Dict]):
        """
        Add documents to the vector store.
        
        TODO: Implement document storage:
        - Store embeddings with metadata
        - Index documents for efficient retrieval
        """
        raise NotImplementedError("Document addition not implemented yet")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        TODO: Implement similarity search:
        - Compute similarity scores
        - Return top_k most similar documents with metadata
        """
        raise NotImplementedError("Vector search not implemented yet")

