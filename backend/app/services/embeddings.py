"""
Embedding service for generating vector embeddings.
"""
from typing import List


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    TODO: Implement embedding generation:
    - Choose embedding model (e.g., OpenAI, Sentence Transformers)
    - Generate embeddings for texts
    - Return list of embedding vectors
    """
    raise NotImplementedError("Embedding service not implemented yet")


def generate_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for a single query.
    
    TODO: Implement query embedding generation.
    """
    raise NotImplementedError("Query embedding not implemented yet")

