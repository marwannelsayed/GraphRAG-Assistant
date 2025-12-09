"""
API endpoints for querying the knowledge base.
"""
import os
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from app.services.vector_store import get_retriever
from app.services.rag_chain import build_rag_chain

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = os.getenv("CHROMA_DEFAULT_COLLECTION", "documents")


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = True
    collection_name: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[dict]
    confidence: Optional[float] = None


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using HybridRAG approach.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY environment variable not set",
        )

    collection_name = request.collection_name or DEFAULT_COLLECTION
    top_k = request.top_k or 5

    try:
        # Build retriever
        retriever = get_retriever(collection_name=collection_name, k=top_k)

        # Initialize LLM
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )

        # Build RAG chain
        chain = build_rag_chain(llm, retriever)

        # Run chain
        result = chain({"question": request.question, "chat_history": []})

        answer = result.get("answer") or result.get("result") or ""
        source_docs = result.get("source_documents", []) or []

        sources_serialized = []
        for doc in source_docs:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            sources_serialized.append(
                {
                    "page_content": doc.page_content if hasattr(doc, "page_content") else "",
                    "metadata": {
                        "doc_id": meta.get("doc_id"),
                        "source": meta.get("source"),
                        "page": meta.get("page"),
                        "score": meta.get("score"),
                    },
                }
            )

        return QueryResponse(
            answer=answer,
            sources=sources_serialized if request.include_sources else [],
            confidence=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}",
        )
"""
API endpoints for querying the knowledge base.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = True


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[dict]
    confidence: Optional[float] = None


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using HybridRAG approach.
    
    TODO: Implement query logic:
    - Generate query embeddings
    - Retrieve relevant documents from vector store
    - Query Neo4j graph for entity relationships
    - Combine results using RAG chain
    - Generate answer with sources
    """
    raise HTTPException(status_code=501, detail="Not implemented yet")

