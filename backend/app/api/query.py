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
from app.services.rag_chain import build_rag_chain, hybrid_query
from app.services.graph_service import GraphService

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = os.getenv("CHROMA_DEFAULT_COLLECTION", "documents")


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = True
    collection_name: Optional[str] = None
    use_hybrid: Optional[bool] = True  # Enable hybrid mode by default


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[dict]
    confidence: Optional[float] = None
    graph_context: Optional[str] = None
    provenance: Optional[dict] = None


@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using HybridRAG approach.
    
    This endpoint supports both vector-only and hybrid (vector + graph) retrieval modes.
    Hybrid mode is enabled by default and provides richer context by combining:
    - Vector similarity search for relevant text passages
    - Graph queries for entity relationships and structured knowledge
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

        # Determine query mode
        use_hybrid = request.use_hybrid if request.use_hybrid is not None else True
        
        if use_hybrid:
            # Use hybrid query mode (graph + vector)
            logger.info(f"Using hybrid query mode for: {request.question}")
            
            # Initialize graph service
            graph_service = GraphService()
            
            try:
                # Execute hybrid query
                result = hybrid_query(
                    question=request.question,
                    llm=llm,
                    retriever=retriever,
                    graph_service=graph_service,
                    top_k=top_k
                )
                
                # Format sources
                sources_serialized = []
                if request.include_sources:
                    for source in result.get("sources", []):
                        sources_serialized.append({
                            "type": source.get("type", "unknown"),
                            "content": source.get("content", "")[:500],  # Limit length
                            "metadata": source.get("metadata", {}),
                        })
                
                return QueryResponse(
                    answer=result["answer"],
                    sources=sources_serialized,
                    confidence=result.get("confidence"),
                    graph_context=result.get("graph_context"),
                    provenance=result.get("provenance") if request.include_sources else None,
                )
            finally:
                # Clean up graph service connection
                graph_service.close()
        
        else:
            # Use vector-only query mode (original behavior)
            logger.info(f"Using vector-only query mode for: {request.question}")
            
            # Build RAG chain
            chain = build_rag_chain(llm, retriever)

            # Run chain
            result = chain({"question": request.question, "chat_history": []})

            answer = result.get("answer") or result.get("result") or ""
            source_docs = result.get("source_documents", []) or []

            sources_serialized = []
            if request.include_sources:
                for doc in source_docs:
                    meta = doc.metadata if hasattr(doc, "metadata") else {}
                    sources_serialized.append({
                        "type": "vector",
                        "content": doc.page_content if hasattr(doc, "page_content") else "",
                        "metadata": {
                            "doc_id": meta.get("doc_id"),
                            "source": meta.get("source"),
                            "page": meta.get("page"),
                            "score": meta.get("score"),
                        },
                    })

            return QueryResponse(
                answer=answer,
                sources=sources_serialized,
                confidence=None,
                graph_context=None,
                provenance=None,
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

