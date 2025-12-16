"""
FastAPI endpoint example for Agentic HybridRAG query.

This shows how to integrate the AgenticRetriever with your existing FastAPI backend.
The endpoint uses phi3:mini to plan retrieval strategies before executing queries.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from app.services.vector_store import VectorStore, get_vector_store
from app.services.graph_service import GraphService, get_graph_service
from app.services.rag_chain import RAGChain, get_rag_chain
from app.services.agentic_integration import create_agentic_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agentic_query"])


class AgenticQueryRequest(BaseModel):
    """Request model for agentic query"""
    question: str = Field(..., description="User's question")
    document_ids: Optional[List[str]] = Field(
        None,
        description="Optional document IDs to scope retrieval (enforces document-aware filtering)"
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve per database")
    max_chunks: int = Field(5, ge=1, le=10, description="Maximum chunks to include in context")
    use_agent: bool = Field(
        True,
        description="Whether to use agent for planning (false = fallback to vector-only)"
    )


class RetrievalPlanResponse(BaseModel):
    """Response model for retrieval plan details"""
    strategy: str
    used_vector: bool
    used_graph: bool
    vector_filters: Dict[str, Any]
    graph_entities: List[str]
    graph_relation_types: List[str]


class AgenticQueryResponse(BaseModel):
    """Response model for agentic query"""
    answer: str
    retrieval_plan: RetrievalPlanResponse
    sources: Dict[str, Any]
    metadata: Dict[str, Any]


@router.post("/query/agentic", response_model=AgenticQueryResponse)
async def agentic_query(
    request: AgenticQueryRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    graph_service: GraphService = Depends(get_graph_service),
    rag_chain: RAGChain = Depends(get_rag_chain)
):
    """
    Execute an agentic HybridRAG query.
    
    This endpoint uses phi3:mini as a planning agent to:
    1. Analyze the user's question
    2. Generate a retrieval strategy (which databases, what filters)
    3. Execute queries with document-aware scoping
    4. Merge results into unified context
    5. Generate answer using RAG chain
    
    The agent ensures:
    - Retrieval stays within specified document scope
    - Appropriate databases are queried based on question type
    - Efficient use of resources (only query what's needed)
    - Safety checks prevent query explosion or injection
    
    Args:
        request: Query request with question and optional document scope
        vector_store: ChromaDB vector store instance
        graph_service: Neo4j graph service instance
        rag_chain: RAG chain for answer generation
    
    Returns:
        Answer with retrieval plan details and source information
    
    Example:
        POST /api/query/agentic
        {
            "question": "How does GPT-4V compare to Claude on vision tasks?",
            "document_ids": ["doc_123"],
            "top_k": 5
        }
    """
    try:
        logger.info(f"🤖 Agentic query: {request.question[:100]}...")
        
        # Create agentic retriever
        retriever = create_agentic_retriever(
            vector_store=vector_store,
            graph_service=graph_service,
            model="phi3:mini",
            ollama_url="http://ollama:11434"
        )
        
        # Execute agentic retrieval
        context = retriever.retrieve(
            question=request.question,
            context_docs=request.document_ids,
            top_k=request.top_k
        )
        
        logger.info(
            f"✅ Retrieved: {len(context.vector_chunks)} chunks, "
            f"{len(context.graph_entities)} entities, "
            f"{len(context.graph_relations)} relations"
        )
        
        # Format context for LLM
        context_string = context.to_context_string(max_chunks=request.max_chunks)
        
        # Generate answer using RAG chain
        answer = await rag_chain.generate_answer(
            question=request.question,
            context=context_string
        )
        
        logger.info(f"✅ Generated answer ({len(answer)} chars)")
        
        # Build response
        return AgenticQueryResponse(
            answer=answer,
            retrieval_plan=RetrievalPlanResponse(
                strategy=context.retrieval_plan.reason,
                used_vector=context.retrieval_plan.use_vector_db,
                used_graph=context.retrieval_plan.use_graph_db,
                vector_filters=context.retrieval_plan.vector_filters,
                graph_entities=context.retrieval_plan.graph_entities,
                graph_relation_types=context.retrieval_plan.graph_relation_types
            ),
            sources={
                "vector_chunks": len(context.vector_chunks),
                "graph_entities": len(context.graph_entities),
                "graph_relations": len(context.graph_relations),
                "total_tokens": context.total_tokens
            },
            metadata={
                "question_length": len(request.question),
                "context_length": len(context_string),
                "document_scope": request.document_ids or "all_documents",
                "top_k": request.top_k
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Agentic query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agentic query failed: {str(e)}"
        )


@router.post("/query/agentic/plan-only")
async def plan_retrieval_only(
    request: AgenticQueryRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    graph_service: GraphService = Depends(get_graph_service)
):
    """
    Generate a retrieval plan WITHOUT executing queries.
    
    Useful for debugging, testing, or showing users the retrieval strategy
    before actual execution.
    
    Args:
        request: Query request with question and optional document scope
        vector_store: ChromaDB vector store instance
        graph_service: Neo4j graph service instance
    
    Returns:
        Retrieval plan details (no answer generated)
    """
    try:
        logger.info(f"📋 Planning retrieval for: {request.question[:100]}...")
        
        # Create agentic retriever
        retriever = create_agentic_retriever(
            vector_store=vector_store,
            graph_service=graph_service,
            model="phi3:mini",
            ollama_url="http://ollama:11434"
        )
        
        # Generate plan (no execution)
        plan = retriever.plan_retrieval(
            question=request.question,
            context_docs=request.document_ids
        )
        
        logger.info(f"✅ Generated plan: {plan.reason}")
        
        return {
            "plan": {
                "strategy": plan.reason,
                "use_vector_db": plan.use_vector_db,
                "use_graph_db": plan.use_graph_db,
                "vector_filters": plan.vector_filters,
                "graph_entities": plan.graph_entities,
                "graph_relation_types": plan.graph_relation_types
            },
            "validation": {
                "is_valid": plan.validate()[0],
                "error": plan.validate()[1]
            },
            "metadata": {
                "question": request.question,
                "document_scope": request.document_ids or "all_documents"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Plan generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Plan generation failed: {str(e)}"
        )


@router.get("/query/agentic/stats")
async def get_agentic_stats():
    """
    Get statistics about agentic retrieval usage.
    
    Returns:
        Statistics about agent performance and retrieval patterns
    """
    # TODO: Implement statistics tracking
    # This would track:
    # - Average planning time
    # - Most common retrieval strategies
    # - Success/failure rates
    # - Resource usage (vector vs graph queries)
    
    return {
        "message": "Statistics tracking not yet implemented",
        "todo": [
            "Track agent planning time",
            "Log retrieval strategy distribution",
            "Monitor validation failure rates",
            "Measure context size distribution"
        ]
    }


# Integration instructions for main.py
"""
To enable agentic queries in your FastAPI app:

1. Add to backend/app/main.py:

```python
from app.api.agentic_query import router as agentic_router

app.include_router(agentic_router)
```

2. Test with curl:

```bash
# Standard agentic query
curl -X POST http://localhost:8000/api/query/agentic \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does GPT-4V compare to Claude?",
    "top_k": 5
  }'

# Document-scoped query
curl -X POST http://localhost:8000/api/query/agentic \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the key findings",
    "document_ids": ["doc_123"],
    "top_k": 3
  }'

# Plan-only (debug)
curl -X POST http://localhost:8000/api/query/agentic/plan-only \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What datasets are used?",
    "document_ids": ["doc_123"]
  }'
```

3. Frontend integration (frontend/src/components/Chat.jsx):

```javascript
const agenticQuery = async (question, documentIds = null) => {
  const response = await fetch('http://localhost:8000/api/query/agentic', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      document_ids: documentIds,
      top_k: 5
    })
  });
  
  const data = await response.json();
  
  // Show retrieval strategy to user
  console.log('Retrieval strategy:', data.retrieval_plan.strategy);
  console.log('Used vector DB:', data.retrieval_plan.used_vector);
  console.log('Used graph DB:', data.retrieval_plan.used_graph);
  
  return data.answer;
};
```
"""


if __name__ == "__main__":
    print("✅ Agentic query endpoint ready!")
    print("\nEndpoints:")
    print("  POST /api/query/agentic - Execute agentic query")
    print("  POST /api/query/agentic/plan-only - Generate plan without execution")
    print("  GET  /api/query/agentic/stats - Get statistics (not implemented)")
    print("\nTo integrate:")
    print("  1. Add router to main.py: app.include_router(agentic_router)")
    print("  2. Test with curl (see integration instructions above)")
    print("  3. Update frontend to use /api/query/agentic endpoint")
