"""
API endpoints for retrieving source document details.
"""
import os
import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma

from app.services.embeddings import get_embeddings
from app.services.graph_service import GraphService

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = os.getenv("CHROMA_DEFAULT_COLLECTION", "documents")


class SourceResponse(BaseModel):
    """Response model for source retrieval."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    graph_entities: Optional[list] = None
    graph_relations: Optional[list] = None


@router.get("/{chunk_id}", response_model=SourceResponse)
async def get_source_chunk(
    chunk_id: str, 
    collection_name: Optional[str] = None,
    doc_id: Optional[str] = None,
    chunk_index: Optional[int] = None
):
    """
    Retrieve full text and metadata for a specific chunk.
    
    Args:
        chunk_id: The ChromaDB ID of the chunk, or special format "docid:index"
        collection_name: Optional collection name (defaults to environment variable)
        doc_id: Optional document ID (if chunk_id is not a ChromaDB ID)
        chunk_index: Optional chunk index (if chunk_id is not a ChromaDB ID)
        
    Returns:
        SourceResponse with full chunk text, metadata, and related graph entities
        
    Raises:
        HTTPException: If chunk not found or retrieval fails
    """
    collection_name = collection_name or DEFAULT_COLLECTION
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
    
    try:
        # Initialize vector store
        embeddings = get_embeddings()
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        
        # Get the underlying Chroma collection
        collection = vector_store._collection
        
        # Parse chunk_id if it's in format "docid:index"
        if ":" in chunk_id and not doc_id:
            parts = chunk_id.split(":", 1)
            if len(parts) == 2:
                doc_id = parts[0]
                try:
                    chunk_index = int(parts[1])
                except ValueError:
                    pass
        
        # Try to fetch by ID or by metadata
        try:
            result = None
            
            # If we have doc_id and chunk_index, search by metadata
            if doc_id and chunk_index is not None:
                logger.info(f"Searching by metadata: doc_id={doc_id}, chunk_index={chunk_index}")
                where_filter = {
                    "$and": [
                        {"doc_id": doc_id},
                        {"chunk_index": chunk_index}
                    ]
                }
                result = collection.get(where=where_filter, include=["documents", "metadatas"])
                if result and result.get("ids"):
                    chunk_id = result["ids"][0]
                    logger.info(f"Found chunk by metadata, ChromaDB ID: {chunk_id}")
            
            # If not found by metadata or no metadata provided, try direct ID lookup
            if not result or not result.get("documents"):
                logger.info(f"Searching by ChromaDB ID: {chunk_id}")
                result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            
            if not result or not result.get("documents") or len(result["documents"]) == 0:
                logger.warning(f"Chunk {chunk_id} not found in collection {collection_name}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chunk with ID '{chunk_id}' not found"
                )
            
            text = result["documents"][0]
            metadata = result["metadatas"][0] if result.get("metadatas") else {}
            
            # Add the actual ChromaDB ID to metadata
            if result.get("ids"):
                metadata["chroma_id"] = result["ids"][0]
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error fetching chunk from Chroma: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve chunk: {str(e)}"
            )
        
        # Try to get related graph entities and relations
        graph_entities = []
        graph_relations = []
        
        try:
            graph_service = GraphService()
            try:
                # Query entities linked to this chunk
                entities_query = """
                MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
                RETURN e.text as text, e.label as label, e.description as description
                LIMIT 20
                """
                entity_result = graph_service.query(entities_query, {"chunk_id": chunk_id})
                graph_entities = [dict(record) for record in entity_result]
                
                # Query relations involving entities from this chunk
                relations_query = """
                MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e1:Entity)
                MATCH (e1)-[r]->(e2:Entity)
                WHERE type(r) <> 'MENTIONS'
                RETURN e1.text as subject, type(r) as relation, e2.text as object
                LIMIT 20
                """
                relations_result = graph_service.query(relations_query, {"chunk_id": chunk_id})
                graph_relations = [dict(record) for record in relations_result]
                
            finally:
                graph_service.close()
                
        except Exception as e:
            logger.warning(f"Could not retrieve graph data for chunk {chunk_id}: {e}")
            # Continue without graph data - it's optional
        
        return SourceResponse(
            chunk_id=chunk_id,
            text=text,
            metadata=metadata,
            graph_entities=graph_entities,
            graph_relations=graph_relations,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Source retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve source: {str(e)}",
        )
