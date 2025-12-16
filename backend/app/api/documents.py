"""
API endpoints for document management (list, delete, etc.).
"""
import os
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import chromadb

from app.services.graph_service import GraphService

router = APIRouter()
logger = logging.getLogger(__name__)


class DocumentInfo(BaseModel):
    """Document information model."""
    collection_name: str
    original_filename: str
    chunk_count: int
    created_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    documents: List[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    """Response model for delete operations."""
    success: bool
    message: str
    deleted_collections: List[str]


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents and their collections.
    
    Returns information about each document collection including:
    - Collection name
    - Original filename
    - Number of chunks
    - Creation timestamp (if available)
    """
    try:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
        client = chromadb.PersistentClient(path=chroma_dir)
        
        collections = client.list_collections()
        documents = []
        
        for collection in collections:
            try:
                # Get collection details
                coll = client.get_collection(name=collection.name)
                count = coll.count()
                
                # Try to get metadata from first chunk to extract original filename
                if count > 0:
                    results = coll.get(limit=1, include=["metadatas"])
                    metadata = results["metadatas"][0] if results["metadatas"] else {}
                    original_filename = metadata.get("source", collection.name)
                else:
                    original_filename = collection.name
                
                documents.append(DocumentInfo(
                    collection_name=collection.name,
                    original_filename=original_filename,
                    chunk_count=count,
                    created_at=None  # ChromaDB doesn't store collection creation time
                ))
            except Exception as e:
                logger.warning(f"Failed to get details for collection {collection.name}: {e}")
                continue
        
        # Sort by collection name
        documents.sort(key=lambda x: x.collection_name)
        
        return DocumentListResponse(
            documents=documents,
            total=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.delete("/{collection_name}", response_model=DeleteResponse)
async def delete_document(collection_name: str):
    """
    Delete a specific document by collection name.
    
    This will:
    1. Delete the ChromaDB collection (vector embeddings)
    2. Delete associated entities and relationships from Neo4j
    3. Keep the uploaded PDF file (for re-ingestion if needed)
    
    Args:
        collection_name: Name of the collection to delete
    """
    try:
        deleted_collections = []
        
        # Delete from ChromaDB
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
        client = chromadb.PersistentClient(path=chroma_dir)
        
        try:
            client.delete_collection(name=collection_name)
            deleted_collections.append(collection_name)
            logger.info(f"Deleted ChromaDB collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete ChromaDB collection {collection_name}: {e}")
        
        # Delete from Neo4j - Delete chunks, entities, and relationships associated with this collection
        graph_service = GraphService()
        try:
            # Delete chunks associated with documents in this collection
            # Note: This is a simplified approach - you might want to track doc_id <-> collection mapping
            query = """
            MATCH (d:Document)
            WHERE d.source = $collection_name OR d.id CONTAINS $collection_name
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            WITH d, c, e, count(e) as entity_refs
            // Delete chunks
            DETACH DELETE c
            // Delete document
            WITH d, collect(DISTINCT e) as entities
            DETACH DELETE d
            // Delete entities that are no longer referenced
            WITH entities
            UNWIND entities as entity
            OPTIONAL MATCH (entity)<-[:MENTIONS]-()
            WITH entity, count(*) as remaining_refs
            WHERE remaining_refs = 0
            DETACH DELETE entity
            """
            
            with graph_service.driver.session() as session:
                result = session.run(query, collection_name=collection_name)
                summary = result.consume()
                nodes_deleted = summary.counters.nodes_deleted
                rels_deleted = summary.counters.relationships_deleted
                logger.info(f"Deleted {nodes_deleted} nodes and {rels_deleted} relationships from Neo4j")
            
            graph_service.close()
        except Exception as e:
            logger.warning(f"Failed to delete from Neo4j for collection {collection_name}: {e}")
            graph_service.close()
        
        if deleted_collections:
            return DeleteResponse(
                success=True,
                message=f"Successfully deleted document: {collection_name}",
                deleted_collections=deleted_collections
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {collection_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.delete("/", response_model=DeleteResponse)
async def delete_all_documents():
    """
    Delete ALL documents from the system.
    
    This will:
    1. Delete all ChromaDB collections (all vector embeddings)
    2. Delete all entities, relationships, and chunks from Neo4j
    3. Keep uploaded PDF files (for re-ingestion if needed)
    
    ⚠️ WARNING: This is a destructive operation that cannot be undone!
    """
    try:
        deleted_collections = []
        
        # Delete all collections from ChromaDB
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
        client = chromadb.PersistentClient(path=chroma_dir)
        
        collections = client.list_collections()
        for collection in collections:
            try:
                client.delete_collection(name=collection.name)
                deleted_collections.append(collection.name)
                logger.info(f"Deleted ChromaDB collection: {collection.name}")
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection.name}: {e}")
        
        # Delete everything from Neo4j
        graph_service = GraphService()
        try:
            with graph_service.driver.session() as session:
                # Delete all relationships first
                result = session.run("MATCH ()-[r]->() DELETE r")
                summary = result.consume()
                rels_deleted = summary.counters.relationships_deleted
                
                # Delete all nodes
                result = session.run("MATCH (n) DELETE n")
                summary = result.consume()
                nodes_deleted = summary.counters.nodes_deleted
                
                logger.info(f"Deleted {nodes_deleted} nodes and {rels_deleted} relationships from Neo4j")
            
            graph_service.close()
        except Exception as e:
            logger.warning(f"Failed to delete from Neo4j: {e}")
            graph_service.close()
        
        return DeleteResponse(
            success=True,
            message=f"Successfully deleted all {len(deleted_collections)} documents",
            deleted_collections=deleted_collections
        )
        
    except Exception as e:
        logger.error(f"Failed to delete all documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete all documents: {str(e)}"
        )
