"""
API endpoints for document ingestion.
"""
import os
import logging
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel

from app.services.embeddings import ingest_pdf_to_chroma
from app.services.extractor import extract_structured_information
from app.services.graph_service import GraphService

logger = logging.getLogger(__name__)

router = APIRouter()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize GraphService (will be None if Neo4j is not available)
_graph_service = None


def get_graph_service() -> GraphService:
    """Get or create GraphService instance."""
    global _graph_service
    if _graph_service is None:
        try:
            _graph_service = GraphService()
            logger.info("GraphService initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize GraphService: {e}. Graph features will be disabled.")
            _graph_service = None
    return _graph_service


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint."""
    message: str
    doc_id: str
    num_chunks: int
    collection_name: str
    filename: str
    num_entities: int = 0
    num_relations: int = 0
    graph_enabled: bool = False


@router.post("/", response_model=IngestionResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingest one or more PDF documents.
    
    Args:
        files: List of uploaded PDF files
        
    Returns:
        IngestionResponse with summary of ingestion
        
    Raises:
        HTTPException: If file validation fails or ingestion errors occur
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    # For now, process only the first file (can be extended to handle multiple)
    file = files[0]
    
    logger.info(f"Received file upload: {file.filename}, size: {file.size} bytes")
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Only PDF files are supported. Got: {file.filename}"
        )
    
    # Generate unique filename to avoid conflicts
    file_extension = Path(file.filename).suffix
    unique_filename = f"{Path(file.filename).stem}_{os.urandom(8).hex()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    try:
        # Save uploaded file
        logger.info(f"Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved successfully: {file_path} ({len(content)} bytes)")
        
        # Ingest PDF to Chroma
        # Use filename (without extension) as collection name, or a default
        collection_name = Path(file.filename).stem.lower().replace(" ", "_")
        if not collection_name:
            collection_name = "documents"
        
        logger.info(f"Ingesting PDF to Chroma collection: {collection_name}")
        result = ingest_pdf_to_chroma(str(file_path), collection_name)
        
        doc_id = result["doc_id"]
        chunks = result.get("chunks", [])
        num_chunks = result["num_chunks"]
        
        logger.info(
            f"Chroma ingestion complete: {num_chunks} chunks ingested "
            f"to collection '{result['collection_name']}'"
        )
        
        # Initialize graph service
        graph_service = get_graph_service()
        graph_enabled = graph_service is not None
        
        num_entities = 0
        num_relations = 0
        
        if graph_enabled and chunks:
            try:
                logger.info("Starting entity and relation extraction for graph storage")
                
                # Upsert document in Neo4j
                graph_service.upsert_document({
                    "id": doc_id,
                    "name": file.filename,
                    "source": str(file_path)
                })
                
                # OPTIMIZED: Extract entities and relationships from ALL chunks in batch
                # This uses the performance-limited extract_structured_information function
                chunk_texts = [
                    chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                    for chunk in chunks
                ]
                
                logger.info(f"Batch extracting entities and relationships from {len(chunk_texts)} chunks (limit: first 5 chunks for LLM relationships)")
                extraction_result = extract_structured_information(chunk_texts)
                
                all_entities = extraction_result["entities"]
                all_relations = extraction_result["relationships"]
                
                logger.info(f"Extracted {len(all_entities)} entities and {len(all_relations)} relationships in batch")
                
                # Now upsert chunks and link entities
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                    chunk_metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
                    
                    # Upsert chunk in Neo4j
                    try:
                        graph_service.upsert_chunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            chunk_index=i,
                            text=chunk_text[:1000],  # Store first 1000 chars
                            metadata=chunk_metadata
                        )
                    except Exception as e:
                        logger.warning(f"Failed to upsert chunk {chunk_id}: {e}")
                
                # Upsert all entities
                entity_id_map = {}  # Map entity text|label to Neo4j ID
                for entity in all_entities:
                    try:
                        entity_id = graph_service.upsert_entity(entity)
                        entity_key = f"{entity['text']}|{entity['label']}"
                        entity_id_map[entity_key] = entity_id
                    except Exception as e:
                        logger.warning(f"Failed to upsert entity {entity.get('text')}: {e}")
                
                # Upsert all relations
                for relation in all_relations:
                    try:
                        subject = relation["subject"]
                        relation_type = relation["relation"]
                        obj = relation["object"]
                        
                        # Try to find subject entity
                        subject_id = None
                        for entity_key, eid in entity_id_map.items():
                            if entity_key.startswith(subject + "|"):
                                subject_id = eid
                                break
                        
                        # Create placeholder if not found
                        if not subject_id:
                            subject_entity = {"text": subject, "label": "CONCEPT", "description": ""}
                            try:
                                subject_id = graph_service.upsert_entity(subject_entity)
                                entity_id_map[f"{subject}|CONCEPT"] = subject_id
                            except Exception as e:
                                logger.warning(f"Failed to create subject entity {subject}: {e}")
                                continue
                        
                        # Try to find object entity
                        object_id = None
                        for entity_key, eid in entity_id_map.items():
                            if entity_key.startswith(obj + "|"):
                                object_id = eid
                                break
                        
                        # Create placeholder if not found
                        if not object_id:
                            object_entity = {"text": obj, "label": "CONCEPT", "description": ""}
                            try:
                                object_id = graph_service.upsert_entity(object_entity)
                                entity_id_map[f"{obj}|CONCEPT"] = object_id
                            except Exception as e:
                                logger.warning(f"Failed to create object entity {obj}: {e}")
                                continue
                        
                        # Upsert relation
                        graph_service.upsert_relation(subject_id, relation_type, object_id)
                    except Exception as e:
                        logger.warning(f"Failed to upsert relation: {e}")
                
                num_entities = len(all_entities)
                num_relations = len(all_relations)
                
                logger.info(
                    f"Graph ingestion complete: {num_entities} entities and "
                    f"{num_relations} relations stored in Neo4j"
                )
                
            except Exception as e:
                logger.error(f"Error during graph ingestion: {e}", exc_info=True)
                # Don't fail the entire ingestion if graph fails
                graph_enabled = False
        
        # Clean up uploaded file (optional - you might want to keep it)
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup file {file_path}: {cleanup_error}")
        
        return IngestionResponse(
            message=f"Successfully ingested {num_chunks} chunks" + 
                   (f", {num_entities} entities, {num_relations} relations" if graph_enabled else ""),
            doc_id=doc_id,
            num_chunks=num_chunks,
            collection_name=result['collection_name'],
            filename=file.filename,
            num_entities=num_entities,
            num_relations=num_relations,
            graph_enabled=graph_enabled
        )
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        # Clean up if file was partially saved
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        # Clean up if file was saved
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}", exc_info=True)
        # Clean up if file was saved
        if file_path.exists():
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.get("/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """
    Get status of an ingestion job.
    
    TODO: Implement job status tracking.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet")


class UploadedDocument(BaseModel):
    """Model for uploaded document information."""
    filename: str
    collection_name: str
    file_size: int
    upload_date: str
    file_path: str


@router.get("/documents", response_model=List[UploadedDocument])
async def list_uploaded_documents():
    """
    List all uploaded documents.
    
    Returns a list of documents that have been uploaded to the system,
    including their filename, collection name, file size, and upload date.
    """
    try:
        documents = []
        
        # Check if uploads directory exists
        if not UPLOAD_DIR.exists():
            return documents
        
        # Iterate through all PDF files in uploads directory
        for file_path in UPLOAD_DIR.glob("*.pdf"):
            try:
                # Get file stats
                stats = file_path.stat()
                
                # Extract original filename (remove UUID suffix)
                filename = file_path.name
                # Try to extract original name before UUID
                if "_" in filename:
                    parts = filename.rsplit("_", 1)
                    if len(parts) == 2 and len(parts[1]) > 10:
                        original_filename = parts[0] + ".pdf"
                    else:
                        original_filename = filename
                else:
                    original_filename = filename
                
                # Generate collection name (same logic as ingestion)
                collection_name = original_filename.lower().replace(".pdf", "").replace(" ", "_").replace("-", "_")
                # Remove non-alphanumeric characters except underscores
                collection_name = "".join(c for c in collection_name if c.isalnum() or c == "_")
                
                # Get modification time
                from datetime import datetime
                upload_date = datetime.fromtimestamp(stats.st_mtime).isoformat()
                
                documents.append(UploadedDocument(
                    filename=original_filename,
                    collection_name=collection_name,
                    file_size=stats.st_size,
                    upload_date=upload_date,
                    file_path=str(file_path)
                ))
            except Exception as e:
                logger.warning(f"Failed to process file {file_path}: {e}")
                continue
        
        # Sort by upload date (most recent first)
        documents.sort(key=lambda x: x.upload_date, reverse=True)
        
        logger.info(f"Found {len(documents)} uploaded documents")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )
