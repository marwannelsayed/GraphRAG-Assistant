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
from app.services.extractor import extract_entities_sections, extract_relations_with_llm
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
                
                # Process each chunk for entity/relation extraction
                all_entities = []
                all_relations = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                    chunk_metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
                    
                    logger.debug(f"Processing chunk {i+1}/{num_chunks} for extraction")
                    
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
                    
                    # Extract entities using spaCy
                    try:
                        entities = extract_entities_sections(chunk_text)
                        logger.debug(f"Extracted {len(entities)} entities from chunk {i+1}")
                        
                        # Upsert entities and link to chunk
                        for entity in entities:
                            try:
                                entity_id = graph_service.upsert_entity(entity)
                                graph_service.link_chunk_to_entity(chunk_id, entity_id)
                                all_entities.append(entity)
                            except Exception as e:
                                logger.warning(f"Failed to upsert entity {entity.get('text')}: {e}")
                    except Exception as e:
                        logger.warning(f"Entity extraction failed for chunk {i+1}: {e}")
                    
                    # Extract relations using LLM
                    try:
                        relations = extract_relations_with_llm(chunk_text)
                        logger.debug(f"Extracted {len(relations)} relations from chunk {i+1}")
                        
                        # Upsert relations
                        for relation in relations:
                            try:
                                subject = relation["subject"]
                                relation_type = relation["relation"]
                                obj = relation["object"]
                                
                                # Find or create subject entity
                                subject_entities = [e for e in all_entities if e["text"] == subject]
                                if not subject_entities:
                                    # Create placeholder entity for subject
                                    subject_entity = {"text": subject, "label": "CONCEPT", "description": ""}
                                    try:
                                        subject_id = graph_service.upsert_entity(subject_entity)
                                        all_entities.append(subject_entity)
                                    except Exception as e:
                                        logger.warning(f"Failed to create subject entity {subject}: {e}")
                                        continue
                                else:
                                    subject_entity = subject_entities[0]
                                    subject_id = f"{subject_entity['text']}|{subject_entity['label']}"
                                
                                # Find or create object entity
                                object_entities = [e for e in all_entities if e["text"] == obj]
                                if not object_entities:
                                    # Create placeholder entity for object
                                    object_entity = {"text": obj, "label": "CONCEPT", "description": ""}
                                    try:
                                        object_id = graph_service.upsert_entity(object_entity)
                                        all_entities.append(object_entity)
                                    except Exception as e:
                                        logger.warning(f"Failed to create object entity {obj}: {e}")
                                        continue
                                else:
                                    object_entity = object_entities[0]
                                    object_id = f"{object_entity['text']}|{object_entity['label']}"
                                
                                # Upsert relation
                                if graph_service.upsert_relation(subject_id, relation_type, object_id):
                                    all_relations.append(relation)
                            except Exception as e:
                                logger.warning(f"Failed to upsert relation: {e}")
                    except Exception as e:
                        logger.warning(f"Relation extraction failed for chunk {i+1}: {e}")
                
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
