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

logger = logging.getLogger(__name__)

router = APIRouter()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint."""
    message: str
    doc_id: str
    num_chunks: int
    collection_name: str
    filename: str


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
        
        logger.info(
            f"Ingestion complete: {result['num_chunks']} chunks ingested "
            f"to collection '{result['collection_name']}'"
        )
        
        # Clean up uploaded file (optional - you might want to keep it)
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup file {file_path}: {cleanup_error}")
        
        return IngestionResponse(
            message=f"Successfully ingested {result['num_chunks']} chunks",
            doc_id=result['doc_id'],
            num_chunks=result['num_chunks'],
            collection_name=result['collection_name'],
            filename=file.filename
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
