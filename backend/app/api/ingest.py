"""
API endpoints for document ingestion.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

router = APIRouter()


@router.post("/documents")
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingest one or more documents.
    
    TODO: Implement document ingestion logic:
    - Validate file types
    - Extract text from documents
    - Generate embeddings
    - Store in vector database
    - Extract entities and relationships
    - Store in Neo4j graph database
    """
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """
    Get status of an ingestion job.
    
    TODO: Implement job status tracking.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet")

