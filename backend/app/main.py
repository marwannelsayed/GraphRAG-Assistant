"""
Main FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ingest, query

app = FastAPI(title="HybridRAG Knowledge Engine API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/api/query", tags=["query"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "HybridRAG Knowledge Engine API", "status": "healthy"}

