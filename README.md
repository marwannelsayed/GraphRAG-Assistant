# HybridRAG Knowledge Engine

A hybrid retrieval-augmented generation (RAG) system that combines vector embeddings with knowledge graph storage (Neo4j) for intelligent document ingestion and querying.

## Overview

HybridRAG Knowledge Engine uses a dual-storage approach:
- **Vector Store**: Stores document embeddings for semantic similarity search
- **Graph Database (Neo4j)**: Stores extracted entities and relationships for structured knowledge queries

This hybrid approach enables both semantic document retrieval and entity relationship exploration, providing richer context for generating answers.

## Project Structure

```
.
├── backend/          # FastAPI backend application
│   ├── app/
│   │   ├── api/      # API routes (ingest, query)
│   │   ├── services/ # Core services (embeddings, vector_store, graph, etc.)
│   │   └── main.py   # FastAPI application entry point
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/         # React frontend application
│   ├── src/
│   │   ├── components/
│   │   └── App.jsx
│   └── package.json
└── infra/            # Infrastructure configuration
    └── docker-compose.yml
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`
   - API docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

### Running with Docker Compose

1. Start Neo4j and backend services:
   ```bash
   cd infra
   docker-compose up -d
   ```

2. Access services:
   - Backend API: `http://localhost:8000`
   - Neo4j Browser: `http://localhost:7474` (username: `neo4j`, password: `password`)

3. Stop services:
   ```bash
   docker-compose down
   ```

## Features (To Be Implemented)

### Backend Services
- **Document Ingestion**: Upload and process documents
- **Embedding Generation**: Create vector embeddings for documents
- **Vector Store**: Store and retrieve documents by similarity
- **Entity Extraction**: Extract entities and relationships from text
- **Graph Storage**: Store knowledge graph in Neo4j
- **Hybrid RAG**: Combine vector and graph retrieval for answers

### Frontend Components
- **Upload**: Upload documents for ingestion
- **Chat**: Interactive Q&A interface
- **Source List**: View and manage ingested documents

## Development Status

This project is in the initial scaffolding phase. All services contain TODO placeholders for implementation. The current codebase provides:
- ✅ Project structure
- ✅ Basic FastAPI setup
- ✅ React frontend boilerplate
- ✅ Docker Compose configuration
- ⏳ Service implementations (TODO)
- ⏳ API endpoint logic (TODO)

## Next Steps

1. Implement embedding service (OpenAI or Sentence Transformers)
2. Set up vector database (ChromaDB, Pinecone, or Weaviate)
3. Implement Neo4j connection and graph operations
4. Add entity extraction (NER models)
5. Build RAG chain with LLM integration
6. Connect frontend to backend APIs
7. Add error handling and validation

## License

MIT

