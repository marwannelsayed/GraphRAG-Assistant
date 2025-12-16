"""
Embedding service for generating vector embeddings and ingesting documents.
"""
import os
import logging
from typing import List
from uuid import uuid4

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


def verify_mps_support():
    """
    Verify MPS (Metal Performance Shaders) support for M1/M2 Macs.
    Returns tuple: (is_available, device_name, details)
    """
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        
        details = {
            'pytorch_version': torch.__version__,
            'mps_available': mps_available,
            'mps_built': mps_built,
        }
        
        if mps_available and mps_built:
            logger.info(f"✅ MPS GPU Support: ENABLED (PyTorch {torch.__version__})")
            logger.info(f"   - MPS Available: {mps_available}")
            logger.info(f"   - MPS Built: {mps_built}")
            return True, 'mps', details
        elif not mps_built:
            logger.warning(f"⚠️  MPS not built in PyTorch {torch.__version__}")
            return False, 'cpu', details
        else:
            logger.warning(f"⚠️  MPS not available on this system")
            return False, 'cpu', details
            
    except ImportError as e:
        logger.warning(f"⚠️  PyTorch not installed: {e}")
        return False, 'cpu', {'error': str(e)}
    except Exception as e:
        logger.error(f"❌ Error checking MPS support: {e}")
        return False, 'cpu', {'error': str(e)}


def get_embeddings():
    """
    Get embedding model based on configuration.
    Uses local sentence-transformers model with GPU acceleration on M1/M2 Macs.
    """
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Verify and use MPS (Metal Performance Shaders) for M1/M2 Macs
    mps_supported, device, details = verify_mps_support()
    
    if mps_supported:
        logger.info(f"🚀 Initializing embeddings with M1/M2 GPU acceleration")
        logger.info(f"   - Model: {embedding_model}")
        logger.info(f"   - Device: {device}")
    else:
        logger.info(f"📊 Initializing embeddings with CPU")
        logger.info(f"   - Model: {embedding_model}")
        logger.info(f"   - Device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Log successful initialization
    logger.info(f"✅ Embeddings initialized successfully on device: {device}")
    
    return embeddings


def ingest_pdf_to_chroma(pdf_path: str, collection_name: str) -> dict:
    """
    Ingest a PDF file into Chroma vector store.
    
    Args:
        pdf_path: Path to the PDF file
        collection_name: Name of the Chroma collection to store embeddings
        
    Returns:
        Dictionary with ingestion summary:
        - num_chunks: Number of text chunks created
        - collection_name: Name of the collection
        - doc_id: Unique document ID
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: For other ingestion errors
    """
    logger.info(f"Starting PDF ingestion: {pdf_path} -> collection: {collection_name}")
    
    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        error_msg = f"PDF file not found: {pdf_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Generate unique document ID
        doc_id = str(uuid4())
        logger.info(f"Assigned document ID: {doc_id}")
        
        # Load PDF using the more robust PyMuPDFLoader
        logger.info(f"Loading PDF with PyMuPDFLoader: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            logger.warning(f"No content extracted from PDF: {pdf_path}")
            return {
                "num_chunks": 0,
                "collection_name": collection_name,
                "doc_id": doc_id
            }
        
        logger.info(f"Loaded {len(documents)} page(s) from PDF")
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=250,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from PDF")
        
        if not chunks:
            logger.warning("No chunks created after splitting")
            return {
                "num_chunks": 0,
                "collection_name": collection_name,
                "doc_id": doc_id
            }
        
        # Add metadata to chunks
        source_name = os.path.basename(pdf_path)
        for i, chunk in enumerate(chunks):
            # Preserve existing metadata or initialize
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            
            # Add our metadata
            chunk.metadata.update({
                "doc_id": doc_id,
                "source": source_name,
                "chunk_index": i,
            })
            
            # Try to preserve page number if available
            if "page" not in chunk.metadata:
                # Extract page number from source if available (format: "source_path:page:X")
                source = chunk.metadata.get("source", "")
                if ":page:" in source:
                    try:
                        page_num = int(source.split(":page:")[-1])
                        chunk.metadata["page"] = page_num
                    except (ValueError, IndexError):
                        chunk.metadata["page"] = 0  # Use 0 instead of None for Neo4j compatibility
                else:
                    chunk.metadata["page"] = 0  # Use 0 instead of None for Neo4j compatibility
            elif chunk.metadata.get("page") is None:
                chunk.metadata["page"] = 0  # Ensure it's never None
        
        # Initialize local embeddings
        logger.info("Initializing local embeddings...")
        embeddings = get_embeddings()
        
        # Create or get Chroma collection and add documents
        logger.info(f"Storing embeddings in Chroma collection: {collection_name}")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_db"  # Persistent storage
        )
        
        # Add documents to vector store
        vector_store.add_documents(chunks)
        
        # Persist the collection
        vector_store.persist()
        logger.info(f"Successfully persisted {len(chunks)} chunks to Chroma")
        
        return {
            "num_chunks": len(chunks),
            "collection_name": collection_name,
            "doc_id": doc_id,
            "chunks": chunks  # Return chunks for further processing
        }
        
    except Exception as e:
        logger.error(f"Error ingesting PDF {pdf_path}: {str(e)}", exc_info=True)
        raise


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)


def generate_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for a single query.
    
    Args:
        query: Query string to embed
        
    Returns:
        Embedding vector
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(query)
