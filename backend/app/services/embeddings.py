"""
Embedding service for generating vector embeddings and ingesting documents.
"""
import os
import logging
from typing import List
from uuid import uuid4

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


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
        ValueError: If OpenAI API key is not set
        Exception: For other ingestion errors
    """
    logger.info(f"Starting PDF ingestion: {pdf_path} -> collection: {collection_name}")
    
    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        error_msg = f"PDF file not found: {pdf_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_msg = "OPENAI_API_KEY environment variable not set"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Generate unique document ID
        doc_id = str(uuid4())
        logger.info(f"Assigned document ID: {doc_id}")
        
        # Load PDF using LangChain
        logger.info(f"Loading PDF: {pdf_path}")
        loader = UnstructuredPDFLoader(pdf_path)
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
            chunk_size=1000,
            chunk_overlap=200,
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
                        chunk.metadata["page"] = None
                else:
                    chunk.metadata["page"] = None
        
        # Initialize OpenAI embeddings
        logger.info("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small"  # Cost-effective model
        )
        
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
            "doc_id": doc_id
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-small"
    )
    
    return embeddings.embed_documents(texts)


def generate_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for a single query.
    
    Args:
        query: Query string to embed
        
    Returns:
        Embedding vector
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-small"
    )
    
    return embeddings.embed_query(query)
