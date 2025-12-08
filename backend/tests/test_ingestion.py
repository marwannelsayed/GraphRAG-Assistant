"""
Unit tests for PDF ingestion functionality.
"""
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.services.embeddings import ingest_pdf_to_chroma


@pytest.fixture
def mock_openai_api_key():
    """Fixture to set OpenAI API key for testing."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-12345"}):
        yield


@pytest.fixture
def sample_pdf_content():
    """Create a minimal PDF content for testing."""
    # Minimal PDF structure (PDF version 1.4)
    # This is a very basic PDF that should parse but may not have much content
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000317 00000 n
0000000444 00000 n
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
527
%%EOF"""
    return pdf_content


@pytest.fixture
def sample_pdf_file(sample_pdf_content, tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_file = tmp_path / "test_sample.pdf"
    pdf_file.write_bytes(sample_pdf_content)
    return str(pdf_file)


class TestIngestPDFToChroma:
    """Test cases for ingest_pdf_to_chroma function."""
    
    def test_file_not_found(self, mock_openai_api_key):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            ingest_pdf_to_chroma("/nonexistent/file.pdf", "test_collection")
    
    def test_missing_openai_api_key(self, sample_pdf_file):
        """Test that ValueError is raised when OpenAI API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                ingest_pdf_to_chroma(sample_pdf_file, "test_collection")
    
    @patch('app.services.embeddings.OpenAIEmbeddings')
    @patch('app.services.embeddings.Chroma')
    @patch('app.services.embeddings.UnstructuredPDFLoader')
    @patch('app.services.embeddings.RecursiveCharacterTextSplitter')
    def test_successful_ingestion(
        self,
        mock_splitter_class,
        mock_loader_class,
        mock_chroma_class,
        mock_embeddings_class,
        sample_pdf_file,
        mock_openai_api_key
    ):
        """Test successful PDF ingestion returns non-zero chunks."""
        # Setup mocks
        mock_doc1 = Mock()
        mock_doc1.page_content = "This is a test document content. " * 100  # Enough for chunking
        mock_doc1.metadata = {"source": f"{sample_pdf_file}:page:0", "page": 0}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "This is more content. " * 100
        mock_doc2.metadata = {"source": f"{sample_pdf_file}:page:1", "page": 1}
        
        mock_documents = [mock_doc1, mock_doc2]
        
        # Mock loader
        mock_loader = Mock()
        mock_loader.load.return_value = mock_documents
        mock_loader_class.return_value = mock_loader
        
        # Mock splitter - return chunks that are smaller versions of documents
        mock_chunk1 = Mock()
        mock_chunk1.page_content = "This is a test document content. " * 50
        mock_chunk1.metadata = {"source": f"{sample_pdf_file}:page:0", "page": 0}
        
        mock_chunk2 = Mock()
        mock_chunk2.page_content = "This is more content. " * 50
        mock_chunk2.metadata = {"source": f"{sample_pdf_file}:page:1", "page": 1}
        
        mock_chunks = [mock_chunk1, mock_chunk2]
        
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = mock_chunks
        mock_splitter_class.return_value = mock_splitter
        
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        # Mock Chroma
        mock_vector_store = Mock()
        mock_chroma_class.return_value = mock_vector_store
        
        # Execute
        result = ingest_pdf_to_chroma(sample_pdf_file, "test_collection")
        
        # Assertions
        assert result["num_chunks"] > 0, "Ingestion should return non-zero chunks"
        assert result["num_chunks"] == 2, "Should have 2 chunks"
        assert result["collection_name"] == "test_collection"
        assert "doc_id" in result
        assert result["doc_id"] is not None
        
        # Verify mocks were called
        mock_loader_class.assert_called_once_with(sample_pdf_file)
        mock_loader.load.assert_called_once()
        mock_splitter_class.assert_called_once()
        mock_splitter.split_documents.assert_called_once()
        mock_embeddings_class.assert_called_once()
        mock_chroma_class.assert_called_once()
        mock_vector_store.add_documents.assert_called_once()
        mock_vector_store.persist.assert_called_once()
    
    @patch('app.services.embeddings.OpenAIEmbeddings')
    @patch('app.services.embeddings.Chroma')
    @patch('app.services.embeddings.UnstructuredPDFLoader')
    @patch('app.services.embeddings.RecursiveCharacterTextSplitter')
    def test_empty_document_handling(
        self,
        mock_splitter_class,
        mock_loader_class,
        mock_chroma_class,
        mock_embeddings_class,
        sample_pdf_file,
        mock_openai_api_key
    ):
        """Test handling of empty PDF documents."""
        # Mock loader to return empty list
        mock_loader = Mock()
        mock_loader.load.return_value = []
        mock_loader_class.return_value = mock_loader
        
        # Execute
        result = ingest_pdf_to_chroma(sample_pdf_file, "test_collection")
        
        # Assertions
        assert result["num_chunks"] == 0
        assert result["collection_name"] == "test_collection"
        
        # Chroma should not be called if no documents
        mock_chroma_class.assert_not_called()
    
    @patch('app.services.embeddings.OpenAIEmbeddings')
    @patch('app.services.embeddings.Chroma')
    @patch('app.services.embeddings.UnstructuredPDFLoader')
    @patch('app.services.embeddings.RecursiveCharacterTextSplitter')
    def test_chunk_metadata_preservation(
        self,
        mock_splitter_class,
        mock_loader_class,
        mock_chroma_class,
        mock_embeddings_class,
        sample_pdf_file,
        mock_openai_api_key
    ):
        """Test that chunk metadata is properly preserved and enhanced."""
        # Setup mock document with metadata
        mock_doc = Mock()
        mock_doc.page_content = "Test content. " * 100
        mock_doc.metadata = {"source": f"{sample_pdf_file}:page:0", "page": 0}
        
        mock_loader = Mock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader
        
        # Setup mock chunk
        mock_chunk = Mock()
        mock_chunk.page_content = "Test content. " * 50
        mock_chunk.metadata = {"source": f"{sample_pdf_file}:page:0", "page": 0}
        
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = [mock_chunk]
        mock_splitter_class.return_value = mock_splitter
        
        # Mock embeddings and Chroma
        mock_embeddings_class.return_value = Mock()
        mock_vector_store = Mock()
        mock_chroma_class.return_value = mock_vector_store
        
        # Capture the chunks passed to add_documents
        saved_chunks = []
        
        def capture_chunks(chunks):
            saved_chunks.extend(chunks)
            return ["mock_id"]
        
        mock_vector_store.add_documents.side_effect = capture_chunks
        
        # Execute
        result = ingest_pdf_to_chroma(sample_pdf_file, "test_collection")
        
        # Assertions
        assert len(saved_chunks) == 1
        chunk = saved_chunks[0]
        assert "doc_id" in chunk.metadata
        assert "source" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["chunk_index"] == 0

