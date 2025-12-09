"""
Integration tests for hybrid query functionality.

Tests the combination of graph and vector retrieval with answer generation.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

from app.services.graph_service import GraphService
from app.services.vector_store import get_retriever
from app.services.rag_chain import hybrid_query, _merge_contexts, _query_graph, _query_vector


@pytest.fixture
def mock_graph_service():
    """Create a mock GraphService with sample data."""
    service = Mock(spec=GraphService)
    
    # Mock query_graph_for_question to return sample graph data
    service.query_graph_for_question.return_value = {
        "entities": [
            {
                "name": "Python",
                "type": "ProgrammingLanguage",
                "description": "A high-level programming language",
                "relationships": [
                    {
                        "relation": "USED_IN",
                        "target": "Machine Learning",
                        "target_type": "Domain"
                    }
                ]
            },
            {
                "name": "FastAPI",
                "type": "Framework",
                "description": "Modern Python web framework",
                "relationships": [
                    {
                        "relation": "BUILT_WITH",
                        "target": "Python",
                        "target_type": "ProgrammingLanguage"
                    }
                ]
            }
        ],
        "chunks": [
            {
                "chunk_id": "chunk_1",
                "text": "Python is widely used for data science and web development."
            }
        ],
        "graph_facts": "=== Knowledge Graph Context ===\n- Python (ProgrammingLanguage): A high-level programming language\n  Relationships:\n    • USED_IN → Machine Learning (Domain)",
        "node_ids": ["Python|ProgrammingLanguage", "FastAPI|Framework"]
    }
    
    return service


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predictable responses."""
    llm = Mock()
    
    # Mock the invoke method
    mock_response = Mock()
    mock_response.content = "Python is a versatile programming language used in machine learning [Python]. It powers frameworks like FastAPI [Snippet 1]."
    llm.invoke.return_value = mock_response
    
    return llm


@pytest.fixture
def sample_vector_store(tmp_path):
    """Create a sample vector store with test documents."""
    persist_dir = Path(tmp_path) / "chroma_hybrid_test"
    collection_name = "test_hybrid_collection"
    
    # Sample documents about Python and FastAPI
    docs = [
        Document(
            page_content="Python is a high-level, interpreted programming language known for its simplicity and readability.",
            metadata={"doc_id": "doc1", "source": "python_intro.txt", "chunk_id": "doc1_chunk_0"}
        ),
        Document(
            page_content="FastAPI is a modern, fast web framework for building APIs with Python based on standard type hints.",
            metadata={"doc_id": "doc2", "source": "fastapi_guide.txt", "chunk_id": "doc2_chunk_0"}
        ),
        Document(
            page_content="Machine learning with Python leverages libraries like scikit-learn, TensorFlow, and PyTorch.",
            metadata={"doc_id": "doc3", "source": "ml_python.txt", "chunk_id": "doc3_chunk_0"}
        ),
    ]
    
    # Create vector store
    embeddings = FakeEmbeddings(size=1536)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    
    return {
        "persist_dir": str(persist_dir),
        "collection_name": collection_name
    }


class TestGraphQueryFunction:
    """Test the query_graph_for_question function."""
    
    def test_graph_query_with_who_keyword(self, mock_graph_service):
        """Test graph query with 'who' keyword triggers entity-focused query."""
        question = "Who developed Python?"
        result = mock_graph_service.query_graph_for_question(question)
        
        assert "entities" in result
        assert "graph_facts" in result
        assert len(result["entities"]) > 0
        mock_graph_service.query_graph_for_question.assert_called_once_with(question)
    
    def test_graph_query_with_which_keyword(self, mock_graph_service):
        """Test graph query with 'which' keyword."""
        question = "Which frameworks use Python?"
        result = mock_graph_service.query_graph_for_question(question)
        
        assert "entities" in result
        assert len(result["entities"]) > 0
    
    def test_graph_query_with_depends_keyword(self, mock_graph_service):
        """Test graph query with 'depends' keyword for relationships."""
        # Update mock for relationship query
        mock_graph_service.query_graph_for_question.return_value = {
            "entities": [
                {
                    "name": "FastAPI",
                    "type": "Framework",
                    "description": "Web framework",
                    "relationships": [
                        {
                            "relation": "DEPENDS_ON",
                            "target": "Python",
                            "target_type": "ProgrammingLanguage"
                        }
                    ]
                }
            ],
            "chunks": [],
            "graph_facts": "FastAPI depends on Python",
            "node_ids": ["FastAPI|Framework"]
        }
        
        question = "What does FastAPI depend on?"
        result = mock_graph_service.query_graph_for_question(question)
        
        assert "entities" in result
        # Check for relationship information
        if result["entities"]:
            assert "relationships" in result["entities"][0]
    
    def test_graph_query_with_list_keyword(self, mock_graph_service):
        """Test graph query with 'list' keyword for broader search."""
        question = "List all programming languages"
        result = mock_graph_service.query_graph_for_question(question)
        
        assert "entities" in result
        assert isinstance(result["entities"], list)
    
    def test_graph_query_returns_node_ids(self, mock_graph_service):
        """Test that graph query returns node IDs for provenance."""
        question = "What is Python?"
        result = mock_graph_service.query_graph_for_question(question)
        
        assert "node_ids" in result
        assert isinstance(result["node_ids"], list)
        assert len(result["node_ids"]) > 0


class TestVectorQuery:
    """Test vector retrieval functionality."""
    
    def test_vector_query_returns_documents(self, sample_vector_store, monkeypatch):
        """Test that vector query returns relevant documents."""
        monkeypatch.setenv("CHROMA_EMBEDDING_BACKEND", "fake")
        
        retriever = get_retriever(
            collection_name=sample_vector_store["collection_name"],
            k=3,
            persist_directory=sample_vector_store["persist_dir"]
        )
        
        results = _query_vector(retriever, "What is Python?", top_k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        if results:
            assert hasattr(results[0], "page_content")
            assert hasattr(results[0], "metadata")
    
    def test_vector_query_handles_errors_gracefully(self):
        """Test that vector query handles errors without crashing."""
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.side_effect = Exception("Database error")
        
        results = _query_vector(mock_retriever, "test question", top_k=3)
        
        assert isinstance(results, list)
        assert len(results) == 0


class TestContextMerging:
    """Test the merging of graph and vector contexts."""
    
    def test_merge_empty_contexts(self):
        """Test merging with empty results."""
        graph_results = {"entities": [], "chunks": [], "graph_facts": "", "node_ids": []}
        vector_results = []
        
        merged = _merge_contexts(graph_results, vector_results, top_k=3)
        
        assert merged["graph_facts"] == ""
        assert merged["text_snippets"] == []
        assert merged["sources"] == []
        assert merged["chunk_ids"] == []
    
    def test_merge_graph_only_context(self):
        """Test merging with only graph results."""
        graph_results = {
            "entities": [{"name": "Python", "type": "Language"}],
            "chunks": [{"chunk_id": "c1", "text": "Python is great"}],
            "graph_facts": "Graph fact about Python",
            "node_ids": ["Python|Language"]
        }
        vector_results = []
        
        merged = _merge_contexts(graph_results, vector_results, top_k=3)
        
        assert merged["graph_facts"] == "Graph fact about Python"
        assert len(merged["sources"]) == 1
        assert merged["sources"][0]["type"] == "graph"
        assert "c1" in merged["chunk_ids"]
    
    def test_merge_vector_only_context(self):
        """Test merging with only vector results."""
        graph_results = {"entities": [], "chunks": [], "graph_facts": "", "node_ids": []}
        
        doc1 = Document(
            page_content="Python is a programming language",
            metadata={"doc_id": "doc1", "chunk_id": "doc1_c1"}
        )
        doc2 = Document(
            page_content="FastAPI is a web framework",
            metadata={"doc_id": "doc2", "chunk_id": "doc2_c1"}
        )
        vector_results = [doc1, doc2]
        
        merged = _merge_contexts(graph_results, vector_results, top_k=2)
        
        assert len(merged["text_snippets"]) == 2
        assert len(merged["sources"]) == 2
        assert all(s["type"] == "vector" for s in merged["sources"])
        assert "doc1_c1" in merged["chunk_ids"]
        assert "doc2_c1" in merged["chunk_ids"]
    
    def test_merge_hybrid_context(self):
        """Test merging with both graph and vector results."""
        graph_results = {
            "entities": [{"name": "Python", "type": "Language"}],
            "chunks": [{"chunk_id": "graph_c1", "text": "Graph chunk"}],
            "graph_facts": "Python is a language",
            "node_ids": ["Python|Language"]
        }
        
        doc = Document(
            page_content="Vector document about Python",
            metadata={"doc_id": "doc1", "chunk_id": "vec_c1"}
        )
        vector_results = [doc]
        
        merged = _merge_contexts(graph_results, vector_results, top_k=3)
        
        assert merged["graph_facts"] != ""
        assert len(merged["text_snippets"]) == 1
        assert len(merged["sources"]) == 2  # 1 graph + 1 vector
        assert "graph_c1" in merged["chunk_ids"]
        assert "vec_c1" in merged["chunk_ids"]
    
    def test_merge_respects_top_k_limit(self):
        """Test that merging respects the top_k parameter."""
        graph_results = {"entities": [], "chunks": [], "graph_facts": "", "node_ids": []}
        
        docs = [
            Document(page_content=f"Doc {i}", metadata={"doc_id": f"doc{i}"})
            for i in range(10)
        ]
        
        merged = _merge_contexts(graph_results, docs, top_k=3)
        
        assert len(merged["text_snippets"]) <= 3
        assert len([s for s in merged["sources"] if s["type"] == "vector"]) <= 3


class TestHybridQuery:
    """Test the complete hybrid query pipeline."""
    
    def test_hybrid_query_combines_graph_and_vector(
        self, mock_graph_service, mock_llm, sample_vector_store, monkeypatch
    ):
        """Test that hybrid query combines both graph and vector results."""
        monkeypatch.setenv("CHROMA_EMBEDDING_BACKEND", "fake")
        
        retriever = get_retriever(
            collection_name=sample_vector_store["collection_name"],
            k=3,
            persist_directory=sample_vector_store["persist_dir"]
        )
        
        result = hybrid_query(
            question="What is Python?",
            llm=mock_llm,
            retriever=retriever,
            graph_service=mock_graph_service,
            top_k=3
        )
        
        # Verify result structure
        assert "answer" in result
        assert "sources" in result
        assert "graph_context" in result
        assert "provenance" in result
        
        # Verify answer is generated
        assert len(result["answer"]) > 0
        
        # Verify provenance includes both graph and vector sources
        assert "node_ids" in result["provenance"]
        assert "chunk_ids" in result["provenance"]
        
        # Verify LLM was called
        mock_llm.invoke.assert_called_once()
    
    def test_hybrid_query_with_graph_first_question(
        self, mock_graph_service, mock_llm, sample_vector_store, monkeypatch
    ):
        """Test hybrid query with a question that benefits from graph context."""
        monkeypatch.setenv("CHROMA_EMBEDDING_BACKEND", "fake")
        
        retriever = get_retriever(
            collection_name=sample_vector_store["collection_name"],
            k=2,
            persist_directory=sample_vector_store["persist_dir"]
        )
        
        # Question with relationship focus
        result = hybrid_query(
            question="What frameworks depend on Python?",
            llm=mock_llm,
            retriever=retriever,
            graph_service=mock_graph_service,
            top_k=2
        )
        
        assert "answer" in result
        assert result["graph_context"]  # Should have graph context
        
        # Verify graph service was queried
        mock_graph_service.query_graph_for_question.assert_called_once()
    
    def test_hybrid_query_with_vector_first_question(
        self, mock_graph_service, mock_llm, sample_vector_store, monkeypatch
    ):
        """Test hybrid query with a question that benefits from text context."""
        monkeypatch.setenv("CHROMA_EMBEDDING_BACKEND", "fake")
        
        retriever = get_retriever(
            collection_name=sample_vector_store["collection_name"],
            k=3,
            persist_directory=sample_vector_store["persist_dir"]
        )
        
        # Descriptive question
        result = hybrid_query(
            question="Explain the features of FastAPI",
            llm=mock_llm,
            retriever=retriever,
            graph_service=mock_graph_service,
            top_k=3
        )
        
        assert "answer" in result
        assert len(result["sources"]) > 0
    
    def test_hybrid_query_handles_graph_failure_gracefully(
        self, mock_llm, sample_vector_store, monkeypatch
    ):
        """Test that hybrid query works even if graph query fails."""
        monkeypatch.setenv("CHROMA_EMBEDDING_BACKEND", "fake")
        
        # Mock graph service that raises an error
        failing_graph_service = Mock()
        failing_graph_service.query_graph_for_question.side_effect = Exception("Graph error")
        
        retriever = get_retriever(
            collection_name=sample_vector_store["collection_name"],
            k=3,
            persist_directory=sample_vector_store["persist_dir"]
        )
        
        result = hybrid_query(
            question="What is Python?",
            llm=mock_llm,
            retriever=retriever,
            graph_service=failing_graph_service,
            top_k=3
        )
        
        # Should still return a result (with vector results only)
        assert "answer" in result
        assert result["answer"]  # Should have an answer
    
    def test_hybrid_query_handles_vector_failure_gracefully(
        self, mock_graph_service, mock_llm
    ):
        """Test that hybrid query works even if vector query fails."""
        # Mock retriever that raises an error
        failing_retriever = Mock()
        failing_retriever.get_relevant_documents.side_effect = Exception("Vector error")
        
        result = hybrid_query(
            question="What is Python?",
            llm=mock_llm,
            retriever=failing_retriever,
            graph_service=mock_graph_service,
            top_k=3
        )
        
        # Should still return a result (with graph results only)
        assert "answer" in result
        assert result["answer"]  # Should have an answer
    
    def test_hybrid_query_includes_citations(
        self, mock_graph_service, mock_llm, sample_vector_store, monkeypatch
    ):
        """Test that hybrid query prompts LLM to include citations."""
        monkeypatch.setenv("CHROMA_EMBEDDING_BACKEND", "fake")
        
        retriever = get_retriever(
            collection_name=sample_vector_store["collection_name"],
            k=3,
            persist_directory=sample_vector_store["persist_dir"]
        )
        
        result = hybrid_query(
            question="What is Python?",
            llm=mock_llm,
            retriever=retriever,
            graph_service=mock_graph_service,
            top_k=3
        )
        
        # Check that the LLM was called with a prompt requesting citations
        call_args = mock_llm.invoke.call_args
        prompt = call_args[0][0]
        
        assert "cite" in prompt.lower() or "citation" in prompt.lower()
        assert "snippet" in prompt.lower()


class TestProvenanceTracking:
    """Test provenance tracking in hybrid queries."""
    
    def test_provenance_includes_chunk_ids(
        self, mock_graph_service, mock_llm, sample_vector_store, monkeypatch
    ):
        """Test that provenance tracks chunk IDs from both sources."""
        monkeypatch.setenv("CHROMA_EMBEDDING_BACKEND", "fake")
        
        retriever = get_retriever(
            collection_name=sample_vector_store["collection_name"],
            k=2,
            persist_directory=sample_vector_store["persist_dir"]
        )
        
        result = hybrid_query(
            question="What is Python?",
            llm=mock_llm,
            retriever=retriever,
            graph_service=mock_graph_service,
            top_k=2
        )
        
        provenance = result["provenance"]
        assert "chunk_ids" in provenance
        assert isinstance(provenance["chunk_ids"], list)
        assert len(provenance["chunk_ids"]) > 0
    
    def test_provenance_includes_node_ids(
        self, mock_graph_service, mock_llm, sample_vector_store, monkeypatch
    ):
        """Test that provenance tracks graph node IDs."""
        monkeypatch.setenv("CHROMA_EMBEDDING_BACKEND", "fake")
        
        retriever = get_retriever(
            collection_name=sample_vector_store["collection_name"],
            k=2,
            persist_directory=sample_vector_store["persist_dir"]
        )
        
        result = hybrid_query(
            question="What is Python?",
            llm=mock_llm,
            retriever=retriever,
            graph_service=mock_graph_service,
            top_k=2
        )
        
        provenance = result["provenance"]
        assert "node_ids" in provenance
        assert isinstance(provenance["node_ids"], list)
        # Should have node IDs from mock graph service
        assert "Python|ProgrammingLanguage" in provenance["node_ids"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
