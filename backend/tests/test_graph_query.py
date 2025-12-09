"""
Unit tests for graph service query functionality.

Tests the query_graph_for_question function and its helper methods.
"""
import pytest
from unittest.mock import Mock, MagicMock
from neo4j import GraphDatabase

from app.services.graph_service import GraphService


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = Mock()
    session = Mock()
    driver.session.return_value.__enter__ = Mock(return_value=session)
    driver.session.return_value.__exit__ = Mock(return_value=False)
    return driver, session


class TestKeywordExtraction:
    """Test keyword extraction from questions."""
    
    def test_extract_keywords_removes_stop_words(self):
        """Test that stop words are removed from questions."""
        graph_service = GraphService.__new__(GraphService)
        
        question = "what is the capital of france"
        keywords = graph_service._extract_keywords(question)
        
        # Should not include stop words
        assert "what" not in keywords
        assert "is" not in keywords
        assert "the" not in keywords
        assert "of" not in keywords
        
        # Should include meaningful words
        assert "capital" in keywords
        assert "france" in keywords
    
    def test_extract_keywords_handles_punctuation(self):
        """Test that punctuation is handled properly."""
        graph_service = GraphService.__new__(GraphService)
        
        question = "who developed python?"
        keywords = graph_service._extract_keywords(question)
        
        assert "python" in keywords
        assert "developed" in keywords
        assert "?" not in keywords
    
    def test_extract_keywords_empty_string(self):
        """Test keyword extraction with empty string."""
        graph_service = GraphService.__new__(GraphService)
        
        keywords = graph_service._extract_keywords("")
        
        assert isinstance(keywords, list)
        assert len(keywords) == 0


class TestGraphQueryStrategies:
    """Test different query strategies based on question types."""
    
    @pytest.fixture
    def graph_service_with_mock(self, mock_neo4j_driver):
        """Create GraphService with mocked driver."""
        driver, session = mock_neo4j_driver
        
        # Create instance without calling __init__
        service = GraphService.__new__(GraphService)
        service.driver = driver
        service.uri = "bolt://localhost:7687"
        service.user = "neo4j"
        service.password = "password"
        
        return service, session
    
    def test_query_entities_by_keywords_success(self, graph_service_with_mock):
        """Test querying entities by keywords."""
        service, session = graph_service_with_mock
        
        # Mock Neo4j result
        mock_record = Mock()
        mock_record.__getitem__ = lambda self, key: {
            "name": "Python",
            "type": "ProgrammingLanguage",
            "description": "A high-level language",
            "relationships": [
                {"relation": "USED_IN", "target": "ML", "target_type": "Domain"}
            ],
            "chunks": [
                {"chunk_id": "c1", "text": "Python is great"}
            ]
        }[key]
        
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        session.run.return_value = mock_result
        
        result = service._query_entities_by_keywords(["python"])
        
        assert "entities" in result
        assert "chunks" in result
        assert "node_ids" in result
        assert len(result["entities"]) > 0
        session.run.assert_called_once()
    
    def test_query_entities_by_keywords_empty_keywords(self, graph_service_with_mock):
        """Test entity query with empty keywords."""
        service, session = graph_service_with_mock
        
        result = service._query_entities_by_keywords([])
        
        assert result["entities"] == []
        assert result["chunks"] == []
        assert result["node_ids"] == []
        session.run.assert_not_called()
    
    def test_query_relationships_by_keywords(self, graph_service_with_mock):
        """Test querying relationships."""
        service, session = graph_service_with_mock
        
        # Mock relationship query result
        mock_record = Mock()
        mock_record.__getitem__ = lambda self, key: {
            "source_name": "FastAPI",
            "source_type": "Framework",
            "source_desc": "Web framework",
            "relation": "BUILT_WITH",
            "target_name": "Python",
            "target_type": "Language",
            "target_desc": "Programming language"
        }[key]
        
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        session.run.return_value = mock_result
        
        result = service._query_relationships_by_keywords(["fastapi", "python"])
        
        assert "entities" in result
        assert "node_ids" in result
        assert len(result["entities"]) > 0
    
    def test_query_entities_broad_with_keywords(self, graph_service_with_mock):
        """Test broad entity query with keywords."""
        service, session = graph_service_with_mock
        
        mock_record = Mock()
        mock_record.__getitem__ = lambda self, key: {
            "name": "Python",
            "type": "Language",
            "description": "Programming language",
            "relationships": []
        }[key]
        
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        session.run.return_value = mock_result
        
        result = service._query_entities_broad(["language"])
        
        assert "entities" in result
        assert len(result["entities"]) > 0
    
    def test_query_entities_broad_without_keywords(self, graph_service_with_mock):
        """Test broad entity query without keywords (recent entities)."""
        service, session = graph_service_with_mock
        
        mock_record = Mock()
        mock_record.__getitem__ = lambda self, key: {
            "name": "Entity1",
            "type": "Type1",
            "description": "Description",
            "relationships": []
        }[key]
        
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        session.run.return_value = mock_result
        
        result = service._query_entities_broad([])
        
        assert "entities" in result
        # Should query for recent entities
        session.run.assert_called_once()
    
    def test_query_connected_entities(self, graph_service_with_mock):
        """Test querying connected entities (2-hop neighborhood)."""
        service, session = graph_service_with_mock
        
        mock_record = Mock()
        mock_record.__getitem__ = lambda self, key: {
            "name": "Python",
            "type": "Language",
            "description": "Programming language",
            "connected_entities": [
                {"name": "FastAPI", "type": "Framework"},
                {"name": "Django", "type": "Framework"}
            ]
        }[key]
        
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        session.run.return_value = mock_result
        
        result = service._query_connected_entities(["python"])
        
        assert "entities" in result
        assert len(result["entities"]) > 0
        if result["entities"]:
            assert "connected_entities" in result["entities"][0]


class TestGraphQueryForQuestion:
    """Test the main query_graph_for_question function."""
    
    @pytest.fixture
    def graph_service_with_mocks(self, mock_neo4j_driver):
        """Create GraphService with all methods mocked."""
        driver, session = mock_neo4j_driver
        
        service = GraphService.__new__(GraphService)
        service.driver = driver
        
        # Mock all helper methods
        service._extract_keywords = Mock(return_value=["python", "programming"])
        service._query_entities_by_keywords = Mock(return_value={
            "entities": [{"name": "Python", "type": "Language"}],
            "chunks": [],
            "node_ids": ["Python|Language"]
        })
        service._query_relationships_by_keywords = Mock(return_value={
            "entities": [],
            "chunks": [],
            "node_ids": []
        })
        service._query_entities_broad = Mock(return_value={
            "entities": [],
            "chunks": [],
            "node_ids": []
        })
        service._query_connected_entities = Mock(return_value={
            "entities": [],
            "chunks": [],
            "node_ids": []
        })
        service._format_graph_facts = Mock(return_value="Formatted facts")
        
        return service
    
    def test_who_question_triggers_entity_query(self, graph_service_with_mocks):
        """Test that 'who' questions trigger entity-focused queries."""
        service = graph_service_with_mocks
        
        result = service.query_graph_for_question("Who created Python?")
        
        service._query_entities_by_keywords.assert_called_once()
        assert "entities" in result
        assert "graph_facts" in result
    
    def test_which_question_triggers_entity_query(self, graph_service_with_mocks):
        """Test that 'which' questions trigger entity-focused queries."""
        service = graph_service_with_mocks
        
        result = service.query_graph_for_question("Which language is Python?")
        
        service._query_entities_by_keywords.assert_called_once()
    
    def test_what_question_triggers_entity_query(self, graph_service_with_mocks):
        """Test that 'what' questions trigger entity-focused queries."""
        service = graph_service_with_mocks
        
        result = service.query_graph_for_question("What is Python?")
        
        service._query_entities_by_keywords.assert_called_once()
    
    def test_depends_question_triggers_relationship_query(self, graph_service_with_mocks):
        """Test that 'depends' questions trigger relationship queries."""
        service = graph_service_with_mocks
        
        result = service.query_graph_for_question("FastAPI depends on Python")
        
        service._query_relationships_by_keywords.assert_called_once()
    
    def test_list_question_triggers_broad_query(self, graph_service_with_mocks):
        """Test that 'list' questions trigger broad queries."""
        service = graph_service_with_mocks
        
        result = service.query_graph_for_question("List all frameworks")
        
        service._query_entities_broad.assert_called_once()
    
    def test_related_question_triggers_connection_query(self, graph_service_with_mocks):
        """Test that 'related' questions trigger connection queries."""
        service = graph_service_with_mocks
        
        result = service.query_graph_for_question("Python related frameworks")
        
        service._query_connected_entities.assert_called_once()
    
    def test_default_query_strategy(self, graph_service_with_mocks):
        """Test default query strategy for questions without special keywords."""
        service = graph_service_with_mocks
        
        result = service.query_graph_for_question("Python programming language")
        
        # Should default to entity query
        service._query_entities_by_keywords.assert_called_once()
    
    def test_result_formatting(self, graph_service_with_mocks):
        """Test that results are properly formatted."""
        service = graph_service_with_mocks
        
        result = service.query_graph_for_question("What is Python?")
        
        assert "entities" in result
        assert "chunks" in result
        assert "graph_facts" in result
        assert "node_ids" in result
        service._format_graph_facts.assert_called_once()


class TestGraphFactsFormatting:
    """Test formatting of graph facts for LLM consumption."""
    
    def test_format_empty_results(self):
        """Test formatting with empty results."""
        service = GraphService.__new__(GraphService)
        
        result = service._format_graph_facts({"entities": []})
        
        assert result == ""
    
    def test_format_entity_without_relationships(self):
        """Test formatting entity without relationships."""
        service = GraphService.__new__(GraphService)
        
        results = {
            "entities": [
                {
                    "name": "Python",
                    "type": "Language",
                    "description": "A programming language",
                    "relationships": []
                }
            ]
        }
        
        formatted = service._format_graph_facts(results)
        
        assert "Python" in formatted
        assert "Language" in formatted
        assert "A programming language" in formatted
    
    def test_format_entity_with_relationships(self):
        """Test formatting entity with relationships."""
        service = GraphService.__new__(GraphService)
        
        results = {
            "entities": [
                {
                    "name": "FastAPI",
                    "type": "Framework",
                    "description": "Web framework",
                    "relationships": [
                        {
                            "relation": "BUILT_WITH",
                            "target": "Python",
                            "target_type": "Language"
                        }
                    ]
                }
            ]
        }
        
        formatted = service._format_graph_facts(results)
        
        assert "FastAPI" in formatted
        assert "Framework" in formatted
        assert "BUILT_WITH" in formatted
        assert "Python" in formatted
    
    def test_format_entity_with_connected_entities(self):
        """Test formatting entity with connected entities."""
        service = GraphService.__new__(GraphService)
        
        results = {
            "entities": [
                {
                    "name": "Python",
                    "type": "Language",
                    "description": "Programming language",
                    "relationships": [],
                    "connected_entities": [
                        {"name": "FastAPI", "type": "Framework"},
                        {"name": "Django", "type": "Framework"}
                    ]
                }
            ]
        }
        
        formatted = service._format_graph_facts(results)
        
        assert "Python" in formatted
        assert "Connected to:" in formatted
        assert "FastAPI" in formatted
        assert "Django" in formatted
    
    def test_format_multiple_entities(self):
        """Test formatting multiple entities."""
        service = GraphService.__new__(GraphService)
        
        results = {
            "entities": [
                {
                    "name": "Python",
                    "type": "Language",
                    "description": "Programming language",
                    "relationships": []
                },
                {
                    "name": "FastAPI",
                    "type": "Framework",
                    "description": "Web framework",
                    "relationships": []
                }
            ]
        }
        
        formatted = service._format_graph_facts(results)
        
        assert "Python" in formatted
        assert "FastAPI" in formatted
        assert "Knowledge Graph Context" in formatted


class TestErrorHandling:
    """Test error handling in graph queries."""
    
    def test_query_handles_neo4j_errors(self, mock_neo4j_driver):
        """Test that queries handle Neo4j errors gracefully."""
        driver, session = mock_neo4j_driver
        session.run.side_effect = Exception("Neo4j connection error")
        
        service = GraphService.__new__(GraphService)
        service.driver = driver
        
        result = service._query_entities_by_keywords(["python"])
        
        # Should return empty results, not raise exception
        assert result["entities"] == []
        assert result["chunks"] == []
        assert result["node_ids"] == []
    
    def test_query_handles_malformed_results(self, mock_neo4j_driver):
        """Test handling of malformed query results."""
        driver, session = mock_neo4j_driver
        
        # Mock result with missing fields
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=KeyError("Missing field"))
        
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        session.run.return_value = mock_result
        
        service = GraphService.__new__(GraphService)
        service.driver = driver
        
        # Should handle the error gracefully
        result = service._query_entities_by_keywords(["python"])
        
        # May return empty or partial results depending on error handling
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
