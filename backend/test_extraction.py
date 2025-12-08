#!/usr/bin/env python3
"""
Test script for entity and relation extraction with Neo4j ingestion.
"""
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test text sample
TEST_TEXT = """
Apple Inc. is a technology company founded by Steve Jobs in Cupertino, California in 1976.
Tim Cook is the current CEO of Apple Inc. and works at the company headquarters.
Steve Jobs was born in San Francisco and later moved to Palo Alto.
Apple Inc. owns subsidiaries like Beats Electronics and has partnerships with Intel Corporation.
The company operates in over 50 countries worldwide including the United States, China, and Japan.
"""


def test_entity_extraction():
    """Test spaCy entity extraction."""
    logger.info("=" * 60)
    logger.info("Testing Entity Extraction with spaCy")
    logger.info("=" * 60)
    
    try:
        from app.services.extractor import extract_entities_sections
        
        logger.info(f"Extracting entities from test text...")
        entities = extract_entities_sections(TEST_TEXT)
        
        logger.info(f"\n✓ Extracted {len(entities)} entities:\n")
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. {entity['text']} ({entity['label']}) - {entity['description']}")
        
        return entities
    except Exception as e:
        logger.error(f"✗ Entity extraction failed: {e}", exc_info=True)
        return None


def test_relation_extraction():
    """Test OpenAI relation extraction."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Relation Extraction with OpenAI")
    logger.info("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("⚠ OPENAI_API_KEY not set. Skipping relation extraction test.")
        logger.info("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return None
    
    try:
        from app.services.extractor import extract_relations_with_llm
        
        logger.info(f"Extracting relations from test text...")
        logger.info("   (This may take a few seconds...)\n")
        relations = extract_relations_with_llm(TEST_TEXT)
        
        logger.info(f"✓ Extracted {len(relations)} relations:\n")
        for i, rel in enumerate(relations, 1):
            print(f"  {i}. {rel['subject']} --[{rel['relation']}]--> {rel['object']}")
        
        return relations
    except Exception as e:
        logger.error(f"✗ Relation extraction failed: {e}", exc_info=True)
        return None


def test_neo4j_connection():
    """Test Neo4j connection."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Neo4j Connection")
    logger.info("=" * 60)
    
    try:
        from app.services.graph_service import GraphService
        
        logger.info("Connecting to Neo4j...")
        graph_service = GraphService()
        
        if graph_service.connect_neo4j():
            logger.info("✓ Successfully connected to Neo4j")
            return graph_service
        else:
            logger.error("✗ Failed to connect to Neo4j")
            return None
    except Exception as e:
        logger.error(f"✗ Neo4j connection failed: {e}")
        logger.info("\n   Make sure Neo4j is running:")
        logger.info("   - Using Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        logger.info("   - Or set: export NEO4J_URI='bolt://localhost:7687'")
        logger.info("   - And: export NEO4J_PASSWORD='your-password'")
        return None


def test_graph_storage(graph_service, entities, relations):
    """Test storing entities and relations in Neo4j."""
    if not graph_service:
        logger.warning("\n⚠ Skipping graph storage test (Neo4j not available)")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing Graph Storage in Neo4j")
    logger.info("=" * 60)
    
    try:
        # Test document upsert
        logger.info("\n1. Creating test document...")
        doc_id = graph_service.upsert_document({
            "id": "test_doc_001",
            "name": "Test Document",
            "source": "test_extraction.py"
        })
        logger.info(f"   ✓ Document created: {doc_id}")
        
        # Test entity upsert
        if entities:
            logger.info(f"\n2. Storing {len(entities)} entities...")
            stored_entity_ids = []
            for entity in entities[:5]:  # Limit to first 5 for testing
                entity_id = graph_service.upsert_entity(entity)
                stored_entity_ids.append(entity_id)
                logger.info(f"   ✓ Entity stored: {entity_id}")
            
            # Test relation upsert
            if relations and stored_entity_ids:
                logger.info(f"\n3. Storing {len(relations)} relations...")
                stored_count = 0
                for rel in relations[:5]:  # Limit to first 5 for testing
                    # Try to find matching entities
                    subject_entities = [e for e in entities if e['text'] == rel['subject']]
                    object_entities = [e for e in entities if e['text'] == rel['object']]
                    
                    if subject_entities and object_entities:
                        subject_id = f"{subject_entities[0]['text']}|{subject_entities[0]['label']}"
                        object_id = f"{object_entities[0]['text']}|{object_entities[0]['label']}"
                        
                        if graph_service.upsert_relation(subject_id, rel['relation'], object_id):
                            stored_count += 1
                            logger.info(f"   ✓ Relation stored: {rel['subject']} --[{rel['relation']}]--> {rel['object']}")
                
                logger.info(f"\n   ✓ Stored {stored_count} relations")
        
        logger.info("\n✓ Graph storage test completed successfully!")
        logger.info("\n   You can query Neo4j with:")
        logger.info("   MATCH (d:Document {id: 'test_doc_001'}) RETURN d")
        logger.info("   MATCH (e:Entity) RETURN e.name, e.type LIMIT 10")
        logger.info("   MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity) RETURN s.name, r.type, o.name LIMIT 10")
        
    except Exception as e:
        logger.error(f"\n✗ Graph storage failed: {e}", exc_info=True)


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("HybridRAG Extraction and Graph Storage Test")
    logger.info("=" * 60 + "\n")
    
    # Test entity extraction
    entities = test_entity_extraction()
    
    # Test relation extraction
    relations = test_relation_extraction()
    
    # Test Neo4j connection
    graph_service = test_neo4j_connection()
    
    # Test graph storage
    if graph_service and (entities or relations):
        test_graph_storage(graph_service, entities or [], relations or [])
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"  Entity Extraction: {'✓ PASS' if entities else '✗ FAIL'}")
    logger.info(f"  Relation Extraction: {'✓ PASS' if relations else '✗ FAIL'}")
    logger.info(f"  Neo4j Connection: {'✓ PASS' if graph_service else '✗ FAIL'}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()

