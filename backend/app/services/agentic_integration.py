"""
Integration module for AgenticRetriever with existing GraphRAG backend.

This module bridges the agentic retriever with the current FastAPI backend,
providing adapters for ChromaDB and Neo4j clients to match the expected interface.
"""

import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase

from app.services.agentic_retriever import AgenticRetriever, UnifiedContext
from app.services.vector_store import VectorStore
from app.services.graph_service import GraphService

logger = logging.getLogger(__name__)


class ChromaDBAdapter:
    """
    Adapter to make VectorStore compatible with AgenticRetriever interface.
    
    The agentic retriever expects:
    - chroma_client.query(query_texts, n_results, where=filters)
    
    This adapter wraps VectorStore to provide that interface.
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query ChromaDB with metadata filters.
        
        Args:
            query_texts: List of query strings (typically just one)
            n_results: Number of results to return
            where: Metadata filters (e.g., {"document_id": "doc_123"})
            
        Returns:
            Dict with 'documents', 'metadatas', 'distances' keys
        """
        query = query_texts[0] if query_texts else ""
        
        logger.info(f"ChromaDB query: '{query[:80]}...' (top_k={n_results}, filters={where})")
        
        try:
            # Get collection
            collection = self.vector_store.collection
            
            # Query with filters
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.info(f"✅ Retrieved {len(results.get('documents', [[]])[0])} chunks from ChromaDB")
            return results
            
        except Exception as e:
            logger.error(f"❌ ChromaDB query failed: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}


class Neo4jAdapter:
    """
    Adapter to make GraphService compatible with AgenticRetriever interface.
    
    The agentic retriever expects:
    - neo4j_client.query_entities(entity_names)
    - neo4j_client.query_relations(entity_names, relation_types)
    - neo4j_client.query_relations_by_type(relation_types)
    """
    
    def __init__(self, graph_service: GraphService):
        self.graph_service = graph_service
    
    def query_entities(self, entity_names: List[str]) -> List[Dict[str, Any]]:
        """
        Query entities by name from Neo4j.
        
        Args:
            entity_names: List of entity names to retrieve
            
        Returns:
            List of entity dicts with 'name', 'type', 'description' keys
        """
        logger.info(f"Neo4j entity query: {entity_names}")
        
        if not entity_names:
            return []
        
        try:
            # Build Cypher query to find entities by name
            query = """
            MATCH (e:Entity)
            WHERE e.name IN $entity_names
            RETURN e.name AS name, 
                   e.type AS type, 
                   e.description AS description,
                   labels(e) AS labels
            LIMIT 50
            """
            
            with self.graph_service.driver.session() as session:
                result = session.run(query, entity_names=entity_names)
                entities = []
                for record in result:
                    entities.append({
                        'name': record['name'],
                        'type': record['type'],
                        'description': record.get('description', ''),
                        'labels': record.get('labels', [])
                    })
            
            logger.info(f"✅ Retrieved {len(entities)} entities from Neo4j")
            return entities
            
        except Exception as e:
            logger.error(f"❌ Neo4j entity query failed: {e}")
            return []
    
    def query_relations(
        self,
        entity_names: List[str],
        relation_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Query relationships between specified entities.
        
        Args:
            entity_names: List of entity names to find relations for
            relation_types: List of relation types to filter by
            
        Returns:
            List of relation dicts with 'subject', 'relation', 'object' keys
        """
        logger.info(f"Neo4j relation query: {entity_names} -> {relation_types}")
        
        if not entity_names:
            return []
        
        try:
            # Build Cypher query for relationships
            if relation_types:
                # Filter by specific relation types
                query = """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE e1.name IN $entity_names 
                  AND type(r) IN $relation_types
                RETURN e1.name AS subject, 
                       type(r) AS relation, 
                       e2.name AS object,
                       r.description AS description
                LIMIT 100
                """
                params = {
                    'entity_names': entity_names,
                    'relation_types': relation_types
                }
            else:
                # All relations for these entities
                query = """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE e1.name IN $entity_names
                RETURN e1.name AS subject, 
                       type(r) AS relation, 
                       e2.name AS object,
                       r.description AS description
                LIMIT 100
                """
                params = {'entity_names': entity_names}
            
            with self.graph_service.driver.session() as session:
                result = session.run(query, **params)
                relations = []
                for record in result:
                    relations.append({
                        'subject': record['subject'],
                        'relation': record['relation'],
                        'object': record['object'],
                        'description': record.get('description', '')
                    })
            
            logger.info(f"✅ Retrieved {len(relations)} relations from Neo4j")
            return relations
            
        except Exception as e:
            logger.error(f"❌ Neo4j relation query failed: {e}")
            return []
    
    def query_relations_by_type(self, relation_types: List[str]) -> List[Dict[str, Any]]:
        """
        Query all relationships of specific types (broader search).
        
        Args:
            relation_types: List of relation types to retrieve
            
        Returns:
            List of relation dicts with 'subject', 'relation', 'object' keys
        """
        logger.info(f"Neo4j relation type query: {relation_types}")
        
        if not relation_types:
            return []
        
        try:
            query = """
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE type(r) IN $relation_types
            RETURN e1.name AS subject, 
                   type(r) AS relation, 
                   e2.name AS object,
                   r.description AS description
            LIMIT 50
            """
            
            with self.graph_service.driver.session() as session:
                result = session.run(query, relation_types=relation_types)
                relations = []
                for record in result:
                    relations.append({
                        'subject': record['subject'],
                        'relation': record['relation'],
                        'object': record['object'],
                        'description': record.get('description', '')
                    })
            
            logger.info(f"✅ Retrieved {len(relations)} relations from Neo4j")
            return relations
            
        except Exception as e:
            logger.error(f"❌ Neo4j relation type query failed: {e}")
            return []


def create_agentic_retriever(
    vector_store: VectorStore,
    graph_service: GraphService,
    model: str = "phi3:mini",
    ollama_url: str = "http://ollama:11434"
) -> AgenticRetriever:
    """
    Factory function to create an AgenticRetriever with adapters.
    
    Args:
        vector_store: VectorStore instance
        graph_service: GraphService instance
        model: Ollama model name for planning agent
        ollama_url: Ollama server URL
        
    Returns:
        Configured AgenticRetriever
    """
    chroma_adapter = ChromaDBAdapter(vector_store)
    neo4j_adapter = Neo4jAdapter(graph_service)
    
    retriever = AgenticRetriever(
        chroma_client=chroma_adapter,
        neo4j_client=neo4j_adapter,
        model=model,
        ollama_url=ollama_url
    )
    
    logger.info(f"✅ Created AgenticRetriever with model={model}")
    return retriever


# Example integration with FastAPI endpoint
"""
# In backend/app/api/query.py

from app.services.agentic_integration import create_agentic_retriever

@router.post("/query/agentic")
async def agentic_query(
    request: QueryRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    graph_service: GraphService = Depends(get_graph_service)
):
    '''
    Agentic HybridRAG query endpoint.
    
    Uses phi3:mini to plan retrieval strategy, then executes and merges results.
    '''
    # Create agentic retriever
    retriever = create_agentic_retriever(vector_store, graph_service)
    
    # Execute agentic retrieval
    context = retriever.retrieve(
        question=request.question,
        context_docs=request.document_ids,  # Optional document scope
        top_k=request.top_k or 5
    )
    
    # Get formatted context for LLM
    context_string = context.to_context_string(max_chunks=5)
    
    # Generate answer using RAG chain
    answer = rag_chain.generate_answer(
        question=request.question,
        context=context_string
    )
    
    return {
        "answer": answer,
        "retrieval_plan": {
            "strategy": context.retrieval_plan.reason,
            "used_vector": context.retrieval_plan.use_vector_db,
            "used_graph": context.retrieval_plan.use_graph_db,
            "filters": context.retrieval_plan.vector_filters,
            "entities": context.retrieval_plan.graph_entities,
            "relations": context.retrieval_plan.graph_relation_types
        },
        "sources": {
            "vector_chunks": len(context.vector_chunks),
            "graph_entities": len(context.graph_entities),
            "graph_relations": len(context.graph_relations),
            "total_tokens": context.total_tokens
        }
    }
"""


if __name__ == "__main__":
    # Test the adapters
    logging.basicConfig(level=logging.INFO)
    
    print("✅ AgenticRetriever integration module ready!")
    print("\nTo use in your backend:")
    print("1. Import: from app.services.agentic_integration import create_agentic_retriever")
    print("2. Create: retriever = create_agentic_retriever(vector_store, graph_service)")
    print("3. Query: context = retriever.retrieve(question, context_docs=['doc_123'])")
    print("4. Format: context_string = context.to_context_string(max_chunks=5)")
    print("5. Generate: answer = rag_chain.generate_answer(question, context_string)")
