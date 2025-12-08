"""
Neo4j graph service for storing and querying knowledge graphs.
"""
import os
import logging
from typing import List, Dict, Optional
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GraphService:
    """Service for interacting with Neo4j graph database."""
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j URI (defaults to NEO4J_URI env var or bolt://localhost:7687)
            user: Neo4j username (defaults to NEO4J_USER env var or "neo4j")
            password: Neo4j password (defaults to NEO4J_PASSWORD env var or "password")
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        logger.info(f"Initializing Neo4j connection to {self.uri} as user {self.user}")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
            
            # Create indexes on first connection
            self._create_indexes()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for better query performance."""
        indexes = [
            # Index on document ID for fast document lookups
            "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)",
            # Index on entity name for fast entity lookups
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            # Index on entity type for filtering
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            # Index on chunk ID
            "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
        ]
        
        with self.driver.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.debug(f"Created index: {index_query}")
                except Exception as e:
                    logger.warning(f"Index creation may have failed (might already exist): {e}")
    
    def connect_neo4j(self) -> bool:
        """
        Verify Neo4j connection.
        
        Returns:
            True if connection is successful
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            logger.info("Neo4j connection verified")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def upsert_document(self, document_meta: Dict) -> str:
        """
        Upsert a document node in Neo4j.
        
        Cypher query:
        MERGE (d:Document {id: $doc_id})
        SET d.name = $name,
            d.source = $source,
            d.created_at = datetime()
        RETURN d.id AS id
        
        Args:
            document_meta: Dictionary with keys: id, name, source (optional)
            
        Returns:
            Document ID
        """
        doc_id = document_meta["id"]
        name = document_meta.get("name", "")
        source = document_meta.get("source", "")
        
        query = """
        MERGE (d:Document {id: $doc_id})
        SET d.name = $name,
            d.source = $source,
            d.created_at = datetime()
        RETURN d.id AS id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    doc_id=doc_id,
                    name=name,
                    source=source
                )
                record = result.single()
                doc_id_returned = record["id"] if record else doc_id
                logger.debug(f"Upserted document: {doc_id_returned}")
                return doc_id_returned
        except Exception as e:
            logger.error(f"Error upserting document {doc_id}: {e}")
            raise
    
    def upsert_chunk(self, chunk_id: str, doc_id: str, chunk_index: int, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Upsert a chunk node and link it to its document.
        
        Cypher query:
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.chunk_index = $chunk_index,
            c.metadata = $metadata,
            c.created_at = datetime()
        WITH c
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN c.id AS id
        
        Args:
            chunk_id: Unique chunk identifier
            doc_id: Document ID this chunk belongs to
            chunk_index: Index of chunk in document
            text: Chunk text content
            metadata: Optional metadata dictionary
            
        Returns:
            Chunk ID
        """
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.chunk_index = $chunk_index,
            c.metadata = $metadata,
            c.created_at = datetime()
        WITH c
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN c.id AS id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=text[:10000],  # Limit text length
                    metadata=metadata or {}
                )
                record = result.single()
                chunk_id_returned = record["id"] if record else chunk_id
                logger.debug(f"Upserted chunk: {chunk_id_returned}")
                return chunk_id_returned
        except Exception as e:
            logger.error(f"Error upserting chunk {chunk_id}: {e}")
            raise
    
    def upsert_entity(self, entity: Dict) -> str:
        """
        Upsert an entity node in Neo4j.
        
        Cypher query:
        MERGE (e:Entity {name: $name, type: $type})
        SET e.description = $description,
            e.updated_at = datetime()
        ON CREATE SET e.created_at = datetime()
        RETURN e.name + '|' + e.type AS id
        
        Args:
            entity: Dictionary with keys: text (name), label (type), description (optional)
            
        Returns:
            Entity identifier (name|type)
        """
        name = entity["text"]
        entity_type = entity["label"]
        description = entity.get("description", "")
        
        # Create unique identifier
        entity_id = f"{name}|{entity_type}"
        
        query = """
        MERGE (e:Entity {name: $name, type: $type})
        ON CREATE SET e.created_at = datetime(), 
                      e.description = $description,
                      e.updated_at = datetime()
        ON MATCH SET e.description = $description,
                     e.updated_at = datetime()
        RETURN e.name + '|' + e.type AS id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    name=name,
                    type=entity_type,
                    description=description
                )
                record = result.single()
                entity_id_returned = record["id"] if record else entity_id
                logger.debug(f"Upserted entity: {entity_id_returned}")
                return entity_id_returned
        except Exception as e:
            logger.error(f"Error upserting entity {name} ({entity_type}): {e}")
            raise
    
    def upsert_relation(self, subject_id: str, relation: str, object_id: str, properties: Optional[Dict] = None) -> bool:
        """
        Upsert a relationship between two entities.
        
        Cypher query:
        MATCH (s:Entity {name: $subject_name, type: $subject_type})
        MATCH (o:Entity {name: $object_name, type: $object_type})
        MERGE (s)-[r:RELATES_TO {type: $relation_type}]->(o)
        SET r.properties = $properties,
            r.updated_at = datetime()
        ON CREATE SET r.created_at = datetime()
        RETURN r
        
        Args:
            subject_id: Entity identifier (format: "name|type")
            relation: Relationship type
            object_id: Entity identifier (format: "name|type")
            properties: Optional relationship properties
            
        Returns:
            True if relationship was created/updated
        """
        # Parse entity IDs (format: "name|type")
        try:
            subject_name, subject_type = subject_id.split("|", 1)
            object_name, object_type = object_id.split("|", 1)
        except ValueError:
            logger.error(f"Invalid entity ID format. Expected 'name|type', got: {subject_id} or {object_id}")
            return False
        
        query = """
        MATCH (s:Entity {name: $subject_name, type: $subject_type})
        MATCH (o:Entity {name: $object_name, type: $object_type})
        MERGE (s)-[r:RELATES_TO {type: $relation_type}]->(o)
        SET r.properties = $properties,
            r.updated_at = datetime()
        ON CREATE SET r.created_at = datetime()
        RETURN r
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    subject_name=subject_name,
                    subject_type=subject_type,
                    object_name=object_name,
                    object_type=object_type,
                    relation_type=relation,
                    properties=properties or {}
                )
                record = result.single()
                if record:
                    logger.debug(f"Upserted relation: {subject_id} -[{relation}]-> {object_id}")
                    return True
                else:
                    logger.warning(f"Failed to create relation: {subject_id} -[{relation}]-> {object_id} (entities not found)")
                    return False
        except Exception as e:
            logger.error(f"Error upserting relation {subject_id} -[{relation}]-> {object_id}: {e}")
            return False
    
    def link_chunk_to_entity(self, chunk_id: str, entity_id: str) -> bool:
        """
        Link a chunk to an entity (chunk mentions entity).
        
        Cypher query:
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {name: $entity_name, type: $entity_type})
        MERGE (c)-[:MENTIONS]->(e)
        RETURN c, e
        
        Args:
            chunk_id: Chunk identifier
            entity_id: Entity identifier (format: "name|type")
            
        Returns:
            True if link was created
        """
        try:
            entity_name, entity_type = entity_id.split("|", 1)
        except ValueError:
            logger.error(f"Invalid entity ID format: {entity_id}")
            return False
        
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {name: $entity_name, type: $entity_type})
        MERGE (c)-[:MENTIONS]->(e)
        RETURN c, e
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    chunk_id=chunk_id,
                    entity_name=entity_name,
                    entity_type=entity_type
                )
                record = result.single()
                if record:
                    logger.debug(f"Linked chunk {chunk_id} to entity {entity_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error linking chunk {chunk_id} to entity {entity_id}: {e}")
            return False
    
    def query_entities(self, entity_names: List[str]) -> List[Dict]:
        """
        Query entities and their relationships.
        
        Cypher query:
        MATCH (e:Entity)
        WHERE e.name IN $entity_names
        OPTIONAL MATCH (e)-[r:RELATES_TO]->(related:Entity)
        RETURN e, collect(r) AS relationships, collect(related) AS related_entities
        
        Args:
            entity_names: List of entity names to query
            
        Returns:
            List of dictionaries with entity and relationship information
        """
        query = """
        MATCH (e:Entity)
        WHERE e.name IN $entity_names
        OPTIONAL MATCH (e)-[r:RELATES_TO]->(related:Entity)
        RETURN e.name AS name, e.type AS type, e.description AS description,
               collect(DISTINCT {relation: r.type, target: related.name, target_type: related.type}) AS relationships
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, entity_names=entity_names)
                entities = []
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "description": record["description"],
                        "relationships": [r for r in record["relationships"] if r["relation"]]
                    })
                return entities
        except Exception as e:
            logger.error(f"Error querying entities: {e}")
            return []
