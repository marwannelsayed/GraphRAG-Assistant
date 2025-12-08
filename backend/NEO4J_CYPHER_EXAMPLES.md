# Neo4j Cypher Query Examples

This document contains example Cypher statements used in the graph_service.py implementation.

## Index Creation

Indexes are created automatically on first connection for better query performance:

```cypher
// Index on document ID for fast document lookups
CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)

// Index on entity name for fast entity lookups
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)

// Index on entity type for filtering
CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)

// Index on chunk ID
CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.id)
```

## Document Operations

### Upsert Document

```cypher
MERGE (d:Document {id: $doc_id})
SET d.name = $name,
    d.source = $source,
    d.created_at = datetime()
RETURN d.id AS id
```

### Query All Documents

```cypher
MATCH (d:Document)
RETURN d.id AS id, d.name AS name, d.source AS source, d.created_at AS created_at
```

## Chunk Operations

### Upsert Chunk and Link to Document

```cypher
MERGE (c:Chunk {id: $chunk_id})
SET c.text = $text,
    c.chunk_index = $chunk_index,
    c.metadata = $metadata,
    c.created_at = datetime()
WITH c
MATCH (d:Document {id: $doc_id})
MERGE (d)-[:HAS_CHUNK]->(c)
RETURN c.id AS id
```

### Query Chunks for a Document

```cypher
MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.id AS id, c.text AS text, c.chunk_index AS chunk_index
ORDER BY c.chunk_index
```

## Entity Operations

### Upsert Entity

```cypher
MERGE (e:Entity {name: $name, type: $type})
SET e.description = $description,
    e.updated_at = datetime()
ON CREATE SET e.created_at = datetime()
RETURN e.name + '|' + e.type AS id
```

### Query Entities by Name

```cypher
MATCH (e:Entity)
WHERE e.name IN $entity_names
RETURN e.name AS name, e.type AS type, e.description AS description
```

### Query All Entities with Their Relationships

```cypher
MATCH (e:Entity)
OPTIONAL MATCH (e)-[r:RELATES_TO]->(related:Entity)
RETURN e.name AS name, e.type AS type, 
       collect(DISTINCT {relation: r.type, target: related.name, target_type: related.type}) AS relationships
```

## Relationship Operations

### Upsert Relation Between Entities

```cypher
MATCH (s:Entity {name: $subject_name, type: $subject_type})
MATCH (o:Entity {name: $object_name, type: $object_type})
MERGE (s)-[r:RELATES_TO {type: $relation_type}]->(o)
SET r.properties = $properties,
    r.updated_at = datetime()
ON CREATE SET r.created_at = datetime()
RETURN r
```

### Query Relationships for an Entity

```cypher
MATCH (e:Entity {name: $entity_name, type: $entity_type})-[r:RELATES_TO]->(related:Entity)
RETURN related.name AS target, r.type AS relation, related.type AS target_type
```

### Query All Relationships

```cypher
MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
RETURN s.name AS subject, r.type AS relation, o.name AS object
LIMIT 100
```

## Chunk-Entity Linking

### Link Chunk to Entity

```cypher
MATCH (c:Chunk {id: $chunk_id})
MATCH (e:Entity {name: $entity_name, type: $entity_type})
MERGE (c)-[:MENTIONS]->(e)
RETURN c, e
```

### Find Entities Mentioned in a Chunk

```cypher
MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
RETURN e.name AS name, e.type AS type
```

### Find Chunks Mentioning an Entity

```cypher
MATCH (e:Entity {name: $entity_name, type: $entity_type})<-[:MENTIONS]-(c:Chunk)
RETURN c.id AS chunk_id, c.text AS text
```

## Complex Queries

### Get Document Graph Structure

```cypher
MATCH (d:Document {id: $doc_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
OPTIONAL MATCH (e)-[r:RELATES_TO]->(related:Entity)
RETURN d, collect(DISTINCT c) AS chunks, 
       collect(DISTINCT e) AS entities,
       collect(DISTINCT r) AS relationships
```

### Find Related Entities (2-hop)

```cypher
MATCH (e:Entity {name: $entity_name})
MATCH (e)-[:RELATES_TO*1..2]->(related:Entity)
RETURN DISTINCT related.name AS name, related.type AS type
```

### Get Entity Co-occurrence (entities mentioned in same chunks)

```cypher
MATCH (e1:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
WHERE e1 <> e2
RETURN e1.name AS entity1, e2.name AS entity2, count(c) AS co_occurrence_count
ORDER BY co_occurrence_count DESC
LIMIT 20
```

### Find Documents by Entity

```cypher
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity {name: $entity_name})
RETURN DISTINCT d.id AS doc_id, d.name AS doc_name, count(c) AS mention_count
ORDER BY mention_count DESC
```

## Data Model

```
(Document)-[:HAS_CHUNK]->(Chunk)-[:MENTIONS]->(Entity)
                                               |
                                               v
                                         (Entity)-[:RELATES_TO]->(Entity)
```

### Node Labels
- **Document**: Represents an ingested document
  - Properties: `id`, `name`, `source`, `created_at`
- **Chunk**: Represents a text chunk from a document
  - Properties: `id`, `text`, `chunk_index`, `metadata`, `created_at`
- **Entity**: Represents a named entity or concept
  - Properties: `name`, `type`, `description`, `created_at`, `updated_at`

### Relationship Types
- **HAS_CHUNK**: Document contains Chunk
- **MENTIONS**: Chunk mentions Entity
- **RELATES_TO**: Entity relates to Entity
  - Properties: `type` (relationship type), `properties` (JSON), `created_at`, `updated_at`

