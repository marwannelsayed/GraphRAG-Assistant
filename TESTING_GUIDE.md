# Testing Guide: Entity and Relation Extraction with Neo4j

## Test Results Summary

### ✅ Entity Extraction (Working!)
The spaCy entity extraction is **working perfectly**! 

Test results:
- ✅ Extracted **17 entities** from sample text
- ✅ Successfully identified:
  - Organizations: Apple Inc., Beats Electronics, Intel Corporation
  - People: Steve Jobs, Tim Cook
  - Locations: Cupertino, California, San Francisco, Palo Alto, United States, China, Japan
  - Dates: 1976
  - Numbers: over 50

### ⚠️ Relation Extraction (Needs OpenAI API Key)
- Status: Requires `OPENAI_API_KEY` environment variable
- To test: `export OPENAI_API_KEY='your-key-here'`

### ⚠️ Neo4j Storage (Needs Neo4j Running)
- Status: Docker not running / Neo4j not available
- To test: Start Neo4j using one of the methods below

## Quick Test

Run the test script:
```bash
cd backend
python3 test_extraction.py
```

## Setup Instructions

### 1. Set OpenAI API Key (for relation extraction)

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

To make it permanent, add to your `~/.zshrc`:
```bash
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Start Neo4j

#### Option A: Using Docker Compose (Recommended)
```bash
# Start Docker Desktop first, then:
cd infra
docker compose up -d neo4j
```

#### Option B: Using Docker Run
```bash
docker run -d \
  --name hybridrag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.14
```

#### Option C: Install Neo4j Desktop
Download from: https://neo4j.com/download/

### 3. Set Neo4j Environment Variables (if needed)

If using non-default settings:
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

### 4. Verify Neo4j is Running

Access Neo4j Browser at: http://localhost:7474
- Username: `neo4j`
- Password: `password` (or your custom password)

## Full End-to-End Test

### Test 1: Extraction Only (No Neo4j needed)

```bash
cd backend
python3 test_extraction.py
```

This will test:
- ✅ Entity extraction (spaCy) - **Working!**
- ⚠️ Relation extraction (OpenAI) - Needs API key
- ⚠️ Neo4j storage - Needs Neo4j running

### Test 2: Full Pipeline with PDF Upload

1. **Start the backend server:**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Upload a PDF via API:**
   ```bash
   curl -X POST "http://localhost:8000/api/ingest/" \
     -F "files=@path/to/your/document.pdf"
   ```

3. **Check the response** - should include:
   - `num_chunks`: Number of text chunks created
   - `num_entities`: Number of entities extracted
   - `num_relations`: Number of relations extracted
   - `graph_enabled`: Whether Neo4j storage succeeded

### Test 3: Query Neo4j Graph

Once data is ingested, you can query Neo4j:

**View all documents:**
```cypher
MATCH (d:Document) RETURN d.id, d.name, d.source
```

**View all entities:**
```cypher
MATCH (e:Entity) RETURN e.name, e.type LIMIT 20
```

**View relationships:**
```cypher
MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity) 
RETURN s.name AS subject, r.type AS relation, o.name AS object 
LIMIT 20
```

**Find entities in a document:**
```cypher
MATCH (d:Document {id: 'your-doc-id'})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN DISTINCT e.name, e.type
```

**Find related entities:**
```cypher
MATCH (e:Entity {name: 'Apple Inc.'})-[r:RELATES_TO]->(related:Entity)
RETURN related.name, r.type
```

## Troubleshooting

### Entity Extraction Not Working
- Make sure spaCy model is installed: `python -m spacy download en_core_web_sm`
- Check Python version (requires Python 3.8+)

### Relation Extraction Failing
- Verify `OPENAI_API_KEY` is set: `echo $OPENAI_API_KEY`
- Check API key is valid and has credits
- Check internet connection

### Neo4j Connection Failing
- Verify Neo4j is running: `docker ps | grep neo4j`
- Check port 7687 is accessible: `telnet localhost 7687`
- Verify credentials match environment variables
- Check Neo4j logs: `docker logs hybridrag-neo4j`

### Docker Issues
- Start Docker Desktop
- Check Docker is running: `docker ps`
- Try restarting Docker: `docker restart hybridrag-neo4j`

## Expected Test Output

When everything is configured correctly, you should see:

```
============================================================
HybridRAG Extraction and Graph Storage Test
============================================================

Testing Entity Extraction with spaCy
✓ Extracted 17 entities:
  1. Apple Inc. (ORG)
  2. Steve Jobs (PERSON)
  ...

Testing Relation Extraction with OpenAI
✓ Extracted 5 relations:
  1. Apple Inc. --[founded_by]--> Steve Jobs
  2. Tim Cook --[works_at]--> Apple Inc.
  ...

Testing Neo4j Connection
✓ Successfully connected to Neo4j

Testing Graph Storage in Neo4j
✓ Document created: test_doc_001
✓ Entity stored: Apple Inc.|ORG
✓ Relation stored: Apple Inc. --[founded_by]--> Steve Jobs

Test Summary
  Entity Extraction: ✓ PASS
  Relation Extraction: ✓ PASS
  Neo4j Connection: ✓ PASS
```

## Next Steps

1. ✅ Entity extraction is working - you can use this immediately!
2. Set up OpenAI API key to test relation extraction
3. Start Neo4j to test full graph storage pipeline
4. Upload a real PDF document to test the complete workflow

