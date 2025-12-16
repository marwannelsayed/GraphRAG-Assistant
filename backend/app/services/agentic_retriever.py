"""
Agentic HybridRAG Orchestrator with Document-Aware Retrieval

This module implements an intelligent retrieval planning system where an LLM agent
(phi3:mini) analyzes each user query and generates a JSON retrieval plan specifying:
- Which databases to query (vector, graph, or both)
- What metadata filters to apply (document IDs, types)
- Which entities and relation types to target in the graph
- Safety constraints to prevent out-of-scope retrieval

The orchestrator validates the plan, executes queries with proper scoping,
and merges results into a unified context object for the RAG chain.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum

import ollama

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types for retrieval"""
    VECTOR = "vector"
    GRAPH = "graph"
    BOTH = "both"


@dataclass
class RetrievalPlan:
    """Structured retrieval plan from the agent"""
    use_vector_db: bool
    use_graph_db: bool
    vector_filters: Dict[str, Any]  # ChromaDB metadata filters
    graph_entities: List[str]  # Entity names to query
    graph_relation_types: List[str]  # Relation types to traverse
    reason: str  # Agent's reasoning for this plan
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate the retrieval plan for safety and correctness.
        
        Returns:
            (is_valid, error_message)
        """
        # Must use at least one database
        if not self.use_vector_db and not self.use_graph_db:
            return False, "Plan must specify at least one database (vector or graph)"
        
        # If using graph, must specify entities or relations
        if self.use_graph_db and not self.graph_entities and not self.graph_relation_types:
            return False, "Graph query requires entities or relation types"
        
        # Vector filters must be a dict
        if self.vector_filters and not isinstance(self.vector_filters, dict):
            return False, "vector_filters must be a dictionary"
        
        # Validate filter keys are safe
        allowed_filter_keys = {"document_id", "document_type", "source", "topic", "chunk_id"}
        if self.vector_filters:
            invalid_keys = set(self.vector_filters.keys()) - allowed_filter_keys
            if invalid_keys:
                return False, f"Invalid filter keys: {invalid_keys}. Allowed: {allowed_filter_keys}"
        
        # Entity and relation lists must be lists of strings
        if self.graph_entities and not all(isinstance(e, str) for e in self.graph_entities):
            return False, "graph_entities must be a list of strings"
        
        if self.graph_relation_types and not all(isinstance(r, str) for r in self.graph_relation_types):
            return False, "graph_relation_types must be a list of strings"
        
        # Limit entity count to prevent query explosion
        if len(self.graph_entities) > 20:
            return False, f"Too many entities ({len(self.graph_entities)}). Max 20 allowed."
        
        return True, None


@dataclass
class UnifiedContext:
    """Merged retrieval results from vector and graph databases"""
    vector_chunks: List[Dict[str, Any]]
    graph_entities: List[Dict[str, Any]]
    graph_relations: List[Dict[str, Any]]
    retrieval_plan: RetrievalPlan
    total_tokens: int
    
    def to_context_string(self, max_chunks: int = 5) -> str:
        """
        Convert to a formatted context string for the LLM.
        
        Args:
            max_chunks: Maximum number of vector chunks to include
            
        Returns:
            Formatted context string
        """
        parts = []
        
        # Add vector context
        if self.vector_chunks:
            parts.append("=== DOCUMENT EXCERPTS ===")
            for i, chunk in enumerate(self.vector_chunks[:max_chunks], 1):
                text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                doc_id = metadata.get('document_id', 'unknown')
                parts.append(f"\n[Chunk {i} from {doc_id}]")
                parts.append(text)
        
        # Add graph context
        if self.graph_entities:
            parts.append("\n\n=== KNOWLEDGE GRAPH ENTITIES ===")
            for entity in self.graph_entities:
                name = entity.get('name', 'Unknown')
                entity_type = entity.get('type', 'Unknown')
                description = entity.get('description', 'N/A')
                parts.append(f"\n• {name} ({entity_type}): {description}")
        
        if self.graph_relations:
            parts.append("\n\n=== KNOWLEDGE GRAPH RELATIONSHIPS ===")
            for rel in self.graph_relations:
                subject = rel.get('subject', '?')
                relation = rel.get('relation', '?')
                obj = rel.get('object', '?')
                parts.append(f"\n• {subject} --[{relation}]--> {obj}")
        
        return "\n".join(parts)


class AgenticRetriever:
    """
    Orchestrates document-aware hybrid retrieval using an LLM planning agent.
    
    The agent analyzes user queries and generates retrieval plans that specify:
    - Which databases to use
    - What filters to apply
    - What entities/relations to retrieve
    
    This ensures retrieval stays within document scope and adapts to query type.
    """
    
    # System prompt for the retrieval planning agent
    RETRIEVAL_PLANNER_PROMPT = """You are a retrieval planning agent for a Hybrid RAG system that combines:
1. Vector Database (ChromaDB): For semantic search over document chunks
2. Knowledge Graph (Neo4j): For entity and relationship retrieval

Your job is to analyze the user's question and output a JSON retrieval plan.

Available metadata filters for vector DB:
- document_id: Specific document to search (e.g., "doc_123")
- document_type: Type of document (e.g., "research_paper", "legal_document")
- source: Source file name
- topic: Document topic or category

Available entity types in graph:
PERSON, ORGANIZATION, DATASET, MODEL, METHOD, TASK, DOMAIN, CONCEPT, LAW, LEGAL_TERM

Available relation types in graph:
uses, evaluated_on, compared_with, proposes, part_of, works_on, affiliated_with

Output a JSON object with this exact schema:
{
  "use_vector_db": true/false,
  "use_graph_db": true/false,
  "vector_filters": {"key": "value"},
  "graph_entities": ["Entity1", "Entity2"],
  "graph_relation_types": ["uses", "evaluated_on"],
  "reason": "Brief explanation of strategy"
}

Guidelines:
1. Definition questions → Vector DB only (semantic search)
2. Comparison questions → Both (vector for context + graph for relations)
3. "Who/what/where" about entities → Graph DB priority
4. Broad exploratory questions → Vector DB with minimal filters
5. Questions about specific documents → Apply document_id filter
6. Multi-hop reasoning → Graph DB with relation traversal

CRITICAL: Always stay within user-specified document scope. If they ask about "this document" or reference a specific source, apply strict document_id filters.

Examples:

User: "What is CODA-LM?"
Plan: {"use_vector_db": true, "use_graph_db": false, "vector_filters": {}, "graph_entities": [], "graph_relation_types": [], "reason": "Definition question - semantic search sufficient"}

User: "How does GPT-4V compare to Claude?"
Plan: {"use_vector_db": true, "use_graph_db": true, "vector_filters": {}, "graph_entities": ["GPT-4V", "Claude"], "graph_relation_types": ["compared_with", "evaluated_on"], "reason": "Comparison requires both semantic context and graph relationships"}

User: "What datasets are used in this research paper?"
Plan: {"use_vector_db": false, "use_graph_db": true, "vector_filters": {}, "graph_entities": [], "graph_relation_types": ["uses", "evaluated_on"], "reason": "Relationship query targeting datasets via graph traversal"}

User: "Summarize the key findings in document_123"
Plan: {"use_vector_db": true, "use_graph_db": false, "vector_filters": {"document_id": "document_123"}, "graph_entities": [], "graph_relation_types": [], "reason": "Document-scoped semantic search for comprehensive content"}

Now analyze this question and output ONLY the JSON plan (no other text):"""

    def __init__(
        self,
        chroma_client,
        neo4j_client,
        model: str = "phi3:mini",
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Initialize the agentic retriever.
        
        Args:
            chroma_client: ChromaDB client with .query(query, filters) method
            neo4j_client: Neo4j client with .query_entities() and .query_relations() methods
            model: Ollama model name for planning agent
            ollama_url: Ollama server URL
        """
        self.chroma_client = chroma_client
        self.neo4j_client = neo4j_client
        self.model = model
        self.ollama_client = ollama.Client(host=ollama_url)
        
        logger.info(f"🤖 Initialized AgenticRetriever with model={model}")
    
    def plan_retrieval(self, question: str, context_docs: Optional[List[str]] = None) -> RetrievalPlan:
        """
        Generate a retrieval plan using the LLM agent.
        
        Args:
            question: User's question
            context_docs: Optional list of document IDs to scope retrieval
            
        Returns:
            Validated RetrievalPlan
        """
        logger.info(f"🧠 Planning retrieval for question: {question[:100]}...")
        
        # Add document context to prompt if provided
        prompt = self.RETRIEVAL_PLANNER_PROMPT + f"\n\nQuestion: {question}"
        if context_docs:
            prompt += f"\n\nContext: User is asking about documents: {', '.join(context_docs)}"
        
        try:
            # Call LLM agent
            response = self.ollama_client.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": 0,  # Deterministic planning
                    "num_predict": 256,  # Short responses only
                }
            )
            
            response_text = response.get("response", "").strip()
            logger.debug(f"Agent response: {response_text}")
            
            # Parse JSON response
            try:
                plan_dict = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Agent returned invalid JSON: {e}")
                return self._get_fallback_plan(question, context_docs)
            
            # Convert to RetrievalPlan
            plan = RetrievalPlan(
                use_vector_db=plan_dict.get("use_vector_db", True),
                use_graph_db=plan_dict.get("use_graph_db", False),
                vector_filters=plan_dict.get("vector_filters", {}),
                graph_entities=plan_dict.get("graph_entities", []),
                graph_relation_types=plan_dict.get("graph_relation_types", []),
                reason=plan_dict.get("reason", "No reason provided")
            )
            
            # Apply document scope if provided
            if context_docs and plan.use_vector_db:
                if len(context_docs) == 1:
                    plan.vector_filters["document_id"] = context_docs[0]
                else:
                    plan.vector_filters["document_id"] = {"$in": context_docs}
            
            # Validate plan
            is_valid, error = plan.validate()
            if not is_valid:
                logger.warning(f"⚠️ Invalid plan: {error}. Using fallback.")
                return self._get_fallback_plan(question, context_docs)
            
            logger.info(f"✅ Generated retrieval plan: {plan.reason}")
            logger.debug(f"Plan details: {asdict(plan)}")
            
            return plan
            
        except Exception as e:
            logger.error(f"❌ Error calling agent: {e}")
            return self._get_fallback_plan(question, context_docs)
    
    def _get_fallback_plan(self, question: str, context_docs: Optional[List[str]] = None) -> RetrievalPlan:
        """
        Generate a safe fallback plan when agent fails.
        
        Default strategy: Use vector DB with document scope filters.
        """
        filters = {}
        if context_docs:
            if len(context_docs) == 1:
                filters["document_id"] = context_docs[0]
            else:
                filters["document_id"] = {"$in": context_docs}
        
        return RetrievalPlan(
            use_vector_db=True,
            use_graph_db=False,
            vector_filters=filters,
            graph_entities=[],
            graph_relation_types=[],
            reason="Fallback: Vector search with document scoping"
        )
    
    def execute_retrieval(
        self,
        question: str,
        plan: RetrievalPlan,
        top_k: int = 5
    ) -> UnifiedContext:
        """
        Execute the retrieval plan and merge results.
        
        Args:
            question: User's question (for vector search)
            plan: Validated retrieval plan
            top_k: Number of top results to retrieve
            
        Returns:
            UnifiedContext with merged results
        """
        logger.info(f"🔍 Executing retrieval plan...")
        
        vector_chunks = []
        graph_entities = []
        graph_relations = []
        total_tokens = 0
        
        # Execute vector retrieval
        if plan.use_vector_db:
            try:
                logger.info(f"📊 Querying vector DB with filters: {plan.vector_filters}")
                
                # Query ChromaDB with filters
                vector_results = self.chroma_client.query(
                    query_texts=[question],
                    n_results=top_k,
                    where=plan.vector_filters if plan.vector_filters else None
                )
                
                # Extract chunks with metadata
                if vector_results and vector_results.get('documents'):
                    for i, doc in enumerate(vector_results['documents'][0]):
                        metadata = vector_results.get('metadatas', [[]])[0][i] if i < len(vector_results.get('metadatas', [[]])[0]) else {}
                        distance = vector_results.get('distances', [[]])[0][i] if i < len(vector_results.get('distances', [[]])[0]) else 1.0
                        
                        vector_chunks.append({
                            'text': doc,
                            'metadata': metadata,
                            'distance': distance
                        })
                        total_tokens += len(doc.split())
                
                logger.info(f"✅ Retrieved {len(vector_chunks)} vector chunks")
                
            except Exception as e:
                logger.error(f"❌ Vector retrieval failed: {e}")
        
        # Execute graph retrieval
        if plan.use_graph_db:
            try:
                # Query entities
                if plan.graph_entities:
                    logger.info(f"🕸️  Querying graph for entities: {plan.graph_entities}")
                    graph_entities = self.neo4j_client.query_entities(
                        entity_names=plan.graph_entities
                    )
                    logger.info(f"✅ Retrieved {len(graph_entities)} entities")
                
                # Query relations
                if plan.graph_entities and plan.graph_relation_types:
                    logger.info(f"🕸️  Querying graph for relations: {plan.graph_relation_types}")
                    graph_relations = self.neo4j_client.query_relations(
                        entity_names=plan.graph_entities,
                        relation_types=plan.graph_relation_types
                    )
                    logger.info(f"✅ Retrieved {len(graph_relations)} relations")
                elif plan.graph_relation_types:
                    # Query relations without specific entities (broader search)
                    logger.info(f"🕸️  Querying all relations of types: {plan.graph_relation_types}")
                    graph_relations = self.neo4j_client.query_relations_by_type(
                        relation_types=plan.graph_relation_types
                    )
                    logger.info(f"✅ Retrieved {len(graph_relations)} relations")
                
                # Count tokens from graph results
                for entity in graph_entities:
                    total_tokens += len(str(entity).split())
                for relation in graph_relations:
                    total_tokens += len(str(relation).split())
                
            except Exception as e:
                logger.error(f"❌ Graph retrieval failed: {e}")
        
        # Create unified context
        context = UnifiedContext(
            vector_chunks=vector_chunks,
            graph_entities=graph_entities,
            graph_relations=graph_relations,
            retrieval_plan=plan,
            total_tokens=total_tokens
        )
        
        logger.info(f"✅ Unified context: {len(vector_chunks)} chunks, {len(graph_entities)} entities, {len(graph_relations)} relations (~{total_tokens} tokens)")
        
        return context
    
    def retrieve(
        self,
        question: str,
        context_docs: Optional[List[str]] = None,
        top_k: int = 5
    ) -> UnifiedContext:
        """
        End-to-end agentic retrieval: plan + execute.
        
        Args:
            question: User's question
            context_docs: Optional document IDs to scope retrieval
            top_k: Number of top results to retrieve
            
        Returns:
            UnifiedContext with merged results
        """
        # Step 1: Plan retrieval strategy
        plan = self.plan_retrieval(question, context_docs)
        
        # Step 2: Execute plan
        context = self.execute_retrieval(question, plan, top_k)
        
        return context


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Mock clients for testing
    class MockChromaClient:
        def query(self, query_texts, n_results, where=None):
            logger.info(f"Mock ChromaDB query: {query_texts[0][:50]}... (filters: {where})")
            return {
                'documents': [["Sample text chunk 1", "Sample text chunk 2"]],
                'metadatas': [[
                    {'document_id': 'doc_123', 'source': 'paper.pdf'},
                    {'document_id': 'doc_123', 'source': 'paper.pdf'}
                ]],
                'distances': [[0.3, 0.5]]
            }
    
    class MockNeo4jClient:
        def query_entities(self, entity_names):
            logger.info(f"Mock Neo4j entity query: {entity_names}")
            return [
                {'name': 'GPT-4V', 'type': 'MODEL', 'description': 'Vision-language model'},
                {'name': 'Claude', 'type': 'MODEL', 'description': 'Anthropic AI assistant'}
            ]
        
        def query_relations(self, entity_names, relation_types):
            logger.info(f"Mock Neo4j relation query: {entity_names} -> {relation_types}")
            return [
                {'subject': 'GPT-4V', 'relation': 'compared_with', 'object': 'Claude'},
                {'subject': 'GPT-4V', 'relation': 'evaluated_on', 'object': 'StreetHazards'}
            ]
        
        def query_relations_by_type(self, relation_types):
            logger.info(f"Mock Neo4j relation type query: {relation_types}")
            return []
    
    # Initialize retriever
    retriever = AgenticRetriever(
        chroma_client=MockChromaClient(),
        neo4j_client=MockNeo4jClient(),
        model="phi3:mini"
    )
    
    # Test retrieval
    print("\n" + "="*80)
    print("TEST 1: Comparison question")
    print("="*80)
    context = retriever.retrieve("How does GPT-4V compare to Claude?", top_k=3)
    print(f"\nPlan reason: {context.retrieval_plan.reason}")
    print(f"Retrieved: {len(context.vector_chunks)} chunks, {len(context.graph_entities)} entities, {len(context.graph_relations)} relations")
    print(f"\nFormatted context:\n{context.to_context_string(max_chunks=2)}")
    
    print("\n" + "="*80)
    print("TEST 2: Document-scoped question")
    print("="*80)
    context = retriever.retrieve(
        "Summarize the methodology",
        context_docs=["doc_123"],
        top_k=5
    )
    print(f"\nPlan reason: {context.retrieval_plan.reason}")
    print(f"Filters applied: {context.retrieval_plan.vector_filters}")
    print(f"Retrieved: {len(context.vector_chunks)} chunks")
