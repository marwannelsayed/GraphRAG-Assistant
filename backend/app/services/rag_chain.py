"""
RAG (Retrieval-Augmented Generation) utilities.
"""
import os
import logging
import asyncio
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def build_rag_chain(llm, retriever):
    """
    Build a basic LangChain ConversationalRetrievalChain that returns source docs.

    Args:
        llm: A LangChain LLM instance (e.g., ChatOpenAI, FakeListLLM for tests)
        retriever: A LangChain retriever (e.g., Chroma retriever)

    Returns:
        A configured ConversationalRetrievalChain instance.
    """
    # Lazy import to avoid heavy dependencies during tests
    from langchain.chains import ConversationalRetrievalChain

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )


def hybrid_query(
    question: str,
    llm,
    retriever,
    graph_service,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Execute a hybrid query that combines graph and vector retrieval.
    
    This function:
    1. Queries both the knowledge graph and vector store in parallel
    2. Merges the results into a unified context
    3. Generates an answer using the LLM with proper citations
    4. Returns the answer with provenance information
    
    Args:
        question: The user's question
        llm: LangChain LLM instance
        retriever: LangChain vector store retriever
        graph_service: GraphService instance for graph queries
        top_k: Number of vector results to retrieve
        
    Returns:
        Dictionary containing:
        - answer: Generated answer string
        - sources: List of source documents with metadata
        - graph_context: Graph facts used in the answer
        - provenance: List of chunk IDs and node IDs used
    """
    logger.info(f"Executing hybrid query for question: {question}")
    
    # Execute graph and vector retrieval in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        graph_future = executor.submit(_query_graph, graph_service, question)
        vector_future = executor.submit(_query_vector, retriever, question, top_k)
        
        # Wait for both to complete
        graph_results = graph_future.result()
        vector_results = vector_future.result()
    
    logger.debug(f"Graph results: {len(graph_results.get('entities', []))} entities")
    logger.debug(f"Vector results: {len(vector_results)} documents")
    
    # Merge results into context
    merged_context = _merge_contexts(graph_results, vector_results, top_k)
    
    # Generate answer using LLM
    answer_data = _generate_answer_with_llm(
        llm=llm,
        question=question,
        graph_facts=merged_context["graph_facts"],
        text_snippets=merged_context["text_snippets"]
    )
    
    # Compile provenance
    provenance = {
        "chunk_ids": merged_context["chunk_ids"],
        "node_ids": graph_results.get("node_ids", []),
        "vector_doc_ids": [doc.metadata.get("doc_id") for doc in vector_results if hasattr(doc, "metadata")]
    }
    
    return {
        "answer": answer_data["answer"],
        "sources": merged_context["sources"],
        "graph_context": merged_context["graph_facts"],
        "provenance": provenance,
        "confidence": answer_data.get("confidence")
    }


def _query_graph(graph_service, question: str) -> Dict:
    """
    Query the knowledge graph for relevant entities and relationships.
    
    Args:
        graph_service: GraphService instance
        question: User's question
        
    Returns:
        Dictionary with entities, chunks, graph_facts, and node_ids
    """
    try:
        return graph_service.query_graph_for_question(question)
    except Exception as e:
        logger.error(f"Graph query failed: {e}", exc_info=True)
        return {"entities": [], "chunks": [], "graph_facts": "", "node_ids": []}


def _query_vector(retriever, question: str, top_k: int) -> List:
    """
    Query the vector store for relevant documents.
    
    Args:
        retriever: LangChain retriever
        question: User's question
        top_k: Number of results to retrieve
        
    Returns:
        List of Document objects
    """
    try:
        # Update search kwargs if needed
        if hasattr(retriever, 'search_kwargs'):
            retriever.search_kwargs['k'] = top_k
        return retriever.get_relevant_documents(question)
    except Exception as e:
        logger.error(f"Vector query failed: {e}", exc_info=True)
        return []


def _merge_contexts(graph_results: Dict, vector_results: List, top_k: int) -> Dict:
    """
    Merge graph and vector retrieval results into a unified context.
    
    Args:
        graph_results: Results from graph query
        vector_results: Results from vector query
        top_k: Number of top vector results to include
        
    Returns:
        Dictionary with merged context information
    """
    # Extract graph facts
    graph_facts = graph_results.get("graph_facts", "")
    
    # Get top K text snippets from vector results
    text_snippets = []
    sources = []
    chunk_ids = []
    
    for i, doc in enumerate(vector_results[:top_k]):
        if hasattr(doc, "page_content"):
            snippet_text = doc.page_content[:500]  # Limit snippet length
            text_snippets.append(f"[Snippet {i+1}] {snippet_text}")
            
            # Collect source information
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            sources.append({
                "type": "vector",
                "content": doc.page_content,
                "metadata": metadata,
                "snippet_id": i + 1
            })
            
            # Track chunk IDs
            if metadata.get("chunk_id"):
                chunk_ids.append(metadata["chunk_id"])
            elif metadata.get("doc_id"):
                chunk_ids.append(f"{metadata['doc_id']}_chunk_{i}")
    
    # Add graph chunks as sources
    for chunk in graph_results.get("chunks", []):
        if chunk.get("chunk_id") and chunk.get("text"):
            sources.append({
                "type": "graph",
                "content": chunk["text"],
                "metadata": {"chunk_id": chunk["chunk_id"]},
                "chunk_id": chunk["chunk_id"]
            })
            chunk_ids.append(chunk["chunk_id"])
    
    return {
        "graph_facts": graph_facts,
        "text_snippets": text_snippets,
        "sources": sources,
        "chunk_ids": list(set(chunk_ids))  # Remove duplicates
    }


def _generate_answer_with_llm(
    llm,
    question: str,
    graph_facts: str,
    text_snippets: List[str]
) -> Dict:
    """
    Generate an answer using the LLM with graph and vector context.
    
    Args:
        llm: LangChain LLM instance
        question: User's question
        graph_facts: Formatted graph context
        text_snippets: List of text snippets from vector search
        
    Returns:
        Dictionary with answer and optional confidence score
    """
    # Build the prompt
    prompt = _build_hybrid_prompt(question, graph_facts, text_snippets)
    
    try:
        # Generate answer
        if hasattr(llm, "invoke"):
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        elif hasattr(llm, "predict"):
            answer = llm.predict(prompt)
        else:
            # Fallback for different LLM interfaces
            answer = llm(prompt)
        
        return {
            "answer": answer,
            "confidence": None  # Could be extracted from LLM response metadata
        }
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        return {
            "answer": "I apologize, but I encountered an error generating an answer.",
            "confidence": None
        }


def _build_hybrid_prompt(question: str, graph_facts: str, text_snippets: List[str]) -> str:
    """
    Build a prompt that combines graph and vector context.
    
    Args:
        question: User's question
        graph_facts: Formatted graph context
        text_snippets: List of text snippets
        
    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        "You are a knowledgeable assistant that answers questions based on the provided context.",
        "Use the knowledge graph facts and text snippets below to provide an accurate, detailed answer.",
        "Always cite your sources by referring to snippet numbers [Snippet N] or entity names.",
        ""
    ]
    
    # Add graph context if available
    if graph_facts:
        prompt_parts.extend([
            graph_facts,
            ""
        ])
    
    # Add text snippets if available
    if text_snippets:
        prompt_parts.append("=== Text Snippets ===")
        prompt_parts.extend(text_snippets)
        prompt_parts.append("")
    
    # Add the question
    prompt_parts.extend([
        "=== Question ===",
        question,
        "",
        "=== Instructions ===",
        "Provide a comprehensive answer based on the context above.",
        "Include specific citations to entities or snippet numbers.",
        "If the context doesn't contain enough information, say so clearly.",
        "Be concise but complete in your response.",
        "",
        "Answer:"
    ])
    
    return "\n".join(prompt_parts)


"""
RAG (Retrieval-Augmented Generation) chain for answering queries.
"""
from typing import List, Dict, Optional


class RAGChain:
    """RAG chain for combining retrieval and generation."""
    
    def __init__(self):
        """Initialize RAG chain."""
        # TODO: Initialize LLM model (e.g., OpenAI, Anthropic, local model)
        pass
    
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        graph_context: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate answer using retrieved documents and graph context.
        
        TODO: Implement RAG chain:
        - Combine retrieved documents with graph context
        - Format prompt for LLM
        - Generate answer using LLM
        - Extract sources and confidence scores
        - Return structured response
        
        Args:
            query: User's question
            retrieved_docs: Documents retrieved from vector store
            graph_context: Entities and relationships from graph
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        raise NotImplementedError("RAG chain not implemented yet")


