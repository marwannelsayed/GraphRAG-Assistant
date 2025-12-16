"""
RAG (Retrieval-Augmented Generation) utilities.
"""
import os
import logging
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from app.services.vector_store import get_vector_store
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Initialize the CrossEncoder model once
# This model is optimized for performance and can run on CPU
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


def build_rag_chain(llm, retriever):
    """
    Build a basic LangChain ConversationalRetrievalChain that returns source docs.
    This is used for the vector-only (non-hybrid) query mode.
    """
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
    collection_name: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Execute a hybrid query that combines graph, vector, and keyword retrieval.

    This function:
    1. Queries the knowledge graph, vector store, and performs a keyword search in parallel.
    2. Merges and re-ranks the results into a unified, relevant context.
    3. Generates an answer using the LLM with proper citations.
    4. Returns the answer with provenance information.
    """
    logger.info(f"Executing hybrid query for question: '{question}'")

    # 1. Execute graph, vector, and keyword retrieval in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        graph_future = executor.submit(_query_graph, graph_service, question)
        vector_future = executor.submit(_query_vector, retriever, question, top_k)
        keyword_future = executor.submit(
            _perform_keyword_search, collection_name, question
        )

        graph_results = graph_future.result()
        vector_results = vector_future.result()
        keyword_results = keyword_future.result()

    logger.debug(f"Vector search returned {len(vector_results)} documents.")
    logger.debug(f"Keyword search returned {len(keyword_results)} documents.")

    # 2. Merge and re-rank results
    all_retrieved_docs = _merge_and_rerank_results(
        vector_results, keyword_results, question, top_k=top_k
    )

    # Diagnostic logging
    try:
        logger.info(f"All retrieved docs count: {len(all_retrieved_docs)}")
        for i, d in enumerate(all_retrieved_docs):
            md = d.metadata if hasattr(d, "metadata") else {}
            logger.info(
                f"Doc {i}: doc_id={md.get('doc_id')}, chunk_id={md.get('chunk_id')}, rerank_score={md.get('rerank_score')}, snippet={d.page_content[:120]!r}"
            )
    except Exception as e:
        logger.exception("Failed to log retrieved docs diagnostics")

    # 3. Build context for the LLM
    merged_context = _build_llm_context(graph_results, all_retrieved_docs)

    # 4. Generate answer using LLM
    # Safety: if there are no text snippets (document excerpts) and no graph facts,
    # do not call the LLM to avoid hallucination; return an explicit 'no info' message.
    if not merged_context["text_snippets"] and not merged_context["graph_facts"]:
        logger.info("No document excerpts or graph facts found; returning explicit 'no info' response to avoid hallucination")
        return {
            "answer": f"The provided excerpts do not contain information about '{question}'.",
            "sources": [],
            "graph_context": merged_context["graph_facts"],
            "provenance": provenance,
            "confidence": None,
        }

    answer_data = _generate_answer_with_llm(
        llm=llm,
        question=question,
        graph_facts=merged_context["graph_facts"],
        text_snippets=merged_context["text_snippets"],
    )

    # 5. Compile provenance
    provenance = {
        "chunk_ids": merged_context["chunk_ids"],
        "node_ids": graph_results.get("node_ids", []),
        "vector_doc_ids": [
            doc.metadata.get("doc_id")
            for doc in all_retrieved_docs
            if hasattr(doc, "metadata") and doc.metadata.get("doc_id")
        ],
    }

    return {
        "answer": answer_data["answer"],
        "sources": merged_context["sources"],
        "graph_context": merged_context["graph_facts"],
        "provenance": provenance,
        "confidence": answer_data.get("confidence"),
    }


def _extract_keywords(question: str) -> set:
    """
    Extracts significant keywords from a question, ignoring common stop words and punctuation.
    """
    import string

    # A basic list of English stop words.
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
        "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
        "will", "with", "what", "who", "which", "when", "where", "why", "how"
    }
    
    # Remove punctuation and convert to lower case
    translator = str.maketrans("", "", string.punctuation)
    question = question.translate(translator).lower()
    
    # Split into words, remove stop words, and filter for length
    words = question.split()
    return {word for word in words if word not in stop_words and len(word) > 2}


def _perform_keyword_search(collection_name: str, question: str) -> List[Document]:
    """
    Performs a direct keyword search across all documents in a collection.
    """
    keywords = _extract_keywords(question)
    logger.info(f"Performing keyword search with: {keywords}")

    try:
        vector_store = get_vector_store(collection_name)
        all_docs_data = vector_store._collection.get(include=["documents", "metadatas"])
        
        found_docs = []
        for i, doc_content in enumerate(all_docs_data.get("documents", [])):
            if not doc_content:
                continue
            
            content_lower = doc_content.lower()
            if any(kw in content_lower for kw in keywords):
                logger.info(f"Found keyword match in chunk {i} via keyword search.")
                metadata = all_docs_data.get("metadatas", [])[i] or {}
                doc = Document(page_content=doc_content, metadata=metadata)
                found_docs.append(doc)
        return found_docs
    except Exception as e:
        logger.error(f"Keyword search failed: {e}", exc_info=True)
        return []


def _merge_and_rerank_results(
    vector_results: List[Document],
    keyword_results: List[Document],
    question: str,
    top_k: int = 5,
) -> List[Document]:
    """
    Merges documents, removes duplicates, and re-ranks them using a CrossEncoder model.
    """
    # Combine and deduplicate documents
    merged_docs = {doc.page_content: doc for doc in vector_results}
    for doc in keyword_results:
        if doc.page_content not in merged_docs:
            merged_docs[doc.page_content] = doc
    
    unique_docs = list(merged_docs.values())
    
    if not unique_docs:
        return []

    # Create pairs of [question, document_content] for the CrossEncoder
    model_input = [[question, doc.page_content] for doc in unique_docs]
    
    # Get scores from the CrossEncoder model
    scores = rerank_model.predict(model_input)
    
    # Add scores to documents and sort
    for doc, score in zip(unique_docs, scores):
        doc.metadata["rerank_score"] = float(score)
        
    sorted_docs = sorted(unique_docs, key=lambda d: d.metadata["rerank_score"], reverse=True)
    
    if not sorted_docs:
        return []

    top_doc_score = sorted_docs[0].metadata["rerank_score"]
    logger.info(f"Re-ranked {len(sorted_docs)} documents with CrossEncoder. Top score: {top_doc_score:.4f}")

    return sorted_docs[:top_k]


def _query_graph(graph_service, question: str) -> Dict:
    """
    Query the knowledge graph for relevant entities and relationships.
    """
    try:
        return graph_service.query_graph_for_question(question)
    except Exception as e:
        logger.error(f"Graph query failed: {e}", exc_info=True)
        return {"entities": [], "chunks": [], "graph_facts": "", "node_ids": []}


def _query_vector(retriever, question: str, top_k: int) -> List[Document]:
    """
    Query the vector store for relevant documents.

    This helper is defensive: different retriever implementations expose different methods
    (get_relevant_documents, get_relevant_texts, retrieve, invoke). We try common ones
    and normalise the output to a List[Document].
    """
    try:
        docs = []
        # Try common LangChain retriever API
        if hasattr(retriever, "get_relevant_documents"):
            logger.info("Using retriever.get_relevant_documents")
            docs = retriever.get_relevant_documents(question)
        elif hasattr(retriever, "get_relevant_texts"):
            logger.info("Using retriever.get_relevant_texts")
            texts = retriever.get_relevant_texts(question)
            docs = [Document(page_content=t) for t in texts]
        elif hasattr(retriever, "retrieve"):
            logger.info("Using retriever.retrieve")
            docs = retriever.retrieve(question)
        elif hasattr(retriever, "invoke"):
            logger.info("Using retriever.invoke")
            res = retriever.invoke(question)
            # Normalise possible shapes
            if isinstance(res, list):
                docs = res
            elif isinstance(res, dict) and res.get("documents"):
                docs = [Document(page_content=d) for d in res.get("documents")]
            else:
                # Fallback: try to coerce to a single Document
                docs = [Document(page_content=str(res))]

        # Ensure we have a list of Document objects
        normalised = []
        for d in docs:
            if isinstance(d, Document):
                normalised.append(d)
            elif isinstance(d, dict) and d.get("page_content"):
                normalised.append(Document(page_content=d.get("page_content"), metadata=d.get("metadata", {})))
            elif isinstance(d, str):
                normalised.append(Document(page_content=d))
            else:
                # last resort, convert to str
                normalised.append(Document(page_content=str(d)))

        # Trim to top_k if needed
        if len(normalised) > top_k:
            normalised = normalised[:top_k]

        logger.info(f"_query_vector: retrieved {len(normalised)} documents (type samples: {[type(x).__name__ for x in normalised[:3]]})")
        for i, doc in enumerate(normalised[:3]):
            md = doc.metadata if hasattr(doc, "metadata") else {}
            logger.info(f"_query_vector sample {i}: len={len(doc.page_content or '')}, metadata_keys={list(md.keys())}")

        return normalised
    except Exception as e:
        logger.error(f"Vector query failed: {e}", exc_info=True)
        return []


def _build_llm_context(
    graph_results: Dict, ranked_docs: List[Document]
) -> Dict:
    """
    Builds the final context for the LLM from graph results and top-ranked documents.
    """
    graph_facts = graph_results.get("graph_facts", "")
    
    text_snippets = []
    sources = []
    chunk_ids = []
    
    logger.info(f"_build_llm_context: received {len(ranked_docs)} ranked_docs")
    for i, doc in enumerate(ranked_docs):
        snippet_text = doc.page_content[:1000]
        text_snippets.append(f"[Snippet {i+1}] {snippet_text}")
        
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        sources.append(
            {
                "type": "retrieved_document",
                "content": doc.page_content,
                "metadata": metadata,
                "snippet_id": i + 1,
            }
        )
        
        if metadata.get("chunk_id"):
            chunk_ids.append(metadata["chunk_id"])
        elif metadata.get("doc_id"):
            chunk_ids.append(f"{metadata['doc_id']}_chunk_{i}")

    logger.info(f"_build_llm_context: built {len(sources)} sources and {len(chunk_ids)} chunk_ids from ranked_docs")

    for chunk in graph_results.get("chunks", []):
        if chunk.get("chunk_id") and chunk.get("text"):
            sources.append(
                {
                    "type": "graph_chunk",
                    "content": chunk["text"],
                    "metadata": {"chunk_id": chunk["chunk_id"]},
                    "chunk_id": chunk["chunk_id"],
                }
            )
            chunk_ids.append(chunk["chunk_id"])
    
    logger.info(f"_build_llm_context: total sources after adding graph chunks: {len(sources)}")
    
    return {
        "graph_facts": graph_facts,
        "text_snippets": text_snippets,
        "sources": sources,
        "chunk_ids": list(set(chunk_ids)),
    }


def _generate_answer_with_llm(
    llm, question: str, graph_facts: str, text_snippets: List[str]
) -> Dict:
    """
    Generate an answer using the LLM with the combined context.
    """
    prompt = _build_hybrid_prompt(question, graph_facts, text_snippets)
    
    try:
        if hasattr(llm, "invoke"):
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        else:
            answer = llm.predict(prompt)
        
        return {"answer": answer, "confidence": None}
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        return {
            "answer": "I apologize, but I encountered an error generating an answer.",
            "confidence": None,
        }


def _build_hybrid_prompt(question: str, graph_facts: str, text_snippets: List[str]) -> str:
    """
    Build a robust prompt for fact-checking and answering with strict grounding.
    """
    prompt_parts = [
        "You are a knowledgeable assistant that provides clear, direct answers based ONLY on the provided documents.",
        "",
        "**CRITICAL RULES (Anti-Hallucination):**",
        "1. ONLY use facts explicitly stated in the 'Document Excerpts' or 'Knowledge Graph Context' below.",
        "2. If information is not in the provided context, say: 'I don't have that information in the provided documents.'",
        "3. NEVER expand acronyms or add explanations unless they appear in the context.",
        "4. NEVER infer, assume, or use external knowledge - stick to what's written.",
        "",
        "**Answer Style:**",
        "- Answer directly and naturally (no phrases like 'According to the document...')",
        "- Be concise and clear",
        "- If uncertain, acknowledge: 'The documents don't specify...'",
        "",
    ]

    if text_snippets:
        prompt_parts.append("=== Document Excerpts ===")
        prompt_parts.extend(text_snippets)
        prompt_parts.append("")
    
    if graph_facts:
        prompt_parts.extend([
            "=== Knowledge Graph Context ===",
            graph_facts,
            ""
        ])
    
    prompt_parts.extend([
        "=== Question ===",
        question,
        "",
        "=== Your Answer ===",
        "Answer using ONLY the information above:",
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


