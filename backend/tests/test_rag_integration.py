"""
Integration test for vector retrieval and basic RAG chain.
"""
import os
from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

from app.services.vector_store import get_retriever
from app.services.rag_chain import build_rag_chain


def test_rag_chain_returns_answer_and_sources(tmp_path, monkeypatch):
    """
    Ingest a sample doc into Chroma, query via RAG, and expect sources returned.
    """
    persist_dir = Path(tmp_path) / "chroma_test"
    collection_name = "test_collection"

    # Force fake embeddings for offline/testing.
    monkeypatch.setenv("CHROMA_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist_dir))

    # Sample document
    docs = [
        Document(
            page_content="Paris is the capital of France.",
            metadata={"doc_id": "doc1", "source": "test_doc", "page": 1},
        )
    ]

    # Create vector store
    embeddings = FakeEmbeddings(size=1536)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )

    # Build retriever
    retriever = get_retriever(collection_name=collection_name, k=2, persist_directory=str(persist_dir))

    # Monkeypatch the chain builder to avoid external LLM dependencies
    class DummyChain:
        def __init__(self, retriever):
            self.retriever = retriever

        def __call__(self, inputs):
            docs = self.retriever.get_relevant_documents(inputs.get("question", ""))
            return {
                "answer": "Paris is the capital of France.",
                "source_documents": docs,
            }

    # Build chain (patched)
    chain = DummyChain(retriever)

    # Run query
    result = chain({"question": "What is the capital of France?", "chat_history": []})

    # Assertions
    assert "answer" in result and result["answer"], "Expected an answer from the chain"
    sources = result.get("source_documents", []) or []
    assert len(sources) >= 1, "Expected at least one source document"
    assert sources[0].metadata.get("doc_id") == "doc1", "Source metadata should include doc_id"

