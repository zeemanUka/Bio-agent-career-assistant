"""
Bio Agent — RAG Pipeline
-------------------------
ChromaDB-backed knowledge retrieval.
Handles: PDF/text ingestion, chunking, and semantic search.
"""

import os
import shutil
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

from config import (
    CHROMA_ANONYMIZED_TELEMETRY,
    CHROMA_PATH,
    CHROMA_PRODUCT_TELEMETRY_IMPL,
    KNOWLEDGE_DIR,
    RAG_COLLECTION_NAME,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
)


# ── Module State ───────────────────────────────────────────────────────

_collection = None  # lazily initialised
_knowledge_ready = False


def _make_client() -> chromadb.PersistentClient:
    """Create a Chroma client with telemetry configured for local app usage."""
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(
            anonymized_telemetry=CHROMA_ANONYMIZED_TELEMETRY,
            chroma_product_telemetry_impl=CHROMA_PRODUCT_TELEMETRY_IMPL,
            chroma_telemetry_impl=CHROMA_PRODUCT_TELEMETRY_IMPL,
        ),
    )


def _get_collection():
    """Return the ChromaDB collection, creating it if needed."""
    global _collection
    if _collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        try:
            client = _make_client()
            _collection = client.get_or_create_collection(RAG_COLLECTION_NAME)
        except Exception as exc:
            if not os.path.isdir(KNOWLEDGE_DIR):
                raise

            # Rebuild from source docs if an older Chroma cache is incompatible
            # with the current runtime version.
            print(f"[RAG] Resetting incompatible Chroma store: {type(exc).__name__}: {exc}")
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
            os.makedirs(CHROMA_PATH, exist_ok=True)
            client = _make_client()
            _collection = client.get_or_create_collection(RAG_COLLECTION_NAME)
    return _collection


# ── Text Extraction ───────────────────────────────────────────────────

def _extract_pdf_text(path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_txt(path: str) -> str:
    """Read a plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ── Chunking ──────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[str]:
    """
    Split text into roughly equal chunks by word count with overlap.
    Returns a list of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward with overlap

    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────

def ingest_knowledge() -> int:
    """
    Read all files from the knowledge directory, chunk them,
    and store in ChromaDB. Returns the number of chunks stored.

    Safe to call multiple times — skips if chunks already exist.
    """
    global _knowledge_ready
    collection = _get_collection()

    # Skip if already ingested
    if _knowledge_ready:
        return collection.count()
    if collection.count() > 0:
        _knowledge_ready = True
        return collection.count()

    if not os.path.isdir(KNOWLEDGE_DIR):
        _knowledge_ready = True
        return 0

    all_text_parts = []

    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        if filename.lower().endswith(".pdf"):
            all_text_parts.append(_extract_pdf_text(filepath))
        elif filename.lower().endswith(".txt"):
            all_text_parts.append(_extract_txt(filepath))

    if not all_text_parts:
        _knowledge_ready = True
        return 0

    combined = "\n\n".join(all_text_parts)
    chunks = _chunk_text(combined)

    if not chunks:
        _knowledge_ready = True
        return 0

    # Generate stable IDs based on position
    ids = [f"chunk_{i:04d}" for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids)
    _knowledge_ready = True

    return len(chunks)


# ── Search ────────────────────────────────────────────────────────────

def search_knowledge_base(query: str) -> str:
    """
    Retrieve the top-K most relevant chunks for a query.
    Returns a formatted string of the matching chunks.
    """
    try:
        chunk_count = ingest_knowledge()
        collection = _get_collection()
    except Exception as exc:
        return (
            "Knowledge base retrieval is temporarily unavailable. "
            f"Runtime error: {type(exc).__name__}: {exc}"
        )

    if chunk_count == 0 or collection.count() == 0:
        return "No knowledge base documents found. Please ingest knowledge first."

    results = collection.query(query_texts=[query], n_results=RAG_TOP_K)

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "No relevant information found in the knowledge base."

    formatted = []
    for i, doc in enumerate(documents, 1):
        formatted.append(f"[Source {i}]\n{doc}")

    return "\n\n---\n\n".join(formatted)
