"""MP2 Mini-RAG pipeline over the Sherlock Holmes corpus.

Run from this directory after configuring learner/.env:
    python mp2_rag.py ingest
    python mp2_rag.py ask
    python mp2_rag.py validate
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ─── Configuration ──────────────────────────────────────────────────────

LEARNER_DIR       = Path(__file__).resolve().parent
CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters

load_dotenv(LEARNER_DIR / ".env")

_openai_client: OpenAI | None = None
_qdrant_client: QdrantClient | None = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing; add it to learner/.env.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is missing; configure the Vocareum endpoint in learner/.env.")
        _openai_client = OpenAI(api_key=api_key, base_url=base_url)
    return _openai_client


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_settings = dotenv_values(LEARNER_DIR / ".env.example")
        url = (
            qdrant_settings.get("QDRANT_URL")
            or qdrant_settings.get("qdrant_url")
            or os.environ.get("QDRANT_URL")
            or os.environ.get("qdrant_url")
        )
        if not url:
            raise RuntimeError("QDRANT_URL is missing; add it to learner/.env.example.")
        _qdrant_client = QdrantClient(
            url=url,
            api_key=(
                qdrant_settings.get("QDRANT_API_KEY")
                or qdrant_settings.get("qdrant_api_key")
                or os.environ.get("QDRANT_API_KEY")
                or os.environ.get("qdrant_api_key")
                or None
            ),
        )
    return _qdrant_client


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).

    Files are read in name order so ingestion and citations are reproducible.
    """
    documents: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
        if text:
            documents.append({"source": path.name, "title": title, "text": text})
    return documents


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk should be a dict with: source, title, section, text.

    Approach (your choice):
      - Simple: fixed-size windows (split text into N-character chunks with overlap)
      - Smarter: split on paragraph boundaries (\\n\\n), then pack paragraphs
        into chunks up to TARGET_CHUNK_SIZE characters

    The reference solution uses the smarter approach, plus heuristic
    section-header detection (short lines without terminal punctuation).
    Either approach is acceptable.

    Paragraphs are packed together up to the target size. Long paragraphs are
    split at word boundaries, with a small overlap to preserve local context.
    """
    text = str(doc.get("text", "")).strip()
    if not text:
        return []

    title = str(doc.get("title") or "Untitled")
    source = str(doc.get("source") or "unknown")
    lines = text.splitlines()
    current_section = "Overview"
    paragraphs: list[tuple[str, str]] = []
    pending: list[str] = []

    def save_pending() -> None:
        paragraph = " ".join(part.strip() for part in pending if part.strip()).strip()
        if paragraph:
            paragraphs.append((current_section, paragraph))
        pending.clear()

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            save_pending()
            continue
        is_heading = (
            len(stripped) <= 80
            and not re.search(r"[.!?;:]$", stripped)
            and len(stripped.split()) <= 10
        )
        if is_heading:
            save_pending()
            current_section = stripped
        else:
            pending.append(stripped)
    save_pending()
    if not paragraphs:
        paragraphs = [(current_section, text)]

    pieces: list[tuple[str, str]] = []
    for section, paragraph in paragraphs:
        if len(paragraph) <= TARGET_CHUNK_SIZE:
            pieces.append((section, paragraph))
            continue
        remaining = paragraph
        while len(remaining) > TARGET_CHUNK_SIZE:
            boundary = remaining.rfind(" ", 0, TARGET_CHUNK_SIZE + 1)
            if boundary < TARGET_CHUNK_SIZE // 2:
                boundary = TARGET_CHUNK_SIZE
            pieces.append((section, remaining[:boundary].strip()))
            remaining = remaining[max(0, boundary - CHUNK_OVERLAP):].strip()
        if remaining:
            pieces.append((section, remaining))

    chunks: list[dict[str, Any]] = []
    chunk_section = ""
    chunk_text = ""

    def save_chunk() -> None:
        if chunk_text:
            chunks.append({"source": source, "title": title, "section": chunk_section, "text": chunk_text})

    for section, piece in pieces:
        separator = "\n\n" if chunk_text else ""
        if chunk_text and (chunk_section != section or len(chunk_text) + len(separator) + len(piece) > TARGET_CHUNK_SIZE):
            previous_section = chunk_section
            save_chunk()
            overlap = chunk_text[-CHUNK_OVERLAP:].lstrip() if previous_section == section else ""
            chunk_text = overlap
            chunk_section = section
            separator = "\n\n" if chunk_text else ""
            if len(chunk_text) + len(separator) + len(piece) > TARGET_CHUNK_SIZE:
                chunk_text = ""
                separator = ""
        if not chunk_text:
            chunk_section = section
        chunk_text = f"{chunk_text}{separator}{piece}" if chunk_text else piece

    save_chunk()
    return chunks


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).

    The API response is reordered by its explicit input index before returning.
    """
    if not texts:
        return []

    vectors: list[list[float] | None] = [None] * len(texts)
    client = get_openai_client()
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in response.data:
            vectors[start + item.index] = item.embedding
    if any(vector is None for vector in vectors):
        raise RuntimeError("OpenAI returned an incomplete embedding response.")
    return [vector for vector in vectors if vector is not None]


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection.

    Existing collections are replaced to keep ingestion idempotent.
    """
    client = get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant.

    Each chunk's source metadata is retained as its Qdrant payload.
    """
    if not chunks:
        return
    vectors = embed_texts([str(chunk["text"]) for chunk in chunks])
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload=chunk)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    get_qdrant_client().upsert(collection_name=COLLECTION_NAME, points=points, wait=True)


# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query.

    Supports both the current query_points API and older search API versions.
    """
    if k <= 0:
        return []
    query_vector = embed_texts([query])[0]
    client = get_qdrant_client()
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        hits = response.points
    else:
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
        )

    results: list[dict[str, Any]] = []
    for hit in hits:
        payload = dict(hit.payload or {})
        payload["score"] = float(hit.score)
        results.append(payload)
    return results


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result.

    The returned citations are structured so both the CLI and validator can use them.
    """
    started_at = time.perf_counter()
    chunks = retrieve(question, k=k)
    citations: list[dict[str, Any]] = []
    context_parts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        title = str(chunk.get("title") or chunk.get("source") or "Unknown source")
        section = str(chunk.get("section") or "Unspecified section")
        context_parts.append(
            f"[Source: {title} — {section}]\n{chunk.get('text', '')}"
        )
        citation_key = (str(chunk.get("source", "")), section)
        if citation_key not in seen:
            citations.append({
                "source": citation_key[0],
                "title": title,
                "section": section,
                "score": chunk.get("score"),
            })
            seen.add(citation_key)

    context = "\n\n---\n\n".join(context_parts) or "No relevant excerpts were found."
    response = get_openai_client().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", CHAT_MODEL),
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
        ],
    )
    answer_text = response.choices[0].message.content or "I could not produce an answer from the excerpts."
    return {
        "question": question,
        "answer": answer_text.strip(),
        "citations": citations,
        "latency_ms": round((time.perf_counter() - started_at) * 1000),
    }


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
