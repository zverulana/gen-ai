from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
CHROMA_PATH = ROOT / "output" / "chroma_db"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_collection = None


def _chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def get_collection():
    global _collection
    if _collection is None:
        CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _collection = client.get_or_create_collection(
            name="support_tickets",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def ingest(corpus_path: Path | None = None) -> dict:
    corpus_path = corpus_path or INPUT / "tickets_corpus.csv"
    df = pd.read_csv(corpus_path)
    collection = get_collection()
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    docs, ids, metas = [], [], []
    for _, row in df.iterrows():
        body = (
            f"Category: {row['category']}\n"
            f"Product: {row['product']}\n"
            f"Issue: {row['issue_description']}\n"
            f"Resolution: {row['resolution_notes']}"
        )
        tid = int(row["ticket_id"])
        for i, chunk in enumerate(_chunk_text(body)):
            docs.append(chunk)
            ids.append(f"{tid}__{i}")
            metas.append(
                {
                    "ticket_id": tid,
                    "category": str(row["category"]),
                    "product": str(row["product"]),
                    "chunk_id": i,
                }
            )

    if docs:
        batch = 128
        for i in range(0, len(docs), batch):
            collection.add(
                documents=docs[i : i + batch],
                ids=ids[i : i + batch],
                metadatas=metas[i : i + batch],
            )

    stats = {"chunks": len(docs), "tickets": len(df)}
    (ROOT / "output" / "rag_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def search_similar_tickets(query: str, category_hint: str | None = None, k: int = 4) -> dict:
    collection = get_collection()
    if collection.count() == 0:
        ingest()
    where = {"category": category_hint} if category_hint else None
    try:
        hits = collection.query(query_texts=[query], n_results=k, where=where)
    except Exception:
        hits = collection.query(query_texts=[query], n_results=k)

    results = []
    seen: set[int] = set()
    for doc, meta, dist in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0]):
        tid = int(meta["ticket_id"])
        if tid in seen:
            continue
        seen.add(tid)
        results.append(
            {
                "ticket_id": tid,
                "category": meta["category"],
                "product": meta["product"],
                "snippet": doc[:600],
                "distance": round(float(dist), 4),
            }
        )
    return {"hits": results, "query": query}


def build_context_blob(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        parts.append(f"[ticket {h['ticket_id']}] {h['snippet']}")
    return "\n\n".join(parts)
