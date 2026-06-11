import argparse
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"
CHROMA_PATH = Path(__file__).parent / "chroma_db"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "],
)


def chunk_fixed(text: str, chunk_size: int = 2000) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_recursive(text: str) -> list[str]:
    return [c.strip() for c in _recursive_splitter.split_text(text) if c.strip()]


def get_chunker(strategy: str):
    if strategy == "fixed":
        return chunk_fixed
    if strategy == "recursive":
        return chunk_recursive
    raise ValueError(f"Неизвестная стратегия: {strategy}. Используйте fixed или recursive.")


def _make_collection(strategy: str):
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
    )
    name = f"corpus_{strategy}"
    return chroma.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


_collections: dict[str, object] = {}


def get_collection(strategy: str = "recursive"):
    if strategy not in _collections:
        _collections[strategy] = _make_collection(strategy)
    return _collections[strategy]


def ingest(strategy: str = "recursive"):
    chunker = get_chunker(strategy)
    collection = get_collection(strategy)

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    all_chunks, all_ids, all_meta = [], [], []

    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        chunks = chunker(text)

        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
            all_meta.append({"source": f.stem, "chunk_id": i, "strategy": strategy})

        print(f"  {f.stem}: {len(chunks)} чанков")

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)
    print(
        f"\n[{strategy}] Индексировано {collection.count()} чанков "
        f"из {len(list(DATA_DIR.glob('*.txt')))} файлов"
    )


def retrieve(query: str, k: int = 5, strategy: str = "recursive") -> dict:
    collection = get_collection(strategy)
    return collection.query(query_texts=[query], n_results=k)


def ask(query: str, strategy: str = "recursive"):
    print(f"Поиск [{strategy}]...", flush=True)
    t0 = time.time()
    hits = retrieve(query, k=5, strategy=strategy)
    found = hits["ids"][0]
    print(f"  нашёл {len(found)} чанков за {time.time() - t0:.1f}с", flush=True)

    print("\n" + "=" * 60)
    print(f"ВОПРОС: {query}")
    print("=" * 60)
    for cid, doc in zip(found, hits["documents"][0]):
        print(f"\n--- [{cid}] ---\n{doc[:300]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest", "ask"])
    parser.add_argument("question", nargs="?", default="")
    parser.add_argument("--strategy", choices=["fixed", "recursive"], default="recursive")
    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.strategy)
    elif args.command == "ask":
        if not args.question:
            print('Нужен вопрос: python pipeline.py ask "..." --strategy recursive')
            sys.exit(1)
        ask(args.question, args.strategy)
