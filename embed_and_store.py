"""
Embedding and Vector Store Ingestion
--------------------------------------
Loads chunks.json produced by rmp_scraper.py, encodes each chunk
using all-mpnet-base-v2, attaches metadata, and stores everything
in a local ChromaDB collection.

Usage:
    python embed_and_store.py
    python embed_and_store.py chunks.json          # custom chunks file
    python embed_and_store.py chunks.json my_db    # custom db directory

Dependencies:
    pip install sentence-transformers chromadb
"""

import sys
import json
from sentence_transformers import SentenceTransformer
import chromadb


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CHUNKS_FILE   = sys.argv[1] if len(sys.argv) > 1 else "chunks.json"
DB_DIR        = sys.argv[2] if len(sys.argv) > 2 else "chroma_db"
COLLECTION    = "professor_reviews"
MODEL_NAME    = "all-mpnet-base-v2"
BATCH_SIZE    = 32   # how many chunks to encode at once


# ─────────────────────────────────────────────
# STEP 1 — LOAD CHUNKS
# ─────────────────────────────────────────────

def load_chunks(path: str) -> list:
    """
    Loads the chunks.json file produced by rmp_scraper.py.

    Each chunk is expected to have:
        chunk_id        int    — global position across all professors
        professor_name  str    — name of the professor
        text            str    — cleaned review text
        token_estimate  int    — approximate word count
    """
    print(f"Loading chunks from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# STEP 2 — ENCODE
# ─────────────────────────────────────────────

def encode_chunks(chunks: list, model: SentenceTransformer) -> list:
    """
    Encodes all chunk texts into vectors using all-mpnet-base-v2.
    Processes in batches for efficiency.

    Returns the same chunks list with an added 'embedding' key.
    """
    print(f"\nEncoding {len(chunks)} chunks with {MODEL_NAME}...")

    texts = [chunk["text"] for chunk in chunks]

    # Encode in batches — show_progress_bar gives a live progress indicator
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_list=True    # return plain Python lists, not tensors
    )

    # Attach each embedding back to its chunk
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    print(f"Encoding complete — {len(embeddings)} vectors produced")
    return chunks


# ─────────────────────────────────────────────
# STEP 3 — BUILD METADATA
# ─────────────────────────────────────────────

def build_metadata(chunk: dict, source_file: str) -> dict:
    """
    Constructs the metadata dict stored alongside each vector in ChromaDB.

    Metadata fields:
        source          str   — name of the file the chunk came from
        chunk_position  int   — position of this chunk within its source document
        professor_name  str   — which professor this review belongs to
        token_estimate  int   — approximate length of the review
    """
    # Derive a per-professor position by tracking how many chunks
    # we've seen for this professor (chunk_id is global, we want local)
    professor = chunk["professor_name"]
    return {
        "source":          f"Rate My Professor - {professor}",
        "chunk_position":  chunk["chunk_id"],
        "professor_name":  professor,
        "token_estimate":  chunk["token_estimate"]
    }


# ─────────────────────────────────────────────
# STEP 4 — STORE IN CHROMADB
# ─────────────────────────────────────────────

def store_in_chroma(chunks: list, source_file: str, db_dir: str) -> chromadb.Collection:
    """
    Initialises a local ChromaDB instance and upserts all chunks.

    Uses upsert (not add) so the script is safe to re-run —
    existing chunks are updated rather than duplicated.

    Args:
        chunks:      encoded chunks with embeddings attached
        source_file: filename of the chunks JSON, stored as metadata
        db_dir:      directory where ChromaDB persists its data

    Returns:
        the ChromaDB collection object
    """
    print(f"\nConnecting to ChromaDB at ./{db_dir}...")
    client     = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}   # use cosine similarity (matches our model)
    )

    print(f"Storing {len(chunks)} chunks in collection '{COLLECTION}'...")

    # ChromaDB expects separate lists for ids, embeddings, documents, metadatas
    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append(build_metadata(chunk, source_file))

    # Upsert in batches to avoid memory issues with large collections
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids        = ids[i : i + batch_size],
            embeddings = embeddings[i : i + batch_size],
            documents  = documents[i : i + batch_size],
            metadatas  = metadatas[i : i + batch_size]
        )
        print(f"  Stored chunks {i} – {min(i + batch_size, len(ids))}")

    print(f"\nDone — {collection.count()} total chunks in ChromaDB")
    return collection


# ─────────────────────────────────────────────
# STEP 5 — VERIFY
# ─────────────────────────────────────────────

def verify_store(collection: chromadb.Collection, model:SentenceTransformer) -> None:
    """
    Runs a quick sanity check query against the collection
    to confirm embeddings were stored and are searchable.
    """
    print("\n" + "=" * 60)
    print("VERIFICATION — test query")
    print("=" * 60)

    test_query = "good professor who explains clearly"
    query_embedding = model.encode(test_query).tolist()
    results    = collection.query(
        query_embeddings = [model.encode(test_query).tolist()],
        n_results   = 3
    )

    print(f"Query: '{test_query}'\n")
    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = round(1 - distance, 4)   # cosine: convert distance to similarity
        print(f"Result {i + 1}  (similarity: {similarity})")
        print(f"  Professor: {meta['professor_name']}")
        print(f"  Position:  {meta['chunk_position']}")
        print(f"  Source:    {meta['source']}")
        print(f"  Text:      {doc[:120]}...")
        print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Step 1: Load ──
    chunks = load_chunks(CHUNKS_FILE)

    # ── Step 2: Encode ──
    print(f"\nLoading model: {MODEL_NAME}")
    model  = SentenceTransformer(MODEL_NAME)
    chunks = encode_chunks(chunks, model)

    # ── Step 3 & 4: Build metadata and store ──
    collection = store_in_chroma(chunks, CHUNKS_FILE, DB_DIR)

    # ── Step 5: Verify ──
    verify_store(collection, model)

    print("\nPipeline complete.")
    print(f"  Chunks file:  {CHUNKS_FILE}")
    print(f"  Database:     ./{DB_DIR}")
    print(f"  Collection:   {COLLECTION}")
    print(f"  Total stored: {collection.count()}")


if __name__ == "__main__":
    main()
