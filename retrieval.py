"""
Retrieval
----------
Accepts a natural language query, automatically detects which
professor(s) are being asked about, and returns the top-k most
relevant chunks from ChromaDB using the architecture we designed:

    Single professor  → K=7,  search scoped to one professor
    Two professors    → K=10 per professor, return both scored
    All professors    → K=10 per professor, return top 3 ranked

Usage:
    python retrieval.py                    # interactive mode
    python retrieval.py "your question"    # single query mode

Dependencies:
    pip install sentence-transformers chromadb rapidfuzz
"""

import sys
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import chromadb


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DB_DIR       = "chroma_db"
COLLECTION   = "professor_reviews"
MODEL_NAME   = "all-mpnet-base-v2"

K_SINGLE     = 5     # K for single professor queries
K_MULTI      = 7    # K per professor for cross-professor queries
SCORE_CUTOFF = 0.60  # minimum similarity — drop weak matches
FUZZY_CUTOFF = 85    # minimum fuzzy match score for name detection (0-100)


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def load_model_and_collection():
    """
    Loads the embedding model and connects to the ChromaDB collection.
    """
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Connecting to ChromaDB at ./{DB_DIR}")
    client     = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Collection '{COLLECTION}' — {collection.count()} chunks loaded\n")
    return model, collection


def get_all_professors(collection: chromadb.Collection) -> list:
    """
    Retrieves the unique list of all professor names stored in ChromaDB.
    """
    results    = collection.get(include=["metadatas"])
    professors = list({m["professor_name"] for m in results["metadatas"]})
    return sorted(professors)


# ─────────────────────────────────────────────
# NAME DETECTION
# ─────────────────────────────────────────────

def detect_professors(query: str, all_professors: list) -> list:
    """
    Scans the query for professor names using fuzzy matching.
    Handles full names, last names only, titles like Dr./Professor,
    minor typos, and missing punctuation.

    Returns a list of matched professor names from the database.

        0 matches → caller should use all professors
        1 match   → single professor mode
        2+ matches → multi professor mode

    Args:
        query:          the user's natural language question
        all_professors: full list of professor names from ChromaDB

    Examples:
        "How hard is Dr. Smith?"         → ["Dr. John Smith"]
        "Compare Smith and Jones"        → ["Dr. John Smith", "Dr. Amy Jones"]
        "Who is the easiest professor?"  → []
    """
    query_lower = query.lower()
    matched     = []

    for professor in all_professors:
        name_parts = professor.lower().split()

        for part in name_parts:
            # Skip titles and very short tokens
            if len(part) <= 2 or part in ("dr.", "dr", "mr.", "mr", "ms.", "ms", "professor", "prof", "prof."):
                continue

            # Fuzzy partial match — catches typos, missing punctuation, partial names
            score = fuzz.partial_ratio(part, query_lower)
            if score >= FUZZY_CUTOFF:
                matched.append(professor)
                break   # don't double-count this professor

    return matched


# ─────────────────────────────────────────────
# CORE RETRIEVAL
# ─────────────────────────────────────────────

def retrieve(
    query:      str,
    professors: list,
    model:      SentenceTransformer,
    collection: chromadb.Collection
) -> list:
    """
    Main retrieval function. Routes to the correct strategy
    based on how many professors are in the list.

        len == 1  → single professor  (K=7, direct results)
        len == 2  → two professors    (K=10 each, return both scored)
        len > 2   → all professors    (K=10 each, return top 3 ranked)

    Args:
        query:      natural language question
        professors: list of professor names to search within
        model:      loaded SentenceTransformer
        collection: ChromaDB collection

    Returns:
        list of result dicts
    """
    query_embedding = model.encode(query).tolist()

    if len(professors) == 1:
        return _query_single(query_embedding, professors[0], collection)

    return _query_multi(query_embedding, professors, collection)


def _query_single(
    query_embedding: list,
    professor:       str,
    collection:      chromadb.Collection
) -> list:
    """
    Searches within one professor's reviews only.
    Returns up to K=7 results filtered by score cutoff.
    """
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = K_SINGLE,
        where            = {"professor_name": professor},
        include          = ["documents", "metadatas", "distances"]
    )

    return _format_results(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )


def _query_multi(
    query_embedding: list,
    professors:      list,
    collection:      chromadb.Collection
) -> list:
    """
    Searches K=10 reviews per professor, scores each professor
    by average similarity, then returns:
        - both professors scored  (if len == 2)
        - top 3 ranked            (if len > 2)
    """
    professor_scores = {}
    professor_chunks = {}

    for professor in professors:
        try:
            results = collection.query(
                query_embeddings = [query_embedding],
                n_results        = K_MULTI,
                where            = {"professor_name": professor},
                include          = ["documents", "metadatas", "distances"]
            )
        except Exception:
            continue

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not docs:
            continue

        # Convert distances to similarity scores
        similarities = [round(1 - d, 4) for d in distances]

        filtered = list(zip(docs, metadatas, similarities))

        # Average similarity = professor's relevance score for this query
        avg_score = round(sum(s for _, _, s in filtered) / len(filtered), 4)
        professor_scores[professor] = avg_score
        professor_chunks[professor] = filtered

    # Rank professors by average score
    ranked = sorted(professor_scores.items(), key=lambda x: x[1], reverse=True)

    # Two professors → return both; all professors → return top 3
    top_professors = ranked if len(professors) == 2 else ranked[:3]

    # Build final result list
    output = []
    for professor, avg_score in top_professors:
        chunks = professor_chunks[professor]
        output.append({
            "professor_name": professor,
            "average_score":  avg_score,
            "review_count":   len(chunks),
            "top_reviews": [
                {
                    "text":           doc,
                    "score":          score,
                    "chunk_position": meta["chunk_position"],
                    "source":         meta["source"]
                }
                for doc, meta, score in chunks[:3]
            ]
        })

    return output


# ─────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────

def _format_results(docs: list, metadatas: list, distances: list) -> list:
    """
    Converts raw ChromaDB output into clean result dicts.
    Filters out results below the score cutoff.
    """
    results = []
    for doc, meta, distance in zip(docs, metadatas, distances):
        score = round(1 - distance, 4)
        if score < SCORE_CUTOFF:
            continue
        results.append({
            "professor_name":  meta["professor_name"],
            "score":           score,
            "chunk_position":  meta["chunk_position"],
            "source":          meta["source"],
            "text":            doc
        })
    return results


def print_results(query: str, results: list, professors: list) -> None:
    """
    Pretty-prints retrieval results to the console.
    Handles both single-professor and multi-professor output formats.
    """
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    if not results:
        print("No results found above the score cutoff.")
        return

    # Single professor output
    if len(professors) == 1:
        print(f"Professor: {professors[0]}\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']}  |  Position: {r['chunk_position']}")
            print(f"     {r['text']}\n")

    # Multi professor output
    else:
        for rank, result in enumerate(results, 1):
            print(f"\nRank {rank}: {result['professor_name']}")
            print(f"  Average relevance score: {result['average_score']}")
            print(f"  Matching reviews:        {result['review_count']}")
            print(f"  Top reviews:")
            for review in result["top_reviews"]:
                print(f"    [{review['score']}] {review['text'][:120]}...")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    model, collection = load_model_and_collection()
    all_professors    = get_all_professors(collection)

    print(f"Professors in database: {', '.join(all_professors)}")
    print("Type your question, or 'quit' to exit.\n")

    # Single query mode — passed as command line argument
    if len(sys.argv) > 1:
        query   = " ".join(sys.argv[1:])
        matched = detect_professors(query, all_professors)

        if len(matched) == 0:
            print("No professor named — comparing all professors")
            professors = all_professors
        elif len(matched) == 1:
            print(f"Single professor detected: {matched[0]}")
            professors = matched
        else:
            print(f"Multiple professors detected: {', '.join(matched)}")
            professors = matched

        results = retrieve(query, professors, model, collection)
        print_results(query, results, professors)
        return

    # Interactive mode — loop until user quits
    while True:
        query = input("Question: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        if not query:
            continue

        # Detect which professors the query is about
        matched = detect_professors(query, all_professors)

        if len(matched) == 0:
            print("→ No professor named — comparing all professors")
            professors = all_professors
        elif len(matched) == 1:
            print(f"→ Single professor detected: {matched[0]}")
            professors = matched
        else:
            print(f"→ Multiple professors detected: {', '.join(matched)}")
            professors = matched

        results = retrieve(query, professors, model, collection)
        print_results(query, results, professors)
        print()


if __name__ == "__main__":
    main()
