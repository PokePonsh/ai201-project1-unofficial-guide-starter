"""
RAG Pipeline — Retrieval + Grounded Generation
------------------------------------------------
Wires the retrieval system into Groq's llama-3.3-70b-versatile.
Forces the LLM to answer only from retrieved student reviews,
never from outside knowledge.

Usage:
    python rag.py                        # interactive mode
    python rag.py "your question here"   # single query mode

Dependencies:
    pip install sentence-transformers chromadb rapidfuzz groq
"""

import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from groq import Groq
import chromadb

load_dotenv()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DB_DIR       = "chroma_db"
COLLECTION   = "professor_reviews"
MODEL_NAME   = "all-mpnet-base-v2"
GROQ_MODEL   = "llama-3.3-70b-versatile"

K_SINGLE     = 5
K_MULTI      = 7
SCORE_CUTOFF = 0.60
FUZZY_CUTOFF = 90


# ─────────────────────────────────────────────
# SYSTEM PROMPT — grounding rules for the LLM
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an assistant that helps students at Temple 
University choose chemistry professors based solely on Rate My Professor 
student reviews.

STRICT RULES — follow these without exception:
1. Answer ONLY using the student reviews provided to you in each query.
2. NEVER use outside knowledge about professors, universities, or courses.
3. NEVER make up or infer information not present in the reviews.
4. If the provided reviews do not contain enough information to answer 
   the question, respond with: "I don't have enough review data to 
   answer that question."
5. Always reference which professor's reviews you are drawing from.
6. If reviews contradict each other, acknowledge both perspectives.
7. Do not recommend a professor based on anything other than what 
   students have written in the provided reviews.
8. Always end your response by listing the source documents you used
   under a line that reads: "Sources:" followed by each document name
   and review position.

You are a summarizer of student opinions, not an expert advisor."""


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def load_clients():
    """
    Loads the embedding model, ChromaDB collection, and Groq client.
    Reads the Groq API key from the GROQ_API_KEY environment variable.
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Connecting to ChromaDB at ./{DB_DIR}")
    client     = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{COLLECTION}' — {collection.count()} chunks loaded")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set.\n"
            "Set it with: set GROQ_API_KEY=your_key_here  (Windows)\n"
            "         or: export GROQ_API_KEY=your_key_here  (Mac/Linux)"
        )
    groq_client = Groq(api_key=api_key)
    print(f"Groq client ready — model: {GROQ_MODEL}\n")

    return model, collection, groq_client


def get_all_professors(collection: chromadb.Collection) -> list:
    """
    Retrieves the unique list of all professor names from ChromaDB.
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
    Returns matched professor names from the database.

        0 matches → use all professors
        1 match   → single professor mode
        2+ matches → multi professor mode
    """
    query_lower = query.lower()
    matched     = []
    skip_tokens = {
        "dr.", "dr", "mr.", "mr", "ms.", "ms",
        "professor", "prof", "prof."
    }

    for professor in all_professors:
        name_parts = professor.lower().split()

        for part in name_parts:
            if len(part) <= 3 or part in skip_tokens:
                continue

            score = fuzz.partial_ratio(part, query_lower)
            if score >= FUZZY_CUTOFF:
                matched.append(professor)
                break

    return matched


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def retrieve(
    query:      str,
    professors: list,
    model:      SentenceTransformer,
    collection: chromadb.Collection
) -> list:
    """
    Routes to the correct retrieval strategy based on professor count.
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
    K=5, scoped to one professor, filtered by score cutoff.
    """
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = K_SINGLE,
        where            = {"professor_name": professor},
        include          = ["documents", "metadatas", "distances"]
    )

    output = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = round(1 - distance, 4)
        if score < SCORE_CUTOFF:
            continue
        output.append({
            "professor_name":  meta["professor_name"],
            "score":           score,
            "chunk_position":  meta["chunk_position"],
            "source":          meta["source"],
            "text":            doc
        })

    return output


def _query_multi(
    query_embedding: list,
    professors:      list,
    collection:      chromadb.Collection
) -> list:
    """
    K=7 per professor, no cutoff filtering, ranked by average score.
    Always returns top 3 (or both if comparing two professors).
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

        # No cutoff filtering in multi mode — ranking handles relevance
        similarities = [round(1 - d, 4) for d in distances]
        filtered     = list(zip(docs, metadatas, similarities))

        avg_score = round(sum(s for _, _, s in filtered) / len(filtered), 4)
        professor_scores[professor] = avg_score
        professor_chunks[professor] = filtered

    ranked         = sorted(professor_scores.items(), key=lambda x: x[1], reverse=True)
    top_professors = ranked if len(professors) == 2 else ranked[:3]

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
# PROMPT BUILDING
# ─────────────────────────────────────────────

def build_prompt(query: str, results: list, professors: list) -> str:

    # Single professor — flat review list with source labels
    if len(professors) == 1:
        if not results:
            evidence = "No relevant reviews found."
        else:
            evidence = "\n".join([
                f"Review {i+1} "
                f"[Source: {r['source']}, Position: {r['chunk_position']}]"
                f" (similarity: {r['score']}): {r['text']}"
                for i, r in enumerate(results)
            ])

    # Multi professor — grouped by professor with source labels
    else:
        if not results:
            evidence = "No relevant reviews found."
        else:
            evidence = ""
            for result in results:
                evidence += f"\n--- {result['professor_name']} "
                evidence += f"(avg relevance: {result['average_score']}) ---\n"
                for i, review in enumerate(result["top_reviews"]):
                    evidence += (
                        f"Review {i+1} "
                        f"[Source: {review['source']}, "
                        f"Position: {review['chunk_position']}]: "
                        f"{review['text']}\n"
                    )

    return f"""Using ONLY the student reviews provided below, answer the question.
Do not use any information outside of these reviews.
If the reviews do not contain enough information, say so explicitly.
At the end of your response, list every source document you drew from.

STUDENT REVIEWS:
{evidence}

QUESTION: {query}

ANSWER (based solely on the reviews above, cite sources at the end):"""


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────

def generate(
    query:       str,
    results:     list,
    professors:  list,
    groq_client: Groq
) -> str:
    """
    Sends the grounded prompt to Groq's llama-3.3-70b-versatile
    and returns the generated response.

    The system prompt strictly forbids the LLM from using
    outside knowledge — it must answer only from the reviews.
    """
    prompt = build_prompt(query, results, professors)

    response = groq_client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature = 0.2,   # low temperature = more factual, less creative
        max_tokens  = 1024
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def ask(
    query:       str,
    model:       SentenceTransformer,
    collection:  chromadb.Collection,
    groq_client: Groq,
    all_professors: list
) -> str:
    """
    Full RAG pipeline in one call:
        1. Detect which professors the query is about
        2. Retrieve relevant chunks from ChromaDB
        3. Build a grounded prompt
        4. Generate a response with Groq
        5. Return the response

    Args:
        query:          natural language question
        model:          loaded SentenceTransformer
        collection:     ChromaDB collection
        groq_client:    Groq API client
        all_professors: full professor list from the database

    Returns:
        LLM response string grounded in retrieved reviews
    """

    # Step 1 — detect professors
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

    # Step 2 — retrieve
    results = retrieve(query, professors, model, collection)

    # Step 3 & 4 — build prompt and generate
    response = generate(query, results, professors, groq_client)

    return response


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    model, collection, groq_client = load_clients()
    all_professors = get_all_professors(collection)

    print(f"Professors in database: {', '.join(all_professors)}")
    print("Type your question, or 'quit' to exit.\n")

    # Single query mode — passed as command line argument
    if len(sys.argv) > 1:
        query    = " ".join(sys.argv[1:])
        response = ask(query, model, collection, groq_client, all_professors)
        print(f"\nQ: {query}")
        print(f"\nA: {response}\n")
        return

    # Interactive mode
    while True:
        query = input("Question: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        if not query:
            continue

        response = ask(query, model, collection, groq_client, all_professors)
        print(f"\nA: {response}\n")


if __name__ == "__main__":
    main()
