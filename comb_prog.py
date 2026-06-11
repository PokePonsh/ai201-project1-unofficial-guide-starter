"""
Combined Pipeline — Q&A Only
------------------------------
Self-contained RAG pipeline that reads from an existing ChromaDB.
Designed to be imported by app.py or run standalone.

Does NOT scrape or embed — run rmp_scraper.py and embed_and_store.py first.

Usage:
    # Standalone interactive mode
    python comb_prog.py

    # Standalone single question
    python comb_prog.py "Who is the easiest grader?"

    # Imported by app.py
    from comb_prog import ask, ALL_PROFESSORS

Dependencies:
    pip install sentence-transformers chromadb rapidfuzz groq python-dotenv
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
# SYSTEM PROMPT
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
# STARTUP — load once, reuse for every query
# ─────────────────────────────────────────────

def _load_clients():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Connecting to ChromaDB at ./{DB_DIR}")
    client     = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name     = COLLECTION,
        metadata = {"hnsw:space": "cosine"}
    )
    print(f"Collection '{COLLECTION}' — {collection.count()} chunks loaded")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    groq_client = Groq(api_key=api_key)
    print(f"Groq client ready — model: {GROQ_MODEL}\n")

    return model, collection, groq_client


def _get_all_professors(collection: chromadb.Collection) -> list:
    results    = collection.get(include=["metadatas"])
    professors = list({m["professor_name"] for m in results["metadatas"]})
    return sorted(professors)


# Load everything once at module import time
print("Initializing pipeline...")
_MODEL, _COLLECTION, _GROQ_CLIENT = _load_clients()
ALL_PROFESSORS = _get_all_professors(_COLLECTION)
print(f"Professors in database: {', '.join(ALL_PROFESSORS)}\n")


# ─────────────────────────────────────────────
# NAME DETECTION
# ─────────────────────────────────────────────

def detect_professors(query: str) -> list:
    """
    Scans the query for professor names using fuzzy matching.

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
    for professor in ALL_PROFESSORS:
        for part in professor.lower().split():
            if len(part) <= 3 or part in skip_tokens:
                continue
            if fuzz.partial_ratio(part, query_lower) >= FUZZY_CUTOFF:
                matched.append(professor)
                break
    return matched


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def _retrieve(query: str, professors: list) -> list:
    query_embedding = _MODEL.encode(query).tolist()
    if len(professors) == 1:
        return _query_single(query_embedding, professors[0])
    return _query_multi(query_embedding, professors)


def _query_single(query_embedding: list, professor: str) -> list:
    results = _COLLECTION.query(
        query_embeddings = [query_embedding],
        n_results        = K_SINGLE,
        where            = {"professor_name": professor},
        include          = ["documents", "metadatas", "distances"]
    )
    all_results = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = round(1 - distance, 4)
        all_results.append({
            "professor_name": meta["professor_name"],
            "score":          score,
            "chunk_position": meta["chunk_position"],
            "review_number":  meta["review_number"],
            "source":         meta["source"],
            "text":           doc
        })
    above_cutoff = [r for r in all_results if r["score"] >= SCORE_CUTOFF]
    return above_cutoff if above_cutoff else all_results[:3]


def _query_multi(query_embedding: list, professors: list) -> list:
    professor_scores = {}
    professor_chunks = {}
    for professor in professors:
        try:
            results = _COLLECTION.query(
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
        similarities             = [round(1 - d, 4) for d in distances]
        filtered                 = list(zip(docs, metadatas, similarities))
        avg_score                = round(sum(s for _, _, s in filtered) / len(filtered), 4)
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
                    "review_number":  meta["review_number"],
                    "source":         meta["source"]
                }
                for doc, meta, score in chunks[:3]
            ]
        })
    return output


# ─────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────

def _build_prompt(query: str, results: list, professors: list) -> str:
    if len(professors) == 1:
        evidence = "\n".join([
            f"[Source: {r['source']}, Review #{r['review_number']}]: {r['text']}"
            for r in results
        ]) if results else "No relevant reviews found."
        citation_format = "cite inline as (Review #[num])"
    else:
        if not results:
            evidence = "No relevant reviews found."
        else:
            evidence = ""
            for result in results:
                evidence += f"\n--- {result['professor_name']} "
                evidence += f"(avg relevance: {result['average_score']}) ---\n"
                for review in result["top_reviews"]:
                    evidence += (
                        f"[Source: {review['source']}, "
                        f"Review #{review['review_number']}]: "
                        f"{review['text']}\n"
                    )
        citation_format = "cite inline as ([professor last name], Review #[num])"

    return f"""Using ONLY the student reviews provided below, answer the question.
Do not use any information outside of these reviews.
If the reviews do not contain enough information, say so explicitly.
When referencing a review inline, {citation_format}.
At the end of your response, list every source document you drew from.

STUDENT REVIEWS:
{evidence}

QUESTION: {query}

ANSWER (based solely on the reviews above, cite sources at the end):"""


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def ask(question: str) -> dict:
    """
    End-to-end RAG pipeline. The only public function app.py needs.

    Args:
        question: natural language question from the user

    Returns:
        dict with keys:
            "answer"   -- LLM response string
            "sources"  -- list of source strings, one per source
            "detected" -- which professors were detected
    """
    if not question.strip():
        return {
            "answer":   "Please enter a question.",
            "sources":  [],
            "detected": ""
        }

    # Detect professors
    matched = detect_professors(question)

    if len(matched) == 0:
        professors = ALL_PROFESSORS
        detected   = "Comparing all professors"
    elif len(matched) == 1:
        professors = matched
        detected   = f"Single professor: {matched[0]}"
    else:
        professors = matched
        detected   = f"Comparing: {', '.join(matched)}"

    # Retrieve and generate
    results       = _retrieve(question, professors)
    prompt        = _build_prompt(question, results, professors)
    response      = _GROQ_CLIENT.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature = 0.2,
        max_tokens  = 1024
    )
    full_response = response.choices[0].message.content

    # Split answer and sources
    if "Sources:" in full_response:
        parts        = full_response.split("Sources:", 1)
        answer       = parts[0].strip()
        sources_list = [
            line.strip()
            for line in parts[1].strip().splitlines()
            if line.strip()
        ]
    else:
        answer       = full_response.strip()
        sources_list = []

    return {
        "answer":   answer,
        "sources":  sources_list,
        "detected": detected
    }
# ─────────────────────────────────────────────
# MAIN — standalone mode
# ─────────────────────────────────────────────

def main():
    print(f"Professors in database: {', '.join(ALL_PROFESSORS)}")
    print("Type your question, or 'quit' to exit.\n")

    # Single question mode
    if len(sys.argv) > 1:
        question         = " ".join(sys.argv[1:])
        result = ask(question)
        print(f"\nDetected: {detected}")
        print(f"\nA {result['answer']}")
        print("\nSources:")
        for s in result['sources']:
            print(f"  • {s}")
        return

    # Interactive mode
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break
        if not question:
            continue
        result = ask(question)
        print(f"\nDetected: {result['detected']}")
        print(f"\nA: {result['answer']}")
        print("\nSources:")
        for s in result['sources']:
            print(f"  • {s}")
        print() 


if __name__ == "__main__":
    main()
