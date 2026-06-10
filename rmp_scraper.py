"""
Rate My Professor Scraper
--------------------------
Takes a RMP professor URL, extracts the professor's name
and all review text, cleans it, chunks it (1 review = 1 chunk),
and saves both raw and chunked output to JSON.

Usage:
    python rmp_scraper.py <url>

Example:
    python rmp_scraper.py "https://www.ratemyprofessors.com/professor/12345"
"""

import sys
import time
import json
import re
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────
# STEP 1 — SCRAPE
# ─────────────────────────────────────────────

def scrape_rmp(url: str) -> dict:
    """
    Loads the RMP professor page via a real browser,
    clicks through all pagination, and returns the raw
    professor name and review texts.

    Args:
        url: Full RMP professor URL

    Returns:
        {
            "professor_name": str,
            "reviews": [str, str, ...]
        }
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Mimic a real browser to avoid bot detection
        page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

        print(f"Loading page: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(7)  # wait for React to render reviews into the DOM

        # Click "Load More Ratings" until all reviews are visible
        while True:
            try:
                load_more = page.locator("button", has_text="Load More Ratings")
                if load_more.count() > 0 and load_more.first.is_visible():
                    load_more.first.click()
                    time.sleep(1.5)
                else:
                    break
            except Exception:
                break

        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")

    # ── Professor Name ──
    professor_name = "Unknown"
    name_tag = soup.find("div", class_=lambda c: c and "NameTitle__Name" in c)
    if name_tag:
        first = name_tag.find("span", class_=lambda c: c and "NameTitle__FirstNameWrapper" in c)
        last  = name_tag.find("span", class_=lambda c: c and "NameTitle__LastNameWrapper" in c)

        if first and last:
            professor_name = first.get_text(strip=True) + " " + last.get_text(strip=True)
        elif name_tag:
            # Fallback: split "JohnSmith" → "John Smith" via regex
            raw = name_tag.get_text(strip=True)
            professor_name = " ".join(re.findall('[A-Z][a-z]+', raw))

    # ── Raw Review Text ──
    reviews = []
    comment_divs = soup.find_all(
        "div", class_=lambda c: c and "Comments__StyledComments" in c
    )
    for div in comment_divs:
        text = div.get_text(strip=True)
        if text:
            reviews.append(text)

    return {
        "professor_name": professor_name,
        "reviews": reviews
    }


# ─────────────────────────────────────────────
# STEP 2 — CLEAN
# ─────────────────────────────────────────────

def clean_review(text: str) -> str:
    """
    Cleans a single review string by removing HTML entities,
    collapsing whitespace, and stripping non-content artifacts.

    Removes:  &amp; &nbsp; &#39; &quot; extra spaces/newlines
    Keeps:    actual review text as written by the student
    """
    text = re.sub(r'&amp;',  '&',  text)
    text = re.sub(r'&nbsp;', ' ',  text)
    text = re.sub(r'&#39;',  "'",  text)
    text = re.sub(r'&quot;', '"',  text)
    text = re.sub(r'&lt;',   '<',  text)
    text = re.sub(r'&gt;',   '>',  text)
    text = re.sub(r'\s+',    ' ',  text)   # collapse all whitespace to single space
    return text.strip()


# ─────────────────────────────────────────────
# STEP 3 — CHUNK
# ─────────────────────────────────────────────

def chunk_reviews(reviews: list, professor_name: str) -> list:
    """
    Converts cleaned reviews into chunks.

    Chunking strategy:
        - Chunk size:  1 review = 1 chunk
        - Overlap:     0 (each review is a self-contained opinion;
                         no sentence depends on the review before it)

    Each chunk is a dict containing the text and its metadata.

    Args:
        reviews:        list of raw review strings
        professor_name: name of the professor these reviews belong to

    Returns:
        list of chunk dicts
    """
    chunks = []
    for i, review in enumerate(reviews):
        cleaned = clean_review(review)

        # Skip empty reviews after cleaning
        if not cleaned:
            continue

        chunks.append({
            "chunk_id":        i,
            "professor_name":  professor_name,
            "text":            cleaned,
            "token_estimate":  len(cleaned.split())   # rough word count
        })

    return chunks


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python rmp_scraper.py <url1> <url2> ...")
        print("       python rmp_scraper.py urls.txt")
        sys.exit(1)

    # Load URLs from text file or command line arguments
    if sys.argv[1].endswith(".txt"):
        with open(sys.argv[1], "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        urls = sys.argv[1:]

    all_chunks = []

    for url in urls:

        # ── Step 1: Scrape ──
        result = scrape_rmp(url)
        professor_name = result["professor_name"]
        reviews        = result["reviews"]

        print(f"\nProfessor:        {professor_name}")
        print(f"Reviews scraped:  {len(reviews)}")

        # ── Step 2: Save raw output before any cleaning ──
        raw_output = {
            "professor_name": professor_name,
            "reviews": reviews
        }
        with open(f"raw_{professor_name.replace(' ', '_')}.json", "w", encoding="utf-8") as f:
            json.dump(raw_output, f, indent=2, ensure_ascii=False)
        print(f"Raw reviews saved to raw_{professor_name.replace(' ', '_')}.json")

        # ── Step 3: Chunk ──
        chunks = chunk_reviews(reviews, professor_name)
        all_chunks.extend(chunks)
        print(f"Chunks created:   {len(chunks)}")

    # ── Re-number chunk IDs across all professors ──
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    print(f"\nTotal chunks across all professors: {len(all_chunks)}")

    # ── Step 4: Inspect 5 representative chunks ──
    print("\n" + "=" * 60)
    print("5 REPRESENTATIVE CHUNKS")
    print("=" * 60)
    for chunk in all_chunks[:5]:
        print(f"\nChunk {chunk['chunk_id']}  (~{chunk['token_estimate']} tokens)")
        print(f"Professor: {chunk['professor_name']}")
        print(chunk["text"])
        print("-" * 40)

    # ── Step 5: Save all chunks ──
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print("\nAll chunks saved to chunks.json")

    return all_chunks


if __name__ == "__main__":
    main()