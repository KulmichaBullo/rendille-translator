#!/usr/bin/env python3
"""
Scrape Rendille Bible text from Bible.is (live.bible.is)
Accurate, rate-limit-aware scraper using __NEXT_DATA__ JSON.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

# ── New Testament with accurate chapter counts ────────────────────────────
BOOK_CHAPTERS = {
    "MAT": 28, "MRK": 16, "LUK": 24, "JHN": 21,
    "ACT": 28,
    "ROM": 16, "1CO": 16, "2CO": 13, "GAL": 6, "EPH": 6, "PHP": 4, "COL": 4,
    "1TH": 5, "2TH": 3, "1TI": 6, "2TI": 4, "TIT": 3, "PHM": 1,
    "HEB": 13, "JAS": 5, "1PE": 5, "2PE": 3, "1JN": 5, "2JN": 1, "3JN": 1,
    "JUD": 1, "REV": 22
}
NT_BOOKS = list(BOOK_CHAPTERS.keys())

RENDILLE_BIBLE_ID = "RELBTL"
ENGLISH_BIBLE_ID = "ENGWEB"
BASE_URL = "https://live.bible.is/bible"

# ── Fetch with retry + UA rotation ────────────────────────────────────────

def fetch_html(bible_id: str, book: str, chapter: int) -> str:
    url = f"{BASE_URL}/{bible_id}/{book}/{chapter}"
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Chrome/120.0.0.0)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (Chrome/119.0.0.0)",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (Chrome/120.0.0.0)",
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bot.html)",
    ]
    for attempt in range(5):
        try:
            import urllib.request
            headers = {
                "User-Agent": random.choice(user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://live.bible.is/",
                "DNT": "1",
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status in (403, 429):
                    raise PermissionError(f"HTTP {r.status}")
                return r.read().decode("utf-8", errors="replace")
        except PermissionError as e:
            if attempt == 4:
                raise
            wait = (2 ** attempt) + random.uniform(0, 2)
            print(f"  Retry {attempt+1}/5 for {book} {chapter} ({e}) — {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

# ── Parse Next.js JSON ──────────────────────────────────────────────────────

def extract_verses(html: str) -> dict:
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html, re.DOTALL
    )
    if not m:
        raise ValueError("No __NEXT_DATA__")
    data = json.loads(m.group(1))
    chapter_text = (
        data.get("props", {})
           .get("pageProps", {})
           .get("chapterText", [])
    )
    verses = {}
    for v in chapter_text:
        num = str(v.get("verse_start", ""))
        text = v.get("verse_text", "").strip()
        if num and text:
            verses[num] = text
    if not verses:
        raise ValueError("No verses found")
    return verses

# ── Scrape one book ────────────────────────────────────────────────────────

def scrape_book(book: str, output_dir: Path) -> int:
    rel_out = output_dir / "rel_extract.txt"
    eng_out = output_dir / "eng_extract.txt"
    vref_out = output_dir / "rel_vref.txt"

    verse_count = 0
    max_chapter = BOOK_CHAPTERS[book]

    for chapter in range(1, max_chapter + 1):
        try:
            html_rel = fetch_html(RENDILLE_BIBLE_ID, book, chapter)
            html_eng = fetch_html(ENGLISH_BIBLE_ID, book, chapter)
        except Exception as e:
            print(f"  ✗ {book} {chapter}: {e}")
            break

        try:
            rel_verses = extract_verses(html_rel)
            eng_verses = extract_verses(html_eng)
        except Exception as e:
            print(f"  ⚠ Parse fail {book} {chapter}: {e}")
            continue

        for vnum, rel_text in rel_verses.items():
            eng_text = eng_verses.get(vnum, "").strip()
            if eng_text:
                ref = f"{book} {chapter}:{vnum}"
                with open(vref_out, "a", encoding="utf-8") as vf:
                    vf.write(ref + "\n")
                with open(rel_out, "a", encoding="utf-8") as rf:
                    rf.write(rel_text + "\n")
                with open(eng_out, "a", encoding="utf-8") as ef:
                    ef.write(eng_text + "\n")
                verse_count += 1

        if chapter % 5 == 0:
            print(f"  {book} → ch {chapter} ({verse_count} verses)")

        time.sleep(0.2)

    print(f"✓ {book}: {verse_count} verses")
    return verse_count

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape Rendille Bible from Bible.is")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear files at start (fresh run)
    for f in ["rel_extract.txt", "eng_extract.txt", "rel_vref.txt"]:
        (out_dir / f).write_text("", encoding="utf-8")

    print(f" Scraping Rendille NT → {out_dir}/ ({len(NT_BOOKS)} books)")

    total = 0
    for book in NT_BOOKS:
        try:
            v = scrape_book(book, out_dir)
            total += v
        except Exception as e:
            print(f" ✗ {book} failed: {e}")

    print(f"\n✓ Total: {total} verses")

    (out_dir / "eng_vref.txt").write_text(
        (out_dir / "rel_vref.txt").read_text(encoding="utf-8"),
        encoding="utf-8"
    )

    for f in ["rel_extract.txt", "eng_extract.txt", "rel_vref.txt", "eng_vref.txt"]:
        p = out_dir / f
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
            print(f"   {p.name}: {len(lines)} lines")

    print("\n Done! Next: run Cell [6] prepare_corpus.py")

if __name__ == "__main__":
    main()
