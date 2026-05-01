#!/usr/bin/env python3
"""
Scrape Rendille Bible text from Bible.is (live.bible.is)
Uses Next.js __NEXT_DATA__ embedded JSON.
Produces: data/rel_vref.txt, data/rel_extract.txt, data/eng_vref.txt, data/eng_extract.txt
"""
import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# ── Rendille New Testament books ───────────────────────────────────────────
NT_BOOKS = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH", "PHP", "COL",
    "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN",
    "3JN", "JUD", "REV"
]

RENDILLE_BIBLE_ID = "RELBTL"
ENGLISH_BIBLE_ID = "ENGWEB"
BASE_URL = "https://live.bible.is/bible"

# ── HTTP fetch with rotation + backoff ─────────────────────────────────────

def fetch_html(bible_id: str, book: str, chapter: int) -> str:
    """Download chapter page, handling 403/429 with backoff and UA rotation."""
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
            print(f"  Retry {attempt+1}/5 for {book} {chapter} after {e} ({wait:.1f}s)")
            time.sleep(wait)
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

# ── Parse __NEXT_DATA__ JSON ───────────────────────────────────────────────

def extract_verses(html: str, book: str, chapter: int) -> Dict[str, str]:
    """Parse Next.js __NEXT_DATA__ → {verse_num: verse_text}."""
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html, re.DOTALL
    )
    if not m:
        raise ValueError(f"No __NEXT_DATA__ for {book} {chapter}")

    data = json.loads(m.group(1))
    chapter_text = (
        data.get("props", {})
           .get("pageProps", {})
           .get("chapterText", [])
    )

    verses: Dict[str, str] = {}
    for v in chapter_text:
        num = str(v.get("verse_start", ""))
        text = v.get("verse_text", "").strip()
        if num and text:
            verses[num] = text

    if not verses:
        raise ValueError(f"Empty verses for {book} {chapter}")

    return verses

# ── Scrape single book ──────────────────────────────────────────────────────

def scrape_book(book: str, output_dir: Path) -> Tuple[int, int]:
    rel_out = output_dir / "rel_extract.txt"
    eng_out = output_dir / "eng_extract.txt"
    vref_out = output_dir / "rel_vref.txt"

    verse_count = 0
    errors = 0

    for chapter in range(1, 200):
        try:
            html_rel = fetch_html(RENDILLE_BIBLE_ID, book, chapter)
            html_eng = fetch_html(ENGLISH_BIBLE_ID, book, chapter)
        except Exception as e:
            if chapter == 1:
                print(f"  ✗ {book} ch {chapter}: {e}")
                return verse_count, errors
            break  # Reached end of book

        try:
            rel_verses = extract_verses(html_rel, book, chapter)
            eng_verses = extract_verses(html_eng, book, chapter)
        except Exception as e:
            print(f"  ⚠ Parse fail {book} {chapter}: {e}")
            errors += 1
            continue

        # Write aligned verses
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
            else:
                errors += 1

        if chapter % 5 == 0:
            print(f"  {book} → ch {chapter} (total: {verse_count})")

        time.sleep(0.2)  # polite delay

    print(f"✓ {book}: {verse_count} verses, {errors} skipped")
    return verse_count, errors

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape Rendille Bible from Bible.is")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear targets
    for f in ["rel_extract.txt", "eng_extract.txt", "rel_vref.txt"]:
        (out_dir / f).write_text("", encoding="utf-8")

    print(f" Scraping Rendille NT → {out_dir}/ ({len(NT_BOOKS)} books)")

    total = 0
    failed: List[Tuple[str, str]] = []  # (book, error)

    # Retry loop — if any book fails initially, retry once after delay
    for attempt in range(2):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(scrape_book, bk, out_dir): bk for bk in NT_BOOKS}
            for fut in as_completed(futures):
                book = futures[fut]
                try:
                    v, e = fut.result()
                    total += v
                except Exception as exc:
                    failed.append((book, str(exc)))
                    print(f" ✗ {book} failed: {exc}")

        if not failed or attempt > 0:
            break
        print(f"\n Retrying {len(failed)} failed books after 10s delay...")
        time.sleep(10)
        NT_BOOKS_RETRY = [b for b, _ in failed]
        NT_BOOKS = NT_BOOKS_RETRY  # only retry failures

    if failed:
        print(f"\n Failed after retry: {failed}")

    print(f"\n✓ Total verses: {total}")

    # Generate English vref (same order)
    (out_dir / "eng_vref.txt").write_text(
        (out_dir / "rel_vref.txt").read_text(encoding="utf-8"),
        encoding="utf-8"
    )

    # Summary
    for f in ["rel_extract.txt", "eng_extract.txt", "rel_vref.txt", "eng_vref.txt"]:
        p = out_dir / f
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
            print(f"   {p.name}: {len(lines)} lines")

    print("\n Done! Next: run Cell [6] prepare_corpus.py")

if __name__ == "__main__":
    main()
