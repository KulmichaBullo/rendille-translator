#!/usr/bin/env python3
"""
Scrape Rendille Bible text from Bible.is (live.bible.is)
Uses Next.js __NEXT_DATA__ embedded JSON.
Produces: data/rel_vref.txt, data/rel_extract.txt, data/eng_vref.txt, data/eng_extract.txt
"""
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

NT_BOOKS = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH", "PHP", "COL",
    "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN",
    "3JN", "JUD", "REV"
]

RENDILLE_BIBLE_ID = "RELBTL"
ENGLISH_BIBLE_ID = "ENGWEB"
BASE_URL = "https://live.bible.is/bible"

# ── Fetch and parse chapter text ────────────────────────────────────────────

def fetch_chapter_json(bible_id: str, book: str, chapter: int) -> List[Dict]:
    """Download page, extract __NEXT_DATA__ → chapterText array."""
    url = f"{BASE_URL}/{bible_id}/{book}/{chapter}"
    for attempt in range(3):
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RendilleMT/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    # Extract __NEXT_DATA__ JSON
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
    if not m:
        raise ValueError(f"No __NEXT_DATA__ found for {book} {chapter}")
    data = json.loads(m.group(1))
    chapter_text = data.get("props", {}) \
                      .get("pageProps", {}) \
                      .get("chapterText", [])
    if not chapter_text:
        raise ValueError(f"chapterText empty for {book} {chapter}")
    return chapter_text

def scrape_book(book: str, output_dir: Path) -> Tuple[int, int]:
    rel_out = output_dir / "rel_extract.txt"
    eng_out = output_dir / "eng_extract.txt"
    vref_out = output_dir / "rel_vref.txt"

    verse_count = 0
    errors = 0

    for chapter in range(1, 200):
        try:
            rel_chapter = fetch_chapter_json(RENDILLE_BIBLE_ID, book, chapter)
            eng_chapter = fetch_chapter_json(ENGLISH_BIBLE_ID, book, chapter)
        except Exception as e:
            if chapter == 1:
                print(f"  ✗ {book} ch {chapter}: {e}")
                return verse_count, errors
            break  # end of book

        # Build verse lookup for English (verse_start may be a range, usually single)
        eng_map = {}
        for v in eng_chapter:
            vs = v.get("verse_start")
            vt = v.get("verse_text", "").strip()
            if vt:
                eng_map[str(vs)] = vt

        # Write Rendille + English in verse order
        for v in rel_chapter:
            v_num = str(v.get("verse_start"))
            rel_text = v.get("verse_text", "").strip()
            eng_text = eng_map.get(v_num, "").strip()
            if rel_text and eng_text:
                ref = f"{book} {chapter}:{v_num}"
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
            print(f"  {book} up to ch {chapter} → {verse_count} verses")

        time.sleep(0.2)  # polite

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

    print(f" Scraping Rendille NT → {out_dir}/")
    print(f" Books: {', '.join(NT_BOOKS)}")

    total = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(scrape_book, bk, out_dir): bk for bk in NT_BOOKS}
        for fut in as_completed(futures):
            book = futures[fut]
            try:
                v, e = fut.result()
                total += v
            except Exception as exc:
                print(f" ✗ {book} failed: {exc}")

    print(f"\n✓ Total: {total} verses")

    # Copy vref for English
    (out_dir / "eng_vref.txt").write_text(
        (out_dir / "rel_vref.txt").read_text(),
        encoding="utf-8"
    )

    # Summary
    for f in ["rel_extract.txt", "eng_extract.txt", "rel_vref.txt", "eng_vref.txt"]:
        p = out_dir / f
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
            print(f"   {p.name}: {len(lines)} lines")

    print("\n Done! Next: run prepare_corpus.py")

if __name__ == "__main__":
    main()
