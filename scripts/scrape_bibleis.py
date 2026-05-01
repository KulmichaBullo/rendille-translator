#!/usr/bin/env python3
"""
Robust Rendille Bible scraper — resumable, rate-limit resilient.
"""
import argparse, json, random, re, time, pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Accurate NT chapter counts
BOOK_CHAPTERS = {
    "MAT":28,"MRK":16,"LUK":24,"JHN":21,"ACT":28,
    "ROM":16,"1CO":16,"2CO":13,"GAL":6,"EPH":6,"PHP":4,"COL":4,
    "1TH":5,"2TH":3,"1TI":6,"2TI":4,"TIT":3,"PHM":1,
    "HEB":13,"JAS":5,"1PE":5,"2PE":3,"1JN":5,"2JN":1,"3JN":1,
    "JUD":1,"REV":22
}
NT_BOOKS = list(BOOK_CHAPTERS.keys())

BASE = "https://live.bible.is/bible"
REL_ID, ENG_ID = "RELBTL", "ENGWEB"
UA_POOL = [
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/119.0",
  "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0",
  "Mozilla/5.0 (compatible; Googlebot/2.1)",
  "Mozilla/5.0 (compatible; Bingbot/2.0)",
]

STATE_FILE = Path(".scrape_state.pkl")

# ── Helpers ────────────────────────────────────────────────────────────────

def save_state(state):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    return {"completed": set(), "failed": {}, "total":0}

def fetch_html(bid, book, chapter, retries=5):
    url = f"{BASE}/{bid}/{book}/{chapter}"
    for attempt in range(retries):
        try:
            import urllib.request
            hdr = {"User-Agent": random.choice(UA_POOL),
                   "Accept":"text/html,*/*;q=0.8",
                   "Referer":"https://live.bible.is/"}
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status in (403,429):
                    raise PermissionError(f"HTTP {r.status}")
                return r.read().decode("utf-8", errors="replace")
        except PermissionError as e:
            if attempt == retries-1: raise
            wait = (2**attempt) + random.uniform(0,3)
            print(f"  ⏳ {book} {chapter}: retry {attempt+1}/{retries} ({e}) — {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(2**attempt)
    return None

def extract_verses(html):
    """Return {verse_num: text} or None if empty/invalid."""
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html, re.DOTALL
    )
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        chapter_text = data.get("props",{}).get("pageProps",{}).get("chapterText",[])
    except (json.JSONDecodeError, AttributeError):
        return None
    verses = {}
    for v in chapter_text:
        num = str(v.get("verse_start","")).strip()
        text = v.get("verse_text","").strip()
        if num and text:
            verses[num] = text
    return verses if verses else None

# ── Scrape one chapter ─────────────────────────────────────────────────────

def scrape_chapter(bid, book, chapter, out_dir, state):
    key = f"{book}:{chapter}"
    if key in state["completed"]:
        return True, 0  # already done

    try:
        html = fetch_html(bid, book, chapter)
        if not html:
            raise ValueError("Empty response")
        verses = extract_verses(html)
        if not verses:
            raise ValueError("No verses parsed")
    except Exception as e:
        state["failed"][key] = str(e)
        save_state(state)
        return False, 0

    # Append verses
    rel_out, eng_out, vref_out = [out_dir / f for f in
        ("rel_extract.txt","eng_extract.txt","rel_vref.txt")]
    written = 0
    for vnum, rel_txt in verses.items():
        eng_txt = extract_verses(fetch_html(ENG_ID, book, chapter)) or {}
        eng_txt = eng_txt.get(vnum, "").strip()
        if eng_txt:
            ref = f"{book} {chapter}:{vnum}"
            vref_out.open("a").write(ref + "\n")
            rel_out.open("a").write(rel_txt + "\n")
            eng_out.open("a").write(eng_txt + "\n")
            written += 1

    state["completed"].add(key)
    state["total"] += written
    save_state(state)
    return True, written

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from previous state")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ["rel_extract.txt","eng_extract.txt","rel_vref.txt"]:
        (out_dir/f).touch(exist_ok=True)

    state = load_state() if args.resume else {"completed":set(),"failed":{}, "total":0}
    if not args.resume:
        # Fresh start — clear files
        for f in ["rel_extract.txt","eng_extract.txt","rel_vref.txt"]:
            (out_dir/f).write_text("", encoding="utf-8")
        state = {"completed":set(),"failed":{}, "total":0}

    print(f"📖 Rendille NT → {out_dir}/  (resume={args.resume})")
    print(f"   Already done: {len(state['completed'])} chapters  "
          f"failed: {len(state['failed'])}  total verses: {state['total']}")

    total_verses = state["total"]
    for book in NT_BOOKS:
        max_ch = BOOK_CHAPTERS[book]
        for ch in range(1, max_ch+1):
            key = f"{book}:{ch}"
            if key in state["completed"]:
                continue
            ok, n = scrape_chapter(REL_ID, book, ch, out_dir, state)
            if ok:
                total_verses += n
                if ch % 5 == 0:
                    print(f"  {book} → ch {ch}  total:{total_verses}")
            else:
                print(f"  ✗ {book} {ch}: {state['failed'][key]}")

    # Duplicate vref for English
    (out_dir/"eng_vref.txt").write_text(
        (out_dir/"rel_vref.txt").read_text(encoding="utf-8"),
        encoding="utf-8"
    )

    # Summary
    print(f"\n✅ Total verses: {total_verses}")
    for fn in ["rel_extract.txt","eng_extract.txt","rel_vref.txt","eng_vref.txt"]:
        p = out_dir/fn
        n = len(p.read_text(encoding="utf-8").splitlines()) if p.exists() else 0
        print(f"   {fn}: {n} lines")

    print("\n👉 Next: run Cell [6] prepare_corpus.py")

if __name__ == "__main__":
    main()
