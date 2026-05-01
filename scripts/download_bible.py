#!/usr/bin/env python3
"""
Simple, robust Bible download for Colab.
Uses only stdlib + requests if available.
"""
from pathlib import Path
import urllib.request
import time

DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)
ISO = "rel"

def fetch_url(url, outpath, retries=3):
    """Download with retries and browser UA."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req, timeout=30) as r:
                outpath.write_bytes(r.read())
                return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                print(f"  ✗ {url}: {e}")
                return False

print("="*60)
print("📥 Downloading Rendille–English Bible Corpus")
print("="*60)

# Direct URLs on ebible.org CDN
base = "https://ebible.org/Download/data"
files = {
    f"{base}/{ISO}/vref.txt": DATA / f"{ISO}_vref.txt",
    f"{base}/{ISO}/extract.txt": DATA / f"{ISO}_extract.txt",
    f"{base}/eng/vref.txt": DATA / "eng_vref.txt",
    f"{base}/eng/extract.txt": DATA / "eng_extract.txt",
}

success = 0
for url, out in files.items():
    print(f"  Downloading {out.name}...")
    if fetch_url(url, out):
        lines = out.read_text().splitlines()
        print(f"    ✓ {len(lines):,} lines")
        success += 1
    else:
        print(f"    ✗ Failed")

if success == 4:
    print("\n✅ All files downloaded!")
else:
    print(f"\n⚠ Only {success}/4 files downloaded")
    print("\n💡 Alternative: manually download from https://ebible.org/")
    print("   Search 'Rendille' and place files in data/ as:")
    print("   - rel_vref.txt, rel_extract.txt")
    print("   - eng_extract.txt (English)")
