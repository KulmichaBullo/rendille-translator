#!/usr/bin/env python3
"""
Download Rendille–English Bible corpus.

Primary: HuggingFace datasets with trust_remote_code=True
Fallback: Direct requests to ebible.org (if HF Hub blocked)
"""
from pathlib import Path
import sys

DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)
ISO = "rel"

print("="*60)
print("📥 DOWNLOADING RENDILLE–ENGLISH BIBLE CORPUS")
print("="*60)

# METHOD 1: HuggingFace datasets (preferred — clean, streaming)
print("\n🔽 Method 1: HuggingFace datasets")
try:
    from datasets import load_dataset
    print("  Loading bible-nlp/biblenlp-corpus (streaming)...")
    ds = load_dataset(
        "bible-nlp/biblenlp-corpus",
        split="train",
        streaming=True,
        trust_remote_code=True  # Required for custom dataset script
    )
    
    rendille = []
   english = []
    refs = []
    
    count = 0
    for ex in ds:
        langs = ex['translation']['languages']
        texts = ex['translation']['translation']
        ref_list = ex.get('ref', [])
        if isinstance(ref_list, str):
            ref_list = [ref_list]
        
        if ISO in langs:
            rendille.append(texts[langs.index(ISO)])
            if 'eng' in langs:
                english.append(texts[langs.index('eng')])
            else:
                english.append("")
            refs.extend(ref_list)
            count += 1
            
            if count % 1000 == 0:
                print(f"    ... {count} verses collected")
            # Remove limit for full Bible
            # if count >= 27000: break
    
    if rendille:
        (DATA / f"{ISO}_extract.txt").write_text("\n".join(rendille), encoding='utf-8')
        (DATA / "eng_extract.txt").write_text("\n".join(english), encoding='utf-8')
        (DATA / f"{ISO}_vref.txt").write_text("\n".join(refs), encoding='utf-8')
        print(f"  ✓ Got {len(rendille):,} verses via HuggingFace")
        print(f"  ✓ Files: {ISO}_extract.txt, eng_extract.txt, {ISO}_vref.txt")
        sys.exit(0)
    else:
        print("  ✗ No Rendille verses found")
        
except Exception as e:
    print(f"  ✗ HuggingFace failed: {e}")

# METHOD 2: Direct download from ebible.org ZIP + extract
print("\n🔽 Method 2: Download & extract from eBible.org")
try:
    import requests, zipfile, io
    
    # First, get translations.csv to find Rendille ID
    print("  Fetching translations list...")
    csv_url = "https://ebible.org/scriptures/translations.csv"
    r = requests.get(csv_url, timeout=30)
    if not r.ok:
        raise Exception(f"translations.csv failed: {r.status_code}")
    
    import csv
    rows = list(csv.DictReader(r.text.splitlines()))
    rel_row = [row for row in rows if row.get('iso', '').lower() == ISO]
    
    if not rel_row:
        print(f"  ✗ Rendille (rel) not in translations.csv")
        raise Exception("Rendille not listed")
    
    translation_id = rel_row[0]['id']  # e.g., 'rel' or 'rel_xxx'
    print(f"  Found Rendille translation ID: {translation_id}")
    
    # Download ZIP
    zip_url = f"https://ebible.org/Download/{translation_id}.zip"
    print(f"  Downloading {zip_url}...")
    rz = requests.get(zip_url, timeout=60, stream=True)
    if not rz.ok:
        raise Exception(f"ZIP download failed: {rz.status_code}")
    
    z = zipfile.ZipFile(io.BytesIO(rz.content))
    
    # Find USFM files (usually .usfm or .sfm)
    usfm_files = [f for f in z.namelist() if f.lower().endswith(('.usfm', '.sfm'))]
    print(f"  ZIP contains {len(usfm_files)} USFM files")
    
    # Extract verses (simple extraction — concatenate USFM text)
    # This is a simplified version; full extraction needs SILNLP
    verses = []
    for usfm in sorted(usfm_files)[:5]:  # Limit for demo
        content = z.read(usfm).decode('utf-8', errors='replace')
        # Very crude: split on newlines, take first line as verse reference
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('\'):
                verses.append(line)
                break
    
    if verses:
        (DATA / f"{ISO}_extract.txt").write_text("\n".join(verses), encoding='utf-8')
        (DATA / f"{ISO}_vref.txt").write_text("\n".join([f"MAT 1:1"] * len(verses)), encoding='utf-8')  # dummy refs
        print(f"  ✓ Extracted {len(verses)} sample verses (placeholder)")
        print("  ⚠ Full extraction requires SIL NLP toolkit")
        sys.exit(0)
    else:
        print("  ✗ No verses extracted")
        
except Exception as e:
    print(f"  ✗ eBible.org method failed: {e}")

# METHOD 3: Fallback to previously working direct URLs (if any still alive)
print("\n🔽 Method 3: Direct file mirrors (unlikely to work)")
print("  All methods failed. Please download manually.")
print("\n❌ Download failed — see manual instructions below.")
print("\n💡 Manual download:")
print("  1. Go to https://ebible.org/")
print("  2. Search for 'Rendille'")
print("  3. Download the translation ZIP")
print("  4. Extract and place .txt files in data/ as:")
print("     - rel_vref.txt (verse references)")
print("     - rel_extract.txt (Rendille text)")
print("  5. Also get English: download 'eng' (World English Bible)")
print("     and place as eng_extract.txt")
sys.exit(1)
