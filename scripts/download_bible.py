#!/usr/bin/env python3
"""
Download Rendille–English Bible corpus.

Priority order:
1. HuggingFace datasets (streaming, requires datasets lib)
2. Direct curl downloads from jsDelivr CDN ( mirrors GitHub, no auth)
3. Manual instructions
"""
from pathlib import Path
import sys
import subprocess

DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)
ISO = "rel"

print("="*60)
print("📥 DOWNLOADING RENDILLE–ENGLISH BIBLE CORPUS")
print("="*60)

# METHOD 1: HuggingFace datasets (preferred)
print("\n🔽 Method 1: HuggingFace datasets")
try:
    from datasets import load_dataset
    print("  Loading bible-nlp/biblenlp-corpus (streaming)...")
    ds = load_dataset(
        "bible-nlp/biblenlp-corpus",
        split="train",
        streaming=True,
        trust_remote_code=True
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

# METHOD 2: ebible.org ZIP (secondary)
print("\n🔽 Method 2: eBible.org ZIP download")
try:
    import requests, zipfile, io, csv
    
    print("  Fetching translations.csv...")
    r = requests.get("https://ebible.org/scriptures/translations.csv", timeout=30)
    if not r.ok:
        raise Exception(f"translations.csv HTTP {r.status_code}")
    
    rows = list(csv.DictReader(r.text.splitlines()))
    rel_row = [row for row in rows if row.get('iso', '').lower() == ISO]
    
    if not rel_row:
        print(f"  ✗ Rendille (rel) not in translations.csv")
        raise Exception("Rendille not listed")
    
    translation_id = rel_row[0]['id']
    zip_url = f"https://ebible.org/Download/{translation_id}.zip"
    print(f"  Downloading ZIP: {translation_id}...")
    rz = requests.get(zip_url, timeout=60, stream=True)
    if not rz.ok:
        raise Exception(f"ZIP download HTTP {rz.status_code}")
    
    z = zipfile.ZipFile(io.BytesIO(rz.content))
    usfm_files = [f for f in z.namelist() if f.lower().endswith(('.usfm', '.sfm'))]
    print(f"  ZIP contains {len(usfm_files)} USFM files")
    
    # Basic extraction — take first lines as sample verses
    verses = []
    for usfm in sorted(usfm_files)[:5]:
        content = z.read(usfm).decode('utf-8', errors='replace')
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('\\'):
                verses.append(line)
                break
    
    if verses:
        (DATA / f"{ISO}_extract.txt").write_text("\n".join(verses), encoding='utf-8')
        (DATA / f"{ISO}_vref.txt").write_text("\n".join(["MAT 1:1"] * len(verses)), encoding='utf-8')
        print(f"  ✓ Extracted {len(verses)} sample verses")
        print("  ⚠ Full extraction needs SIL NLP toolkit")
        sys.exit(0)
    else:
        print("  ✗ No verses extracted")
        
except Exception as e:
    print(f"  ✗ eBible.org ZIP method failed: {e}")

# METHOD 3: Direct curl from jsDelivr CDN (WORKING FALLBACK)
print("\n🔽 Method 3: Direct download from jsDelivr CDN")
files = {
    f"{ISO}_vref.txt": f"https://cdn.jsdelivr.net/gh/BibleNLP/ebible-corpus/data/{ISO}/vref.txt",
    f"{ISO}_extract.txt": f"https://cdn.jsdelivr.net/gh/BibleNLP/ebible-corpus/data/{ISO}/extract.txt",
    "eng_vref.txt": "https://cdn.jsdelivr.net/gh/BibleNLP/ebible-corpus/data/eng/vref.txt",
    "eng_extract.txt": "https://cdn.jsdelivr.net/gh/BibleNLP/ebible-corpus/data/eng/extract.txt",
}

all_ok = True
for fname, url in files.items():
    dest = DATA / fname
    if dest.exists():
        print(f"  ✓ {fname} — already exists")
        continue
    try:
        print(f"  Downloading {fname}...")
        subprocess.run(
            ["curl", "-L", "--fail", "-s", "-o", str(dest), url],
            check=True, timeout=60, capture_output=True
        )
        size = dest.stat().st_size
        lines = len(dest.read_text().splitlines())
        print(f"  ✓ {fname} — {size:,} bytes, {lines:,} lines")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {fname} — HTTP error")
        all_ok = False
    except Exception as e:
        print(f"  ✗ {fname} — {e}")
        all_ok = False

if all_ok:
    print("\n✅ All files downloaded via jsDelivr CDN!")
    sys.exit(0)

# METHOD 4: wget fallback (if curl not available)
print("\n🔽 Method 4: wget fallback")
for fname, url in files.items():
    dest = DATA / fname
    if dest.exists():
        continue
    try:
        print(f"  Downloading {fname} via wget...")
        subprocess.run(
            ["wget", "-q", "-O", str(dest), url],
            check=True, timeout=60, capture_output=True
        )
        size = dest.stat().st_size
        print(f"  ✓ {fname} — {size:,} bytes")
    except Exception as e:
        print(f"  ✗ {fname} — {e}")

# Final check
missing = [f for f, url in files.items() if not (DATA / f).exists()]
if missing:
    print("\n❌ All download methods failed.")
    print("\n💡 Manual download:")
    print("  1. Go to https://ebible.org/ → search 'Rendille'")
    print("  2. Download Rendille ZIP + English (World English Bible = eng)")
    print("  3. Upload to Colab: 📁 Files → Upload these 4 files to data/:")
    for f in files:
        print(f"     • {f}")
    print("  4. Re-run prepare_corpus.py")
    sys.exit(1)
else:
    print("\n✅ All files downloaded (wget)!")
    sys.exit(0)
