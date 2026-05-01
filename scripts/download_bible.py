#!/usr/bin/env python3
"""
Download Rendille–English Bible parallel corpus.

Strategy:
1. Primary: HuggingFace datasets (bible-nlp/biblenlp-corpus) — no rate limits
2. Fallback: wget with browser UA from GitHub (if HF fails)
"""
from pathlib import Path
import subprocess
import sys

DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)
ISO = "rel"

def check_hf_available():
    try:
        import datasets  # noqa
        return True
    except ImportError:
        return False

def download_via_huggingface():
    """Download using HuggingFace datasets library (recommended for Colab)."""
    print("\n🔽 Method 1: HuggingFace datasets")
    try:
        from datasets import load_dataset
        # Load streaming to avoid downloading entire 5GB corpus
        print("  Loading dataset from bible-nlp/biblenlp-corpus...")
        ds = load_dataset("bible-nlp/biblenlp-corpus", split="train", streaming=True)
        
        rendille = []
        english = []
        refs = []
        
        print("  Filtering for Rendille verses...")
        for ex in ds:
            langs = ex['translation']['languages']
            texts = ex['translation']['translation']
            if ISO in langs and 'eng' in langs:
                rendille.append(texts[langs.index(ISO)])
                english.append(texts[langs.index('eng')])
                refs.extend(ex['ref'] if isinstance(ex['ref'], list) else [ex['ref']])
                # Limit for demo; remove [:1000] for full Bible
                if len(rendille) >= 1000:
                    break
        
        if rendille:
            (DATA / f"{ISO}_extract.txt").write_text("\n".join(rendille), encoding='utf-8')
            (DATA / "eng_extract.txt").write_text("\n".join(english), encoding='utf-8')
            (DATA / f"{ISO}_vref.txt").write_text("\n".join(refs), encoding='utf-8')
            print(f"  ✓ Got {len(rendille)} verses via HuggingFace")
            return True
        else:
            print("  ✗ No Rendille verses found in dataset")
            return False
    except Exception as e:
        print(f"  ✗ HuggingFace error: {e}")
        return False

def download_via_wget():
    """Direct download using wget with browser User-Agent."""
    print("\n🔽 Method 2: wget with browser UA")
    # Use ebible-corpus repo
    base_url = "https://raw.githubusercontent.com/BibleNLP/ebible-corpus/main"
    files = {
        f"data/{ISO}/vref.txt": f"{ISO}_vref.txt",
        f"data/{ISO}/extract.txt": f"{ISO}_extract.txt",
        f"data/eng/vref.txt": "eng_vref.txt",  # English vref (same structure)
        f"data/eng/extract.txt": "eng_extract.txt",
    }
    
    success = True
    for src, dst in files.items():
        url = f"{base_url}/{src}"
        out = DATA / dst
        print(f"  Downloading {dst}...")
        cmd = ["wget", "-q", "-U", "Mozilla/5.0", "-O", str(out), url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out.exists():
            lines = out.read_text().splitlines()
            print(f"    ✓ {dst} — {len(lines)} lines")
        else:
            print(f"    ✗ Failed: {result.stderr[:100]}")
            success = False
    
    return success

def download_via_curl():
    """Last resort: curl with UA."""
    print("\n🔽 Method 3: curl fallback")
    base_url = "https://cdn.jsdelivr.net/gh/BibleNLP/ebible-corpus@main"
    # Try jsDelivr CDN
    for ftype in ['vref', 'extract']:
        url = f"{base_url}/data/{ISO}/{ftype}.txt"
        out = DATA / f"{ISO}_{ftype}.txt"
        cmd = ["curl", "-s", "-L", "-A", "Mozilla/5.0", "-o", str(out), url]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and out.exists():
            size = out.stat().st_size
            print(f"  ✓ {ISO}_{ftype}.txt — {size:,} bytes")
        else:
            print(f"  ✗ {ftype} failed")
            return False
    return True

if __name__ == "__main__":
    print("="*60)
    print("📥 DOWNLOADING RENDILLE–ENGLISH BIBLE CORPUS")
    print("="*60)
    
    # Try methods in order
    if check_hf_available():
        if download_via_huggingface():
            sys.exit(0)
    else:
        print("⚠ HuggingFace datasets not installed")
    
    if download_via_wget():
        sys.exit(0)
    
    if download_via_curl():
        sys.exit(0)
    
    print("\n❌ All download methods failed.")
    print("\n💡 Manual options:")
    print("  1. Visit https://ebible.org/ and search for Rendille")
    print("  2. Download the .txt files manually and place in data/")
    print("  3. Contact MegaVoice for aligned audio+text corpus")
    sys.exit(1)
