#!/usr/bin/env python3
"""
Download Rendille–English Bible parallel corpus.

Multiple fallback strategies to handle GitHub rate limits and HF changes.
"""
from pathlib import Path
import sys
import time

DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)
ISO = "rel"

def try_direct_cdn():
    """Try direct downloads from eBible CDN mirrors."""
    print("\n🔽 Method: Direct CDN download")
    
    # Multiple mirrors to try
    bases = [
        "https://ebible.org/Download/",
        "https://mirror.c punkt.de/ebible/",
        "https://cdn.jsdelivr.net/gh/BibleNLP/ebible-corpus@main",
        "https://raw.githubusercontent.com/BibleNLP/ebible-corpus/main",
    ]
    
    # Files we need for Rendille
    targets = {
        "vref.txt": f"{ISO}_vref.txt",      # Scripture references (book ch:vs)
        "extract.txt": f"{ISO}_extract.txt" # Rendille verses
    }
    
    # Also get English for alignment
    eng_targets = {
        "vref.txt": "eng_vref.txt",
        "extract.txt": "eng_extract.txt"
    }
    
    success = True
    
    # Download Rendille
    for fname, out_name in targets.items():
        ok = False
        for base_url in bases:
            url = f"{base_url}/data/{ISO}/{fname}"
            print(f"  Trying: {url}")
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; RendilleMT/1.0)"
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        content = resp.read().decode('utf-8', errors='replace')
                        out_path = DATA / out_name
                        out_path.write_text(content, encoding='utf-8')
                        lines = content.count('\n') + 1
                        print(f"    ✓ {out_name} — {lines:,} lines")
                        ok = True
                        break
            except Exception as e:
                # print(f"    ✗ {e}")  # Silently try next mirror
                pass
        if not ok:
            print(f"    ✗ Failed to download {out_name} from all mirrors")
            success = False
    
    # Download English
    for fname, out_name in eng_targets.items():
        ok = False
        for base_url in bases:
            url = f"{base_url}/data/eng/{fname}"
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; RendilleMT/1.0)"
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        content = resp.read().decode('utf-8', errors='replace')
                        out_path = DATA / out_name
                        out_path.write_text(content, encoding='utf-8')
                        lines = content.count('\n') + 1
                        print(f"    ✓ {out_name} — {lines:,} lines")
                        ok = True
                        break
            except:
                pass
        if not ok:
            print(f"    ⚠ Could not download {out_name} (English)")
    
    return success

def try_huggingface_streaming():
    """Fallback: stream from HuggingFace (no file downloads)."""
    print("\n🔽 Method: HuggingFace streaming (experimental)")
    try:
        from datasets import load_dataset
        ds = load_dataset("bible-nlp/biblenlp-corpus", split="train", streaming=True)
        
        rendille = []
        english = []
        refs = []
        
        count = 0
        batch = 1000  # process in batches
        
        for ex in ds:
            langs = ex['translation']['languages']
            texts = ex['translation']['translation']
            ref_list = ex['ref'] if isinstance(ex['ref'], list) else [ex['ref']]
            
            if ISO in langs:
                rendille.append(texts[langs.index(ISO)])
                if 'eng' in langs:
                    english.append(texts[langs.index('eng')])
                else:
                    english.append("")
                refs.extend(ref_list)
                count += 1
            
            if count % 1000 == 0 and count > 0:
                print(f"  ... collected {count} verses")
            if count >= 5000:  # limit for demo; increase for production
                break
        
        if rendille:
            (DATA / f"{ISO}_extract.txt").write_text("\n".join(rendille), encoding='utf-8')
            (DATA / "eng_extract.txt").write_text("\n".join(english), encoding='utf-8')
            (DATA / f"{ISO}_vref.txt").write_text("\n".join(refs), encoding='utf-8')
            print(f"  ✓ Got {len(rendille)} verses via streaming")
            return True
    except Exception as e:
        print(f"  ✗ Streaming failed: {e}")
    return False

if __name__ == "__main__":
    print("="*60)
    print("📥 DOWNLOADING RENDILLE–ENGLISH BIBLE CORPUS")
    print("="*60)
    
    # Try methods in order
    if try_direct_cdn():
        print("\n✅ Download complete via CDN mirrors!")
        sys.exit(0)
    
    print("\n⚠ CDN mirrors didn't work — trying HuggingFace streaming...")
    if try_huggingface_streaming():
        print("\n✅ Download complete via HuggingFace!")
        sys.exit(0)
    
    print("\n❌ All download methods failed.")
    print("\n💡 Manual options:")
    print("  1. Download directly from https://ebible.org/")
    print("     Search for 'Rendille' and download the .txt files")
    print("  2. Place files in data/ as: rel_vref.txt, rel_extract.txt, eng_extract.txt")
    print("  3. Or contact MegaVoice for audio+aligned Bible corpus")
    sys.exit(1)
