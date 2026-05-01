#!/usr/bin/env python3
from pathlib import Path
import importlib

BASE = Path(__file__).parent.parent

def check_env():
    """Run environment check for Rendille NMT project."""
    print("\n" + "="*60)
    print("🔍 RENDILLE NMT - ENVIRONMENT CHECK")
    print("="*60)
    
    items = []
    items.append(("scripts dir", (BASE/"scripts").exists()))
    items.append(("data dir", (BASE/"data").exists()))
    items.append(("models dir", (BASE/"models").exists()))
    
    for pkg in ["torch", "transformers", "datasets", "sacrebleu", "peft", "accelerate", "sentencepiece"]:
        try:
            importlib.import_module(pkg)
            items.append((f"pkg:{pkg}", True))
        except:
            items.append((f"pkg:{pkg}", False))
    
    for f in ["rel_vref.txt", "rel_extract.txt", "eng_extract.txt"]:
        items.append((f"data/{f}", (BASE/"data"/f).exists()))
    
    passed = sum(1 for _, ok in items if ok)
    pct = passed / len(items) * 100
    
    print("\nStatus check:")
    for name, ok in items:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n📊 SCORE: {passed}/{len(items)} — {pct:.0f}%")
    
    if passed == len(items):
        print("\n✅ All systems go! Next: python scripts/train_nmt.py")
    else:
        print("\n⏳ Actions needed:")
        if not all(ok for n, ok in items if n.startswith('pkg:')):
            print("  • pip install torch transformers datasets sacrebleu peft accelerate sentencepiece")
        if not all(ok for n, ok in items if n.startswith('data/')):
            print("  • python scripts/download_bible.py (get Rendille-English Bible)")
    print("="*60 + "\n")

if __name__ == "__main__":
    check_env()
