#!/usr/bin/env python3
from pathlib import Path
import random

DATA = Path("data")

# Read aligned Bible: vref + extracts
# Format per line: source_text ||| target_text
vref = (DATA/"rel_vref.txt").read_text().splitlines() if (DATA/"rel_vref.txt").exists() else []
rel_text = (DATA/"rel_extract.txt").read_text().splitlines() if (DATA/"rel_extract.txt").exists() else []
eng_text = (DATA/"eng_extract.txt").read_text().splitlines() if (DATA/"eng_extract.txt").exists() else []

print(f"Loaded {len(vref)} verses")
print(f"  Rendille lines: {len(rel_text)}")
print(f"  English lines:  {len(eng_text)}")

if vref and rel_text and eng_text and len(rel_text) == len(eng_text) == len(vref):
    # Create parallel pairs: (Rendille, English)
    pairs = list(zip(rel_text, eng_text))
    random.shuffle(pairs)
    
    n = len(pairs)
    splits = {
        "train": pairs[:int(0.8*n)],
        "val":   pairs[int(0.8*n):int(0.9*n)],
        "test":  pairs[int(0.9*n):],
    }
    
    for split, data in splits.items():
        out = DATA/f"{split}.txt"
        out.write_text("\n".join(f"{src} ||| {tgt}" for src, tgt in data))
        print(f"✓ {split}: {len(data)} lines → {out}")
else:
    print("ERROR: Missing files or length mismatch")
    print("  Expected: rel_vref.txt, rel_extract.txt, eng_extract.txt all same length")
    exit(1)
