#!/usr/bin/env python3
from pathlib import Path
import random
DATA = Path("data")
# Read aligned Bible: vref + extract (English assumed as eng.txt if parallel exists)
# Format: source ||| target per line
vref = (DATA/"rel_vref.txt").read_text().splitlines() if (DATA/"rel_vref.txt").exists() else []
text = (DATA/"rel_extract.txt").read_text().splitlines() if (DATA/"rel_extract.txt").exists() else []
print(f"Loaded {len(vref)} verses, {len(text)} lines")
if vref and text and len(vref) == len(text):
    # Simple split: 80/10/10 by random (real: hold out books)
    pairs = list(zip(text, text))  # TODO: get actual English align
    random.shuffle(pairs)
    n = len(pairs)
    for split, s in [("train", (0, int(0.8*n))), ("val", (int(0.8*n), int(0.9*n))), ("test", (int(0.9*n), n))]:
        out = DATA/f"{split}.txt"
        out.write_text("\n".join(f"{src} ||| {tgt}" for src,tgt in pairs[s[0]:s[1]]))
        print(f"Wrote {split}: {s[1]-s[0]} lines → {out}")
else:
    print("ERROR: Missing vref or text file; corpus not ready")
