#!/usr/bin/env python3
import requests, csv, io
from pathlib import Path
DATA = Path(__file__).parent.parent / "data"; DATA.mkdir(exist_ok=True)
meta = "https://raw.githubusercontent.com/BibleNLP/ebible/main/metadata.tsv"
r = requests.get(meta, timeout=30)
if r.ok:
    rows = list(csv.DictReader(io.StringIO(r.text), delimiter='\t'))
    rel = [x for x in rows if x.get('ISO','').lower() == 'rel']
    if rel:
        iso, name = 'rel', rel[0].get('Name','Rendille')
        print(f"Found: {name} ({iso}) — {rel[0].get('Total Verses','?')} verses")
        for f in ['vref', 'extract']:
            u = f"https://cdn.jsdelivr.net/gh/BibleNLP/ebible@main/data/{iso}/{f}.txt"
            fr = requests.get(u, timeout=30)
            if fr.ok:
                (DATA/f"{iso}_{f}.txt").write_text(fr.text)
                print(f"  Downloaded {iso}_{f}.txt — {len(fr.text.splitlines())} lines")
            else: print(f"  Failed {f}: HTTP {fr.status_code}")
    else: print("Rendille not in eBible metadata — check alternative sources")
else: print(f"Metadata download failed: {r.status_code}")
