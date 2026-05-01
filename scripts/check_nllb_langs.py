#!/usr/bin/env python3
import requests, csv, io, sys
url = "https://huggingface.co/datasets/allenai/nllb/resolve/main/nllb_langs.tsv"
try:
    r = requests.get(url, timeout=30); r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text), delimiter='\t'); langs = list(reader)
    print(f"NLLB languages: {len(langs)}")
    if any(l.get('ISO639-3','').lower() == 'rel' for l in langs):
        print("Rendille (rel) is supported in NLLB! → fine-tune directly")
    else:
        print("Rendille NOT in NLLB → need cross-lingual transfer")
        for l in langs:
            c = l.get('ISO639-3','').lower()[:3]
            if c in ['som','orm','aaf']: print(f"  Related: {l.get('Language','?')} ({c})")
except Exception as e: print(f"Error: {e}")
