#!/usr/bin/env python3
"""
Download Rendille Bible audio from MegaVoice or Global Recordings.
Creates aligned dataset: audio/ + manifest.jsonl for Whisper fine-tuning.
"""
import requests, os, json, time
from pathlib import Path

# MegaVoice URL pattern (needs inspection; may require API key)
# Alternative: GlobalRecordings.net allows bulk ZIP download for research
# This script is a TEMPLATE — fill in actual URLs after manual download

DATA = Path(__file__).parent.parent / "data"
AUDIO_DIR = Path(__file__).parent / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def create_manifest_from_existing_audio():
    """
    If you manually downloaded MP3s, this creates the Whisper manifest.
    Assumes: audio files named like MAT_001_001.mp3, 1Cor_001_001.mp3, etc.
    Aligned to rel_extract.txt verse order.
    """
    text_file = DATA / "rel_extract.txt"
    verses = text_file.read_text(encoding="utf-8").splitlines() if text_file.exists() else []
    print(f"Found {len(verses)} verses in text corpus")

    manifest = []
    # Simple mapping: verse index → audio file
    # For real use, you need proper book→chapter→verse mapping
    for idx, text in enumerate(verses):
        audio_mp3 = AUDIO_DIR / f"verse_{idx:05d}.mp3"
        audio_wav = AUDIO_DIR / f"verse_{idx:05d}.wav"
        # Prefer WAV; convert if only MP3
        src = audio_mp3 if audio_mp3.exists() else audio_wav
        if src.exists():
            manifest.append({"audio": str(src), "text": text})

    out = Path(__file__).parent / "data" / "train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(m) for m in manifest))
    print(f"✅ Manifest: {out} — {len(manifest)} pairs")
    return manifest

def download_from_globalrecordings():
    """
    Use Global Recordings Network Bulk Download.
    1. Visit https://globalrecordings.net/en/language/rel
    2. Request research access: info@globalrecordings.net
    3. They provide ZIP of all audio + text
    """
    print("⚠️  Manual step required:")
    print("  1. Contact info@globalrecordings.net with Rendille language code 'rel'")
    print("  2. Request audio Bible dataset for research")
    print("  3. Extract ZIP to data/audio_raw/")
    print("  4. Rerun this script to convert+align")

if __name__ == "__main__":
    print("Rendille Audio Dataset Builder")
    print("="*50)
    print("Options:")
    print("  Option A — If you already have audio files:")
    create_manifest_from_existing_audio()
    print()
    print("  Option B — No audio yet:")
    download_from_globalrecordings()
