#!/usr/bin/env python3
"""Prepare ASR training data: convert audio to 16kHz WAV, create Whisper manifest"""
import json, subprocess
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RAW_AUDIO = DATA_DIR / "audio_raw"  # downloaded MP3s
PROCESSED_AUDIO = DATA_DIR / "audio"  # 16kHz WAVs
MANIFEST = DATA_DIR / "train.jsonl"

def convert_to_wav(src_mp3: Path, dst_wav: Path):
    """ffmpeg: MP3 → 16kHz mono WAV"""
    cmd = ["ffmpeg", "-i", str(src_mp3), "-ar", "16000", "-ac", "1", str(dst_wav), "-y"]
    subprocess.run(cmd, capture_output=True)
    return dst_wav.exists()

def build_manifest():
    verses = (Path(__file__).parent.parent / "data" / "rel_extract.txt").read_text().splitlines()
    pairs = []
    audio_files = sorted(RAW_AUDIO.glob("*.mp3")) + sorted(RAW_AUDIO.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files for {len(verses)} verses")
    for idx, (audio_path, verse_text) in enumerate(zip(audio_files, verses)):
        dst = PROCESSED_AUDIO / f"verse_{idx:05d}.wav"
        if audio_path.suffix == ".mp3":
            convert_to_wav(audio_path, dst)
        else:
            dst.write_bytes(audio_path.read_bytes())  # already WAV
        pairs.append({"audio": str(dst), "text": verse_text})
    MANIFEST.write_text("\n".join(json.dumps(p) for p in pairs))
    print(f"✅ Manifest: {MANIFEST} ({len(pairs)} samples)")

if __name__ == "__main__":
    build_manifest()
