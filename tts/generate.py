#!/usr/bin/env python3
"""Text-to-Speech: English output using Coqui XTTS v2 (GPU) or Piper (CPU)"""
import sys, argparse
from pathlib import Path

def generate_xtts(text, output, speaker_wav=None, language="en"):
    """Coqui XTTS v2 — high quality, voice cloning"""
    try:
        from TTS.api import TTS
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda")
        tts.tts_to_file(text=text, file_path=output, speaker_wav=speaker_wav, language=language)
        print(f"✅ Generated {output} (XTTS)")
    except ImportError:
        print("❌ Install TTS: pip install TTS")
        sys.exit(1)

def generate_piper(text, output, voice_model=None):
    """Piper TTS — fast CPU inference"""
    import subprocess
    model = voice_model or "en_US-lessac-medium.onnx"
    cmd = ["piper", "--model", model, "--output_file", output]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(input=text.encode())
    print(f"✅ Generated {output} (Piper)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--output", default="output_eng.wav")
    p.add_argument("--speaker-wav", help="3-6 sec reference voice for cloning")
    p.add_argument("--backend", choices=["xtts","piper"], default="xtts")
    args = p.parse_args()
    if args.backend == "xtts":
        generate_xtts(args.text, args.output, args.speaker_wav)
    else:
        generate_piper(args.text, args.output)
