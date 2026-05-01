#!/usr/bin/env python3
"""Full speech-to-speech translation: Rendille audio → English audio"""
import sys, json, tempfile
from pathlib import Path
import torch

class SpeechToSpeechPipeline:
    def __init__(self):
        print("🔄 Loading models...")
        # ASR
        from transformers import pipeline as hf_pipeline
        self.asr = hf_pipeline("automatic-speech-recognition", model="asr/checkpoints/whisper-rel")
        # MT
        self.mt = hf_pipeline("translation", model="../models/rendille-rel")
        # TTS
        from TTS.api import TTS
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda")
        print("✅ All models loaded")

    def translate(self, audio_path: str, speaker_ref: str = None):
        # ASR
        print("🎤 ASR...")
        rel_text = self.asr(audio_path)["text"]
        # MT
        print("🌐 MT...")
        eng_text = self.mt(rel_text, src_lang="rel", tgt_lang="eng")[0]["translation_text"]
        # TTS
        print("🗣️  TTS...")
        out_path = "output_english.wav"
        self.tts.tts_to_file(text=eng_text, file_path=out_path, speaker_wav=speaker_ref, language="en")
        return out_path, eng_text, rel_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <rendille_audio.wav> [speaker_ref.wav]")
        sys.exit(1)
    pipe = SpeechToSpeechPipeline()
    out, eng, rel = pipe.translate(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None)
    print(f"
🎧 Output: {out}")
    print(f"📝 English: {eng}")
    print(f"📝 Rendille: {rel}")
