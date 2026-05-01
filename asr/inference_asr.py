#!/usr/bin/env python3
"""Transcribe Rendille speech using fine-tuned Whisper"""
from transformers import pipeline
import sys

model_path = "asr/checkpoints/whisper-rel"
asr = pipeline("automatic-speech-recognition", model=model_path)

audio_file = sys.argv[1] if len(sys.argv) > 1 else None
if audio_file:
    result = asr(audio_file)
    print(f"📝 Transcription: {result['text']}")
else:
    print("Usage: python inference_asr.py <audio.wav>")
