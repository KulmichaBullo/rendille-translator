# Speech-to-Speech Translation: Rendille ↔ English
## Complete Architecture & Implementation Guide

---

## 🎯 **System Overview**

Build a **real-time universal translator** for Rendille:

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Rendille      │────▶│  ASR         │────▶│  MT          │────▶│  TTS            │
│  Speaker       │     │  (Speech→   │     │  (Rendille→ │     │  (Text→        │
│  (Microphone)  │     │   Text)     │     │   English)   │     │   Speech)      │
└─────────────────┘     └──────────────┘     └──────────────┘     └─────────────────┘
        │                       │                    │                     │
        ▼                       ▼                    ▼                     ▼
   Audio (.wav/MP3)       Text (rel)           Text (eng)           Audio (.wav/MP3)
```

**Latency target:** <2 seconds end-to-end on consumer GPU  
**Accuracy target:**  
- ASR: WER 15–25% (Bible domain)  
- MT: BLEU 20–28  
- TTS: MOS ≥4.0 (naturalness)

---

## 📊 **Component Breakdown**

| Component | Model | Data Required | Expected Quality | Est. Training Time |
|-----------|-------|---------------|-----------------|-------------------|
| **ASR** | Whisper-small fine-tuned | ~20 hrs audio + transcripts (Bible) | WER 15–25% | 4–8 hrs (GPU) |
| **MT** | NLLB-200 + LoRA | ~27K verse pairs (text) | BLEU 20–28 | 2–4 hrs |
| **TTS** | Coqui XTTS v2 (English) | None (pretrained) | MOS 4.0+ | 0 (inference only) |

**Total pipeline:** 6–12 hours training + deployment

---

## 🗂️ **Data Acquisition Strategy**

### **1. ASR Data: Rendille Audio + Transcripts**

**Primary source: Bible audio recordings**

| Source | Format | Size | Access |
|--------|--------|------|--------|
| **MegaVoice** (Rendille NT) | Verse-level MP3 | ~40 hrs | Manual download from web player |
| **Global Recordings Network** | Story-level MP3 | ~10 hrs | Bulk download (requires request) |
| **YouVersion Bible App** | Streaming only | — | Not downloadable |
| **Faith Comes By Hearing** | Audio Bible | ~40 hrs | API access possible |

**Recommended approach:**

1. **MegaVerse scraper** — Each verse is a separate MP3 file with predictable URL pattern
   - Example: `https://megavoice.com/media/.../MAT_001_001.mp3` (Matthew 1:1)
   - Write script to download all 27K verses automatically
   - Expected: 20–40 hours total audio

2. **Forced alignment (if audio is chapter-level not verse-level)**
   - Use **Montreal Forced Aligner (MFA)** to segment+align
   - Input: audio files + reference text (our Bible corpus)
   - Output: word-level timestamps + verse-level splits

**ASR dataset format (Whisper):**
```jsonl
{"audio": "data/audio/MAT_001_001.wav", "text": "K优雅i ninyo algeyo narre..."}
{"audio": "data/audio/MAT_001_002.wav", "text": "Abraham ki Abraham..."}
```

**Target stats:**
- Total duration: 20–40 hours (New Testament)
- Average clip: 3–8 seconds (verse length)
- Sampling rate: 16 kHz (mono WAV for Whisper)

---

### **2. MT Data: Text Translation** (already covered)

Use eBible corpus: `rel_extract.txt` + `eng_extract.txt` (verse-aligned)

---

### **3. TTS Data: English Speech Synthesis**

**Off-the-shelf solution:** Coqui XTTS v2 (multilingual, supports English)

- No training needed
- Supports **voice cloning** with 3-second sample
- Can preserve speaker identity if we also clone Rendille voice to English (research)

**Alternative: Piper TTS** (faster, less compute)
- Pre-trained English voices (male/female)
- Real-time on CPU
- Lower quality but fast

---

## 🏗️ **Implementation Steps**

### **Phase 1: ASR System (Whisper Fine-tuning)**

#### Step 1.1: Download Rendille Bible Audio

```python
# scripts/download_rendille_audio.py
import requests, os, time
from pathlib import Path

# MegaVoice structure: Each verse has unique ID
# Use their API or scrape the web player
# Convert MP3 → WAV 16kHz (sox/ffmpeg)

def download_verse(audio_url, output_path):
    r = requests.get(audio_url, stream=True)
    # save as MP3 then convert
    ...

# Batch download all NT verses by book/chapter/verse
# Map: MAT.1.1 → audio filename
```

**Alternative if MegaVoice blocks scraping:**
- Use **Global Recordings Network** (`globalrecordings.net`) — they allow bulk download for research
- Contact them directly: `info@globalrecordings.net` with project description
- They often provide ZIPs of audio+text for endangered languages

#### Step 1.2: Force Alignment (if needed)

If audio files are chapter-level (not per verse):

```bash
# Install Montreal Forced Aligner
conda install -c conda-forge montreal-forced-aligner

# Align using Bible text as reference
mfa align \
  -- acoustic_model_path english_us_arpa \
  -- dictionary_path english_us_arpa.dict \
  data/audio_chapters/ data/text/ data/mfa_output/
```

But for verse-level audio, alignment is trivial: 1 audio file = 1 verse text.

#### Step 1.3: Preprocess for Whisper

```python
# scripts/preprocess_asr.py
import os, json, subprocess
from pathlib import Path

AUDIO_DIR = Path("data/audio")  # verse-level .mp3 files
TEXT_FILE = Path("data/rel_extract.txt")  # verses in order
OUTPUT = Path("asr/data/train.jsonl")

# Read Bible text (order matches audio file naming if consistent)
verses = TEXT_FILE.read_text(encoding='utf-8').splitlines()

samples = []
for i, verse_text in enumerate(verses):
    # Map index to filename: MAT_001_001.mp3, MAT_001_002.mp3...
    # This requires knowing book list and verse count per chapter
    # Simpler: have download script output manifest mapping
    audio_path = AUDIO_DIR / f"verse_{i:05d}.wav"
    if audio_path.exists():
        samples.append({"audio": str(audio_path), "text": verse_text})

OUTPUT.write_text("\n".join(json.dumps(s) for s in samples))
print(f"Created {len(samples)} ASR training samples")
```

**Audio conversion** (MP3 → 16kHz mono WAV):
```bash
# Using ffmpeg (batch)
for f in data/audio/*.mp3; do
  ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp3}.wav"
done
```

#### Step 1.4: Fine-tune Whisper

```python
# scripts/train_asr.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Audio
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="rendille", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Load dataset
dataset = load_dataset("json", data_files={"train": "asr/data/train.jsonl"}, split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_example(example):
    audio = example["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    # Truncate to 30 sec max
    return {"input_features": inputs.input_features[0], "labels": processor.tokenizer(example["text"]).input_ids}

dataset = dataset.map(prepare_example, remove_columns=dataset.column_names)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="asr/checkpoints/whisper-rel",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=100,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

**Estimated performance:**
- Base Whisper (zero-shot on Rendille): WER ~60–80% (terrible)
- Fine-tuned on 20 hrs: WER 15–25% (intelligible)
- With more data (40 hrs): WER 10–18%

---

### **Phase 2: Integrate MT (Existing)**

Use the text translation model from `models/rendille-rel/`:

```python
# mt/translate.py (already done)
from transformers import pipeline
translator = pipeline("translation", model="../models/rendille-rel", tokenizer="../models/rendille-rel")
def translate_rel_to_eng(text):
    return translator(text, src_lang="rel", tgt_lang="eng")[0]['translation_text']
```

---

### **Phase 3: TTS (English Output)**

**Option A: Coqui XTTS (best quality, needs GPU)**

```python
# tts/generate.py
from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda")

def text_to_speech(text, output_path="output.wav", speaker_wav="reference.wav"):
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=speaker_wav,  # 3-6 sec voice sample
        language="en"
    )
```

**Option B: Piper TTS (fast, CPU-friendly)**

```bash
# Install piper
pip install piper-tts

# Download English voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac-medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac-medium/en_US-lessac-medium.onnx.json

# Generate
echo "Hello, this is a test" | piper --model en_US-lessac-medium.onnx --output_file out.wav
```

---

### **Phase 4: Orchestrate Full Pipeline**

```python
# pipeline/speech_to_speech.py
import subprocess
import tempfile
from pathlib import Path
from transformers import pipeline
import torch

class RendilleTranslator:
    def __init__(self):
        print("Loading models...")
        # ASR
        self.asr = pipeline("automatic-speech-recognition", model="asr/checkpoints/whisper-rel")
        # MT
        self.mt = pipeline("translation", model="../models/rendille-rel")
        # TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    def translate_audio(self, input_audio_path: str, speaker_ref: str = "tts/voices/reference.wav"):
        """Full pipeline: Rendille speech → English speech"""
        # 1. ASR: speech → Rendille text
        print("🎤 Transcribing Rendille speech...")
        rel_text = self.asr(input_audio_path)["text"]
        print(f"📝 Rendille: {rel_text}")

        # 2. MT: Rendille → English
        print("🌐 Translating to English...")
        eng_text = self.mt(rel_text, src_lang="rel", tgt_lang="eng")[0]["translation_text"]
        print(f"📝 English: {eng_text}")

        # 3. TTS: English text → speech (voice from speaker_ref)
        print("🗣️  Generating English speech...")
        output_path = "output_english.wav"
        self.tts.tts_to_file(
            text=eng_text,
            file_path=output_path,
            speaker_wav=speaker_ref,
            language="en"
        )
        print(f"✅ Output: {output_path}")
        return output_path, eng_text, rel_text

# CLI usage
if __name__ == "__main__":
    import sys
    translator = RendilleTranslator()
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "input_rendille.wav"
    out, eng, rel = translator.translate_audio(audio_file)
    print(f"\n🎧 Play: {out}")
```

---

## 🎛️ **Real-Time Demo Options**

### **Option 1: Web App (Gradio) — Easiest**

```python
# demo/app.py
import gradio as gr
from pipeline.speech_to_speech import RendilleTranslator

translator = RendilleTranslator()

def process_audio(audio):
    # audio is (sample_rate, array) from Gradio
    # Save temp file
    return output_wav, {"Rendille": rel_text, "English": eng_text}

gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(source="microphone", type="numpy"),
    outputs=[gr.Audio(), gr.Textbox()],
    title="Rendille → English Real-Time Translator",
    description="Speak Rendille, hear English translation"
).launch(share=True)  # generates public URL
```

**Deploy:** `python demo/app.py` → opens localhost:7860 with microphone button

### **Option 2: Mobile App (Flutter + API)**

- Host translation API on server (FastAPI)
- Mobile app records audio → sends to API → plays response

```python
# api/server.py
from fastapi import FastAPI, UploadFile
from pipeline.speech_to_speech import RendilleTranslator

app = FastAPI()
translator = RendilleTranslator()

@app.post("/translate")
async def translate_audio(file: UploadFile):
    audio_bytes = await file.read()
    # Save temp, process
    output_wav = translator.translate_audio(temp_path)
    return {"audio_url": output_wav}
```

### **Option 3: Desktop App (PyQt/PySide)**

All-local, no cloud — best for privacy/offline use

---

## ⚙️ **Hardware Requirements**

| Stage | RAM | Disk | GPU | Time |
|-------|-----|------|-----|------|
| ASR training | 16 GB | 20 GB (audio) | RTX 3090+ (8GB VRAM) | 4–8 hr |
| MT training | 8 GB | 5 GB | RTX 2060+ | 2–4 hr |
| TTS inference | 8 GB | 2 GB | Optional (CPU works) | real-time |
| Full pipeline | 16 GB | 50 GB total | RTX 3060+ recommended | — |

**Minimum workable setup:**
- CPU-only: ASR slower (~5×), TTS slow (~2×), but works
- GPU required for real-time (<2 sec latency)

---

## 📈 **Expected Performance**

| Metric | Target | Baseline (no fine-tune) |
|--------|--------|------------------------|
| ASR WER (Bible domain) | 15–25% | 60–90% (Whisper zero-shot) |
| MT BLEU (text) | 20–28 | 5–12 (NLLB zero-shot) |
| TTS MOS (naturalness) | 4.0/5 | 4.2 (XTTS pretrained) |
| End-to-end latency | <2 sec | — |

**Realistic demo:** In-browser Gradio app translating 5-second Bible verses with understandable (if not perfect) output.

---

## 🏃 **Quick-Start Commands** (once data acquired)

```bash
# 1. ASR: Fine-tune Whisper on Rendille Bible audio
cd ~/rendille-translation
python asr/train_whisper.py \
  --data asr/data/train.jsonl \
  --model openai/whisper-small \
  --epochs 10 \
  --batch-size 16

# 2. MT: Already in models/rendille-rel/

# 3. TTS: Use pretrained XTTS (no training)

# 4. Launch demo
python demo/app.py --share
```

---

## 🔍 **Critical Challenges & Mitigations**

| Challenge | Why hard | Mitigation |
|-----------|----------|------------|
| **ASR data scarcity** | Rendille audio Bible exists but hard to download | Contact MegaVoice/Global Recordings directly for research partnership; offer to cite them |
| **Verse alignment** | Audio may be chapter-level not verse-level | Use Montreal Forced Aligner + Bible text reference |
| **Rendille not in Whisper vocab** | Whisper supports 99 languages; Rendille absent | Fine-tuning adds acoustic+language modeling; tokenizer already multilingual byte-level BPE handles OOV reasonably |
| **Real-time latency** | Three neural nets (ASR+MT+TTS) sequential | Batch processing; quantize models; optimize with ONNX/CTranslate2 |
| **No Rendille TTS** | Can't generate Rendille speech (output English only) | Acceptable for demo; future: voice-transfer research |
| **Domain mismatch** | Bible language ≠ conversational | Collect conversational audio separately (future); use back-translation for MT domain adaptation |

---

## 🎁 **Alternative: Two-Stage Hybrid**

If full ASR is too difficult, build **text-input-first** MVP:

```
User types or pastes Rendille text → MT → English TTS → speech output
```

This is **90% of the value** with 10% of the effort (skip ASR data collection).

**Then later:** Add ASR as second phase once audio data acquired.

---

## 📋 **Project Timeline (Speech-to-Speech Extended)**

| Week | Milestone |
|------|-----------|
| 1–2 | Text MT pipeline complete (already planned) |
| 3 | Acquire Rendille Bible audio (download + organize) |
| 4 | Align audio ↔ text (force-align if needed) → ASR dataset |
| 5 | Preprocess & fine-tune Whisper (ASR) |
| 6 | Evaluate ASR WER on held-out verses |
| 7 | Integrate ASR → MT → TTS pipeline |
| 8 | Build Gradio demo (microphone I/O) |
| 9 | Optimize latency (quantize, batch, GPU) |
| 10 | Deploy & handoff |

**Total effort:** ~150 hours of dev time (vs. 80 hours for text-only)

---

## 🧪 **Evaluation Plan**

### **ASR Metrics**
- **WER** (Word Error Rate) on held-out Bible books
- Target: ≤25%
- Test set: 500 verses not in training (e.g., Genesis, Exodus)

### **End-to-End Quality**
- Human evaluation: 10 native Rendille speakers rate translation accuracy 1–5
- BLEU cascaded: ASR output → MT → compare to reference English

### **Latency**
- Measure per-verse (avg 5 sec input) on target hardware (RTX 3060)

---

## 🆘 **Getting Help/Data**

**If MegaVoice/GRN download fails:**
- Email: `info@megavoice.com` — explain research project, ask for dataset access
- Email: `info@globalrecordings.net` — they often provide academic access
- Alternative: **Use ALFFA/OpenSLR** African speech data (Amharic, Swahili) to pre-train Whisper, then fine-tune limited Rendille

**If you have a Rendille-speaking contact:**
- Record 1–2 hours of conversational speech + transcripts (gold mine!)
- This would improve ASR WER dramatically

---

## 🎯 **Minimum Viable Demo (1-Week Sprint)**

If you want something working fast:

1. **Day 1–2:** Get text MT model working (already done in pipeline)
2. **Day 3:** Set up Coqui XTTS TTS (pretrained English)
3. **Day 4:** Build Gradio UI: text input → MT → TTS audio output
4. **Day 5:** Test with sample Rendille sentences
5. **Day 6–7:** Integrate ASR (if audio available), else demo text→speech only

**You can have a working demo in 1 week** that does:
- Input: Rendille text (pasted)
- Output: English speech (downloadable audio)

Adding microphone input (ASR) adds ~1 week of audio data grind.

---

## 📦 **Complete Scripts to Create**

I'll generate next:

- `asr/download_audio.py` — Batch-download Bible audio from MegaVoice/GRN
- `asr/align.py` — Force-align if needed (MFA wrapper)
- `asr/preprocess.py` — Create Whisper JSONL manifest
- `asr/train_whisper.py` — LoRA fine-tune Whisper
- `asr/inference_asr.py` — ASR demo
- `tts/generate.py` — XTTS/Piper wrapper
- `pipeline/orchestrator.py` — Full pipeline
- `demo/gradio_app.py` — One-click web demo
- `demo/cli.py` — Command-line: `python demo/cli.py input.wav`

Want me to write all these scripts now and complete the scaffold? That would make the entire system ready for you to just `pip install` and run (once audio data is gathered).
