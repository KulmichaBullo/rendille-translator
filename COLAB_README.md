# Rendille Translator — Colab Setup Guide

**Project:** Speech-to-speech translation for Rendille (East Cushitic language) using Whisper ASR + NLLB MT + Coqui XTTS

**Repository:** https://github.com/KulmichaBullo/rendille-translator

---

## Quick Start (5 minutes)

1. Open the notebook: [Open in Colab](https://colab.research.google.com/github/KulmichaBullo/rendille-translator/blob/master/rendille_colab.ipynb)
2. **Runtime → Change runtime type → T4 GPU → Save**
3. **Runtime → Restart runtime** (ensures GPU activates)
4. **Runtime → Run all** (or press Ctrl+F9)
5. Wait 2–4 hours for training; Gradio demo launches automatically

---

## What Each Cell Does

  Cell 0 (markdown): # Rendille Translation Pipeline — Google Colab **Speech-to-speech translation: R...
  Cell 1 (code): # Check GPU availability !nvidia-smi  import torch print(f'PyTorch: {torch.__ver...
  Cell 2 (code): # Install Python dependencies FIRST (before any other imports) !pip install torc...
  Cell 3 (code): # Clone repository !git clone https://github.com/KulmichaBullo/rendille-translat...
  Cell 4 (code): # Quick environment verification import sys sys.path.append('.') from scripts.se...
  Cell 5 (code): # Download Rendille–English Bible corpus # Uses multiple fallback strategies (CD...
  Cell 6 (code): # MANUAL UPLOAD FALLBACK (if automatic download fails)\n#\n# If the cell above f...
  Cell 7 (code): # Clean and split corpus into train/val/test !python scripts/prepare_corpus.py...
  Cell 8 (code): # Train NLLB-200 fine-tuned model with LoRA # Reduced epochs for demo — increase...
  Cell 9 (code): # Test the trained model !python scripts/translate.py --text "Kaayo" --model mod...
  Cell 10 (code): # Launch Gradio web UI import threading, time  def launch_demo():     from pipel...
  Cell 11 (markdown): ## 🔊 Speech-to-Speech (When Audio Data Arrives)  Once you obtain the Rendille Bi...

### Cell Execution Flow

| Order | Cell | Purpose | Duration |
|-------|------|---------|----------|
| 1 | GPU check | Verify T4 GPU is active | 5 sec |
| 2 | Install dependencies | pip install torch, transformers, datasets, etc. | 2–5 min |
| 3 | Git clone | Pull project code from GitHub | 30 sec |
| 4 | Environment check | Verify packages + expected files | 5 sec |
| 5 | Download Bible corpus | Auto-download via HuggingFace/ebible fallbacks | 1–3 min |
| 6 | Manual upload fallback | Shows instructions if Cell 5 failed (safe to skip) | Instant |
| 7 | Prepare corpus | Clean + split into train/val/test | 10 sec |
| 8 | Train NLLB model | Fine-tune with LoRA (2–4 hrs on T4) | 2–4 hrs |
| 9 | Test translation | Try "Kaayo" → English output | 5 sec |
| 10 | Launch Gradio | Starts web demo (shareable URL) | 5 sec |

---

## Expected Outputs

### After Cell 5 (Download)

```
📥 DOWNLOADING RENDILLE–ENGLISH BIBLE CORPUS
============================================================
🔽 Method 1: HuggingFace datasets
  Loading bible-nlp/biblenlp-corpus (streaming)...
  ✓ Got 27,000+ verses via HuggingFace
  ✓ Files: rel_extract.txt, eng_extract.txt, rel_vref.txt
✅ Download complete!
```

If all methods fail, Cell 6 will show manual upload instructions.

### After Cell 7 (Prepare)

```
✓ Train size: 25,000 verses
✓ Val size: 1,000 verses  
✓ Test size: 1,000 verses
✓ UTF-8 validated
```

### After Cell 8 (Train)

Training progress bar showing:
- Loss decreasing over epochs
- Steps: 100/2500, 200/2500, ...
- Checkpoints saved to `models/rendille-rel/`

**Colab may disconnect after 12 hours** — training continues in background. Reconnect to notebook to see final weights.

---

## Troubleshooting

### GPU not detected
- **Symptom:** `nvidia-smi: not found` or `CUDA available: False`
- **Fix:** Runtime → Change runtime type → Select **T4 GPU** → Save → Restart runtime

### Download fails with 403/404
Colab networks block direct URLs. The script tries 3 methods automatically. If all fail, Cell 6 will guide manual upload.

### ImportError after code changes
You edited a script file (e.g., `download_bible.py`) and re-ran cells. Colab cached the old version.
- **Fix:** Runtime → Restart runtime, then Run all

### Out of memory during training
- **Symptom:** `RuntimeError: CUDA out of memory`
- **Fix 1:** Reduce batch size — edit Cell 8: change `--batch 4` to `--batch 2`
- **Fix 2:** Use gradient accumulation — add `--gradient_accumulation_steps 2`
- **Fix 3:** Switch to smaller model — NLLB-200 distilled (already default)

### Colab disconnects mid-training
Colab free tier limits sessions to 12 hours.
- **Recovery:** The script saves checkpoints every 100 steps. After reconnecting:
  - Find latest checkpoint in `models/rendille-rel/checkpoint-*`
  - Resume by adding `--resume_from_checkpoint models/rendille-rel/checkpoint-XXX`

### Disk quota exceeded
- **Fix:** Clear large files: `!rm -rf data/raw/ downloads/`
- **Or:** Move intermediate files to `/tmp` (ephemeral, larger quota)

---

## Model Details

- **Base model:** `facebook/nllb-200-distilled-600M` (NLLB-200, 600M parameters)
- **Fine-tuning:** LoRA (rank=16, alpha=32) — only 0.1% of weights train
- **Language pair:** Rendille (`rel`) → English (`eng`)
- **Corpus:** New Testament (~27,000 verse pairs)
- **Expected BLEU:** 18–25 (roughly understandable, needs post-editing)
- **Training time:** 2–4 hrs on T4 GPU (10 epochs)

---

## After Training

When Cell 10 launches Gradio:
- A public URL appears: `https://xxxxxxxx.gradio.live`
- Open it in any browser
- Type Rendille text (e.g., "Kaayo", "Haba", "Natumkese") → see English translation
- Share the link with collaborators (valid while Colab session runs)

**To keep demo running permanently:** Deploy to Hugging Face Spaces (free) or run locally with `python pipeline/demo_app.py` after downloading model weights.

---

## Next Steps (Speech-to-Speech)

Once text MT works, add speech:

1. **Obtain Rendille audio Bible** (~40 hours)
   - Email MegaVoice: request `rel` (Rendille) NT audio
   - Alternative: Global Recordings Network
2. **Train Whisper ASR** using `asr/train_whisper.py`
3. **Integrate TTS** using Coqui XTTS (already set up in `pipeline/orchestrator.py`)
4. **Full pipeline:** Speech (Rendille) → Text (Rendille) → Text (English) → Speech (English)

See `SPEECH_PIPELINE_GUIDE.md` for details.

---

## Files Overview

```
rendille-translator/
├── rendille_colab.ipynb      ← Colab notebook
├── COLAB_README.md           ← This file
├── scripts/
│   ├── check_nllb_langs.py   # Verify NLLB supports Rendille
│   ├── download_bible.py     # Multi-strategy Bible downloader
│   ├── prepare_corpus.py     # Clean + train/val/test split
│   ├── train_nmt.py          # LoRA fine-tuning
│   ├── translate.py          # Inference script
│   └── setup_checklist.py    # Environment validator
├── data/                     # Created by download/prepare
│   ├── rel_extract.txt
│   ├── eng_extract.txt
│   └── rel_vref.txt
├── models/rendille-rel/      # Created by training
│   ├── adapter_model/
│   ├── training_args.json
│   └── ...
└── pipeline/
    ├── demo_app.py           # Gradio UI
    └── orchestrator.py       # ASR+MT+TTS (when audio added)
```

---

## Time & Cost

| Resource | Estimate |
|----------|----------|
| Colab GPU (T4) | Free |
| Training time | 2–4 hours |
| Download time | 1–3 minutes |
| Total wall-clock | ~4 hours (training runs in background) |
| After training | Export model (~500 MB) or keep on Colab |

---

## Support

Issues? Check:
- Notebook cell error messages (they include recovery hints)
- `PROJECT_PLAN.md` for development roadmap
- `SPEECH_PIPELINE_GUIDE.md` for audio integration steps

Report bugs: https://github.com/KulmichaBullo/rendille-translator/issues

---

**Last updated:** 2026-05-01 | Cell count: 12 | Training: ~10 epochs default
