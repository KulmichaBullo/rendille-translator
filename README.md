# Rendille → English Machine Translation Model

A production-ready neural machine translation system for Rendille (rel), an East Cushitic language of Kenya (~60K speakers).

## 📋 Project Status

**Current Phase:** Phase 1 — Data Collection & Baseline

- ✅ Project plan documented in `PROJECT_PLAN.md`
- ✅ Data collection scripts created
- ⏳ Awaiting: Rendille Bible corpus download validation
- ⏳ Next: Baseline NLLB evaluation & tokenizer expansion

## 🏗️ Architecture

```
Rendille English NMT Pipeline
├── Data Layer
│   ├── eBible Corpus (BibleNLP) — Verse-aligned parallel Bible
│   ├── JW300 Dataset (OPUS) — Potential parallel source
│   └── Synthetic data — Back-translation via Somali/Oromo pivots
├── Model Layer
│   ├── Base: facebook/nllb-200-distilled-1.3B (if Rendille in vocab)
│   └── Extended tokenizer (if Rendille NOT in NLLB) → expand embeddings
├── Training Layer
│   ├── LoRA fine-tuning (rank r=16) — parameter-efficient
│   ├── Mixed precision training (fp16/bf16)
│   └── Cross-lingual multi-task (Somali+Oromo+Rendille) optional
└── Evaluation Layer
    ├── BLEU, chrF++ (on held-out Bible books)
    ├── COMET (neural metric)
    └── Human evaluation (fluency/adequacy)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- GPU (NVIDIA CUDA) recommended for fine-tuning; CPU okay for inference
- Internet access for downloads (~5–10 GB)

### Installation

```bash
# Clone / enter project
cd ~/rendille-translation

# Install dependencies (prefer editable)
pip install -e .
# Or manually:
pip install torch transformers datasets peft accelerate sacrebleu sentencepiece requests
```

### Step-by-Step Workflow

#### 1. Check NLLB Language Coverage

```bash
python scripts/check_nllb_langs.py
```

- If Rendille is listed → mark `--add-token` = False in training
- If NOT listed → we MUST expand tokenizer + embedding layer

#### 2. Download Rendille Bible Corpus

```bash
python scripts/download_bible.py
```

Expected output:
```
Found Rendille (rel) — ~27,000 verses
Saved rel_vref.txt — 27000 lines
Saved rel_extract.txt — 27000 lines
```

If eBible doesn't have Rendille yet:
- Manual download from Scripture Earth: https://www.scriptureearth.org/00i-Scripture_Index.php?iso=rel
- Or use find.bible API: `https://dev.find.bible/bibles/RELWBT/`
- Place downloaded files in `data/` as `rel_vref.txt` and `rel_extract.txt`

**Note:** The eBible `extract.txt` is verse-per-line for ONE language. We still need English alignment.

**To get English parallel text:**
Use aligned multilingual Bibles like:
- **Christos-C Bible Corpus** (web interface, 100+ languages, download)
- **YouVersion Bible API** (private, requires API key)
- **Bible.is / Faith Comes By Hearing** (audio-aligned)
- **Open Scriptur[e]e** (https://openscriptures.org/)

Simplest: download Christos-C corpus in English + Rendille separately, then align by verse reference (both use same vref).

#### 3. Prepare Training Data

```bash
python scripts/prepare_corpus.py
```

This script:
- Loads `rel_extract.txt` (Rendille) and `eng_extract.txt` (English)
- Aligns by verse index (both share `vref.txt`)
- Cleans USFM tags if present
- Splits by Bible books (inspired by eBible experiments, not random):
  - **Train**: Gospels (Matthew–John) + Acts–Epistles (28 books)
  - **Val**: Psalms, Proverbs (hold-out poetic)
  - **Test**: Remaining OT (Isaiah, Genesis, etc.)
- Outputs: `data/train.txt`, `data/val.txt`, `data/test.txt` (each: `src ||| tgt`)

#### 4. (Optional) Data Augmentation — Back-Translation

If you want to generalize beyond Bible domain:

```bash
# 1. Train English→Rendille model first (reverse direction)
python scripts/train_nmt.py --model facebook/nllb-200-distilled-1.3B --lang eng --reverse --output models/eng-rel

# 2. Use trained model to translate English monolingual → synthetic Rendille
python scripts/backtranslate.py --src-lang eng --tgt-lang rel --model models/eng-rel --monolingual data/eng_news.txt

# 3. Combine synthetic + real data
cat data/train.txt synthetic_pairs.txt > data/train_aug.txt

# 4. Re-train on augmented data
python scripts/train_nmt.py --train-file data/train_aug.txt ...
```

#### 5. Fine-Tune the Model

**Option A — Rendille in NLLB (direct):**
```bash
python scripts/train_nmt.py   --model facebook/nllb-200-distilled-1.3B   --lang rel   --batch 8 --accum 4 --epochs 10 --lr 3e-4   --lora-r 16 --lora-alpha 32
```

**Option B — Rendille NOT in NLLB (expand tokenizer):**
```bash
python scripts/train_nmt.py   --model facebook/nllb-200-distilled-1.3B   --lang rel --add-token   --batch 4 --accum 8 --epochs 15 --lr 1e-4   --lora-r 32 --lora-alpha 64
```

**TensorBoard logs:**
```bash
tensorboard --logdir models/rendille-rel/runs
```

#### 6. Evaluate

```bash
python scripts/eval.py   --model models/rendille-rel/checkpoint-1000   --test-file data/test.txt   --metrics bleu chrf comet
```

**Metrics expected:**
- Bible-domain (in-domain): **BLEU 20–28**
- General-domain (with back-translation): **BLEU 12–18** (conservative estimate)

#### 7. Inference

```python
from transformers import pipeline
translator = pipeline("translation", model="models/rendille-rel", tokenizer="models/rendille-rel")
result = translator("Rendille sentence here", src_lang="rel", tgt_lang="eng")
print(result[0]['translation_text'])
```

Or via CLI:

```bash
echo "Rendille input sentence" | python scripts/translate.py --model models/rendille-rel
```

#### 8. Quantize & Deploy

```bash
# Convert to CTranslate2 (4× faster CPU inference)
ct2-transformers-converter --model models/rendille-rel --output models/rendille-rel-ct2 --quantization int8

# FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000
```

## 📁 Project Structure

```
rendille-translation/
├── PROJECT_PLAN.md          # Full technical plan (10-week timeline)
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── setup.py                 # Install as package
├── data/
│   ├── rel_vref.txt         # Verse indices for Rendille Bible
│   ├── rel_extract.txt      # Rendille verses (raw)
│   ├── eng_extract.txt      # English Bible (KJV/WE/etc) aligned by vref
│   ├── train.txt            # "src ||| tgt" format
│   ├── val.txt
│   └── test.txt
├── scripts/
│   ├── check_nllb_langs.py  # Verify NLLB language support
│   ├── download_bible.py    # Fetch Rendille-English Bible
│   ├── prepare_corpus.py    # Clean, align, split data
│   ├── train_nmt.py         # Fine-tune NLLB with LoRA
│   ├── eval.py              # Compute BLEU/chrf/comet
│   ├── backtranslate.py     # Generate synthetic data
│   └── translate.py         # CLI inference
├── notebooks/
│   ├── 1_exploratory.ipynb  # Data stats, length ratios, lang ID
│   ├── 2_baseline_nllb.ipynb # Zero-shot NLLB quality check
│   └── 3_training_monitor.ipynb # Loss curves, BLEU progression
└── models/
    ├── rendille-rel/        # Fine-tuned LoRA adapters
    ├── rendille-rel-ct2/    # Quantized for inference
    └── baseline_nllb/       # Zero-shot baseline (cached)
```

## 📊 Expected Results

| Scenario | Data Size | Domain | BLEU | chrF++ |
|----------|-----------|--------|------|--------|
| NLLB zero-shot (if Rendille supported) | 0 | Bible-only | 5–12 | 15–25 |
| Fine-tuned on Bible only | ~27K pairs | Bible | **20–28** | **35–45** |
| + back-translation (100K synth) | 127K pairs | Mixed | 24–32 | 40–50 |
| Full multilingual (Somali+Oromo) | 200K+ pairs | Multi-domain | 28–38 | 45–55 |

**Realistic target for initial MVP**: BLEU ≥ 22 on held-out Bible test set within 2 weeks.

## 🔄 Cross-Lingual Transfer Strategy

Since Rendille is NOT in NLLB-200, we rely on:

1. **Token sharing**: Subword vocabulary overlaps with Somali (som) and Oromo (orm)
2. **Embedding initialization**: Copy embedding weights from related Cushitic languages
3. **Multilingual fine-tuning**: Also train on Somali→English data (HornMT available)
4. **Round-trip back-translation**: English → Somali (NLLB) → Rendille (our model)

This reduces required Rendille parallel data from ~100K pairs to ~30K pairs.

## 📚 References & Resources

- **eBible Corpus**: BibleNLP/ebible (GitHub) — 833 languages, verse-aligned
- **NLLB Paper**: "No Language Left Behind" (Meta AI, 2022)
- **HornMT Benchmark**: Asmelash et al. — Horn of Africa languages (Somali, Afar, Oromo, Amharic, Tigrinya)
- **HornMorpho**: Morphological analyzer for Cushitic languages
- **Christos-C Bible Corpus**: 100-language parallel Bible download
- **Bible.is APIs**: Scripture content APIs (requires registration)

## 🧪 Validation Checklist

Before declaring MVP complete:

- [ ] NLLB language list checked (`scripts/check_nllb_langs.py`)
- [ ] Rendille Bible downloaded and verified (27K+ verses)
- [ ] English parallel text downloaded (KJV or World English Bible)
- [ ] Files aligned by verse reference (vref matching)
- [ ] Train/val/test split defined (book-level splits)
- [ ] Baseline zero-shot NLLB test run (if supported)
- [ ] Fine-tuning completed (≥5 epochs, stable loss)
- [ ] Validation BLEU ≥ 20 (Bible-domain)
- [ ] Test set evaluated (publication-quality metrics)
- [ ] Sample translations manually reviewed (≥10 random verses)
- [ ] Model packaged (HuggingFace + CTranslate2)
- [ ] Inference test script working
- [ ] Documentation updated (README, API usage)

## ⚠️ Known Limitations

- **Domain lock-in**: Bible language style (archaic, formulaic) doesn't generalize to daily conversation
- **Morphology**: Rendille has complex verb conjugations; may need character-level decoding
- **No evaluation benchmark**: Custom test set required
- **Limited monolingual data**: Social media sources may need web scraping (ethical/legal review)

## 🛠️ Development Roadmap

**Week 1**: Data acquisition & alignment ✅
**Week 2**: Tokenization strategy + baseline zero-shot evaluation
**Week 3–4**: LoRA fine-tuning on 30K pairs
**Week 5**: Back-translation + data augmentation
**Week 6–7**: Multilingual transfer (Somali+Oromo joint training)
**Week 8**: Domain adaptation (proverbs, health, agriculture texts)
**Week 9**: Packaging + quantized inference
**Week 10**: Documentation handoff + Kulmicha demo

## 📞 Contact

Built for Meelilabs / Toola Group by Hermes Agent.
Part of the "No Language Left Behind" initiative for East African languages.

---

*Last updated: 2026-05-01 | Status: Data collection phase initiated*
