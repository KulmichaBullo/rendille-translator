# Building a Rendille-to-English Translation Model
## Complete Project Plan

## 1. Executive Summary

This document outlines a comprehensive approach to building a machine translation model for **Rendille (ISO: rel) → English**. Rendille is a Lowland East Cushitic language spoken by ~60,000 people in northern Kenya. This is a **low-resource language** scenario with:
- Very limited parallel corpora beyond religious texts (Bible)
- No dedicated NMT research prior work
- ~600K speakers maximum

**Goal**: Build a working NMT system (BLEU > 20 on held-out test set) that can translate general domain text, not just Bible verses.

---

## 2. Dataset Strategy

### 2.1 Primary Source: Bible Corpus (Parallel)
- **eBible Corpus / BibleNLP**: 1,009 translations in 833 languages, verse-aligned
  - Repository: https://github.com/BibleNLP/ebible
  - Hugging Face: `bible-nlp/biblenlp-corpus`
  - Format: verse-per-line, with vref.txt index
  - Rendille (rel) likely included as partial or complete NT
- **Rendille Wycliffe Bible (RELWBT)**: Dedicated translation
  - API: `https://dev.find.bible/bibles/RELWBT/`
  - Scripture Earth: Downloadable formats (USFM, plain text)
- **YouVersion Bible API**: Bible.com returns verse-aligned text

**Estimated Size**: 27,000+ New Testament verses (if full NT available) = ~30K sentence pairs

### 2.2 Secondary Sources: Monolingual & Augmentation
- **Web-scraped news**: If any Rendille news sites exist (Kenya)
- **Social media**: Twitter/X, Facebook posts (with API access)
- **HornMT**: Regional parallel corpus (Afar, Oromo, Somali, etc.) — NOT Rendille but can enable cross-lingual transfer if Rendille not in NLLB
- ** JW300**: Jehovah's Witnesses publications in 300+ languages (JW300 dataset on OPUS, possibly includes Rendille)

### 2.3 Synthetic Data Generation
- **Back-translation**: Use high-resource pivot languages (Somali, Amharic, Swahili) → English via NLLB
  1. Collect English monolingual (news, Wikipedia, Common Crawl)
  2. Translate English → Pivot (Somali) using NLLB
  3. Human-verified pivot → English = parallel; pivot → Rendille can be created via bilingual dictionaries or transferred from related languages if limited native Rendille →
  4. Train backward model: Pivot → Rendille to create synthetic parallel data
- **Round-trip translation**: Use existing English → Somali → Rendille for data augmentation
- **Word-for-word glossing**: Create word-aligned corpus from Bible interlinear resources

---

## 3. Model Architecture

### 3.1 Base Model Options

| Option | Pros | Cons | When to Use |
|--------|------|------|-------------|
| **NLLB-200 (1.3B or 3.3B)** | 200 languages, strong zero-shot; if Rendille (rel) in vocab → direct fine-tuning | May not have Rendille token coverage | First check; if included, fine-tune on Bible + synthetic data |
| **mBART-50 / mT5** | Multilingual encoder-decoder; good for low-resource with language adapters | Limited East Cushitic coverage | If NLLB doesn't support Rendille |
| **From-scratch Transformer** | Full control; can add Rendille tokens | Needs more data; slower convergence | Rich parallel corpus (50K+ pairs) |
| **Inductive bias: Cushitic-specific** | Leverage morphological analyzers (HornMorpho) | Complex integration | Research phase |

**Recommended**: **NLLB-200 1.3B-distilled** with:
- LoRA/PEFT adapters (rank 16, alpha 32) to reduce trainable parameters to ~0.1%–0.3%
- Starting point: Check if Rendille (rel) ∈ NLLB lang list (200 languages). If yes → fine-tune token embedding + adapter. If no → add Rendille tokens to tokenizer, expand embedding layer, and fine-tune.

### 3.2 Cross-lingual Transfer Approach

If Rendille not in NLLB directly:
1. **Pivot through related Cushitic languages**: Somali (som), Oromo (orm), Afar (aaf)
2. **Zero-shot translation**: NLLB supports these languages; use them as bridges
3. **Train multilingual model**: Combine Rendille + Somali + Oromo + Amharic parallel data to create shared representation space
4. **Shared subword vocabulary**: Create joint SentencePiece model (SPM) covering all Cushitic languages + English

---

## 4. Training Pipeline

### Phase 1: Corpus Preparation

1. **Download** eBible / Scripture Earth Rendille-English aligned Bible
   - Extract verse-aligned pairs (vref.txt format)
   - Split into train/val/test by verse ranges
      - Train: Matthew–John (28%) + Acts–Revelation (72%) = ~24K pairs
      - Val: 2,000 pairs (Psalms/Proverbs)
      - Test: 3,000 pairs (remaining books)
2. **Clean & normalize**:
   - Remove footnotes, chapter numbers, punctuation noise
   - Unicode normalization (NFC/NFKC)
   - Strip markup (USFM tags: \c, \v, \q, etc.)
   - Sentence segmentation (Bible verses already split)
3. **Quality filter**:
   - Length ratio between languages (1:1.5 typical)
   - Remove empty/mismatched alignments
   - Language detection validation (fastText)
4. **Pre-tokenize** with SentencePiece or use NLLB's SPM-200 tokenizer

### Phase 2: Base Model & Tokenizer

**If Rendille in NLLB**: Load `facebook/nllb-200-distilled-1.3B` or `nllb-200-3.3B`
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
```

**If Rendille NOT in NLLB**:
1. Augment tokenizer: Add Rendille-specific subwords (train on Bible + monolingual)
2. Expand embedding layer: `model.resize_token_embeddings(len(tokenizer))`
3. Initialize new token embeddings from average of similar languages (Somali/Oromo) or random

### Phase 3: Fine-tuning + LoRA

```python
from peft import LoraConfig, get_peft_model
import torch

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
```

**Training hyperparameters**:
- Batch size: 8–16 (gradient accumulation 4 if needed)
- Learning rate: 3e-4 (LoRA heads) / 1e-5 (full model)
- Epochs: 10–20 (early stopping on validation BLEU)
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: Linear warmup 500 steps, then decay
- Mixed precision: fp16/bf16

### Phase 4: Data Augmentation

Apply **back-translation** to increase corpus size:
1. Train **English→Rendille** model (reverse direction)
2. Translate large English monolingual corpus (Wikipedia, News) to synthetic Rendille
3. Filter synthetic pairs with confidence scores (beam diversity, round-trip consistency)
4. Combine with real parallel data for final training

### Phase 5: Multilingual Fine-tuning (Optional)

If adding Somali/Oromo data:
- Multi-source: Training on `[Somali-English, Oromo-English, Rendille-English]` simultaneously
- Language tags: prepend `>>som<<`, `>>orm<<`, `>>rel<<` to source text per NLLB convention
- Benefits: Shared representation improves low-resource target performance

---

## 5. Evaluation & Metrics

**Primary metrics**:
- **BLEU** (sacrebleu): Standard
- **chrF++**: Character n-gram metric; better for morphologically rich languages
- **COMET**: Neural metric; correlates with human judgment

**Test sets**:
- Held-out Bible verses
- **Domain shift**: Translate other domains (proverbs, health, agriculture) if available
- Human evaluation: Random sample, score fluency/adequacy 1–5

**Baselines**:
- Rule-based/word-by-word transfer (dictionary lookup)
- Google Translate (if it covers Rendille) [likely NOT covered]
- NLLB zero-shot (if Rendille absent) via pivot languages

**Expected Performance**: With ~30K pairs + LoRA + back-translation → BLEU 20–28 achievable; state-of-the-art for low-resource Cushitic.

---

## 6. Deployment Options

### 6.1 Local / API Server
- Package as **Hugging Face pipeline**:
```python
pipe = pipeline("translation", model="./rendille-nllb-lora", tokenizer="./rendille-tokenizer")
```
- **FastAPI** wrapper with endpoint `/translate?text=...&src=rel&tgt=eng`
- **ONNX conversion** for CPU-optimized inference (via `optimum`)

### 6.2 Edge / Mobile
- Convert to **CTranslate2** format (`ct2-transformers-converter`)
- 4x faster inference, smaller size; quantize to int8

### 6.3 Batch Translation
- Script to translate entire books/collections offline

---

## 7. Project Timeline

| Week | Milestone |
|------|-----------|
| 1 | Data collection: Download eBible BibleNLP; extract all Rendille-English pairs; clean & align |
| 2 | Baseline analysis: Check Rendille in NLLB; test zero-shot NLLB output; estimate sentence lengths |
| 3 | Tokenization & preprocessing: Create tokenizer; split train/val/test |
| 4 | Fine-tune base model (LoRA) on Bible subset (~20 tokens/pair) |
| 5 | Evaluate; compute BLEU; identify error patterns (morphology, word order) |
| 6 | Back-translation & synthetic data: Generate 100K English→Rendille pairs |
| 7 | Second fine-tuning with augmented data; early stopping |
| 8 | Human evaluation: Random samples scored for fluency |
| 9 | Package: Export model; FastAPI server; README & usage |
| 10 | Deployment / handoff |

---

## 8. Hardware Requirements

- **Training**: Single GPU (A100 40GB or RTX 4090) sufficient for LoRA finetuning + back-translation (1hr–6hr depending on data size)
- **Storage**: ~10 GB (models + corpora)
- **Inference**: CPU acceptable if quantized (~2–4 sec/sentence)

---

## 9. Known Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Limited parallel corpus (Bible domain only) | Back-translation from English monolingual news/proverbs to expand domain diversity |
| Rendille vocabulary not in NLLB tokenizer | Expand tokenizer with Rendille-specific subwords; leverage related languages for initialization |
| Morphological complexity | Use character-level metrics; consider subword regularization; incorporate morphological analyzer (HornMorpho) for augmentations |
| No standard benchmark | Create held-out test set from non-overlapping Bible books |

---

## 10. Next Steps - Immediate Actions

1. ✅ **Explore BibleNLP corpus** (<24h):
   - Run Python script to list available languages; confirm Rendille presence
   - Download and inspect ~3,000 verse pairs
2. 📊 **Baseline NLLB test** (<12h):
   - Load `facebook/nllb-200-distilled-1.3B`
   - Generate translation for sample Rendille verse
   - Document output quality and tokenization issues
3. 📁 **Create project directory** (5min):
   - Set up `~/rendille-translation/` with `data/`, `scripts/`, `notebooks/`, `models/`
4. 🐍 **Write data preprocessing scripts** (<8h):
   - `download_ebible.py`
   - `clean_bible.py` (remove USFM tags)
   - `create_splits.py` (train/val/test by book)
5. 🚀 **First training run** (<3 days):
   - Fine-tune NLLB-LoRA on ~25K pairs
   - Evaluate on held-out val set
   - Iterate on hyperparameters (LR, rank r)

---

## 11. Resources & References

- eBible Corpus: GitHub `BibleNLP/ebible`, HF `bible-nlp/biblenlp-corpus`
- NLLB: Facebook AI (2022), `facebook/nllb-200-*` models
- Fine-tuning tutorial: https://medium.com/@meinnps/fine-tuning-nllb-200-with-lora-on-a-650-sentence-corpus…
- Bible APIs: bible.com, find.bible, Scripture Earth
- HornMorpho (morphology for Cushitic langs): https://github.com/hltdi/HornMorpho
- JW300 dataset (OPUS): https://opus.nlpl.eu/JW300.php (check for Rendille)

---

**Status**: Ready to start Phase 1 (Data Collection)
