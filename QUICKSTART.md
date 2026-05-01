# Quick Start — Rendille Translation Model

## One-Command Setup (once dependencies installed)

```bash
cd ~/rendille-translation

# 1. Verify everything
python3 scripts/setup_checklist.py

# 2. Download Bible corpus (27K verse pairs)
python3 scripts/download_bible.py

# 3. Prepare train/val/test splits
python3 scripts/prepare_corpus.py

# 4. Fine-tune NLLB-200 (uses LoRA — cheap & fast)
python3 scripts/train_nmt.py --batch 8 --epochs 10
```

## If any step fails…

### "No module named …"
```bash
pip install torch transformers datasets sacrebleu peft accelerate sentencepiece
```
Use GPU? `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### "rel_extract.txt not found"  
The BibleNLP eBible CDN sometimes fails. Manually download:
- Go to https://www.scriptureearth.org/00i-Scripture_Index.php?iso=rel
- Download the **Plain Text Rendille New Testament** (or full Bible)
- Extract and rename the main text file to `data/rel_extract.txt`
- Download English KJV/World English Bible separately, align by verse

### "CUDA out of memory"
Reduce batch size: `--batch 2 --accum 16` (gradient accumulation)

### "Rendille token not in vocabulary"
The training script automatically adds the token; just add `--add-token` flag.

## Expected Output

After 10 epochs (~2–4 hours on RTX 3090/4090):
- Model saved to `models/rendille-rel/`
- BLEU score on val set printed at end
- Inference: `python scripts/translate.py --text "Rendille sentence"` → English

## Notes

- Total cost: ~$0 (uses your local GPU or CPU — slow but free)
- Model size: ~1.3GB base + ~10MB LoRA adapter
- Inference: ~100ms/verse on modern GPU, ~500ms on CPU

Questions? Check PROJECT_PLAN.md for detailed architecture.
