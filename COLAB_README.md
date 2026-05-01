# How to Use Google Colab

## Quick Start (3 steps)

1. **Open the notebook:**
   - Go to https://colab.research.google.com/
   - Click "File" → "Upload notebook"
   - Upload `rendille_colab.ipynb` from this repo

2. **Connect GPU:**
   - Click "Runtime" → "Change runtime type"
   - Select "T4 GPU" (or any GPU) → "Save"
   - Click "Connect" (top-right)

3. **Run all cells:**
   - Click "Runtime" → "Run all"
   - Or press `Ctrl+F9`

## What Happens

| Cell | Action | Time |
|------|--------|------|
| 1–2 | Setup + GPU check | 30 sec |
| 3 | Clone repo | 30 sec |
| 4 | Install packages (~2 GB) | 2–5 min |
| 5 | Environment check | 10 sec |
| 6 | Download Bible text | 1 min |
| 7 | Prepare corpus | 30 sec |
| 8 | **Train translation model** | **2–4 hrs** |
| 9 | Test translation | 10 sec |
| 10 | Launch Gradio demo | 5 sec |

## After Training

- Model saved to `models/rendille-rel/` (persists in Colab session)
- Gradio UI opens with public URL (shareable)
- To keep model after session ends:
  - Download `models/rendille-rel/` folder to your computer
  - Or push to GitHub (already tracked by Git)

## Troubleshooting

**Session disconnects during training?**
- Re-run Cell 8 only (resumes from checkpoint).

**Out of memory (OOM)?**
- Edit Cell 8 → `--batch 2`

**Want to stop early?**
- Interrupt kernel (Ctrl+M I) — checkpoint saved every 100 steps

## Next Steps After Colab

1. Download trained model locally
2. Acquire Rendille audio Bible from MegaVoice
3. Run ASR training (see `asr/` section)
4. Deploy full pipeline
