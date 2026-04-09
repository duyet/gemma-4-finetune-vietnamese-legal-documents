# Colab Training Notebooks

This directory contains Google Colab notebooks for fine-tuning Gemma 4 models.

## 📓 Available Notebooks

### Gemma4_Vietnamese_Legal_Train.ipynb (Main - Full Pipeline)

**Complete training notebook with flag-controlled pipeline** - One notebook to do everything!

**Features:**
- ✅ Clone repository automatically
- ✅ Crawl data (optional, controlled by flag)
- ✅ Download/merge datasets
- ✅ Upload to HuggingFace (optional)
- ✅ Fine-tune Gemma 4 E2B
- ✅ Export to GGUF
- ✅ Upload model to HF (optional)
- ✅ Generate evaluation scores
- ✅ Push results to GitHub (optional)

**Control everything via FLAGS in the first cell:**
```python
# Data Pipeline
CRAWL_ENABLED = False  # Set True to crawl
CRAWL_PAGES = 100
DOWNLOAD_HF_DATASET = True
UPLOAD_DATASET_TO_HF = False

# Training
RUN_TRAINING = True
TRAINING_STAGE = "pretrain"  # "pretrain", "sft", "both"

# Export & Upload
EXPORT_TO_GGUF = True
UPLOAD_MODEL_TO_HF = False

# Evaluation
GENERATE_SCORES = True
PUSH_RESULTS_TO_GITHUB = False
```

**Quick Start:**
```bash
# Open Colab and get instructions
./scripts/colab.sh
```

**Features:**
- ✅ Auto-clones latest code from GitHub
- ✅ Installs dependencies automatically
- ✅ Downloads dataset directly (no upload needed)
- ✅ Fine-tunes Gemma 4 E2B with Vietnamese legal data
- ✅ Exports to GGUF format
- ✅ Simple configuration at the top

**Usage:**
1. Open Google Colab: https://colab.research.google.com/
2. Upload `Gemma4_Vietnamese_Legal_Train.ipynb`
3. Edit flags in first cell (set True/False for each step)
4. Connect to T4 GPU (Runtime → Change runtime type)
5. Run all cells (Runtime → Run all)

**Configuration:**
```python
# Edit in first cell
GITHUB_REPO = "https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents"
GITHUB_USERNAME = "duyet"
HF_USERNAME = "duyet"
STAGE = "pretrain"  # or "both"
MAX_PAGES = 0  # Set 0 to skip crawling
```

**What happens:**
1. Clone repository
2. Install dependencies
3. Download HF dataset
4. Build pretraining corpus
5. Train model (Stage 1: Pretrain)
6. Export to GGUF
7. Download model

### 01_pretrain.ipynb & 02_sft_rag.ipynb (Legacy)

Manual training notebooks - use these if you need more control over the training process.

## 🚀 Quick Start

**Fastest path to trained model:**
1. Open Colab
2. Upload `Colab_Train.ipynb`
3. Set `MAX_PAGES = 0` (skip crawling)
4. Set `STAGE = "pretrain"`
5. Run all cells
6. Download model when done (~4-8 hours)

## 💡 Why Notebooks are Gitignored

Jupyter notebooks are binary files that:
- Contain cell outputs (large)
- Cause merge conflicts
- Are hard to review in git

**Best practice:**
- Keep training logic in Python scripts (in `scripts/`)
- Notebooks just call those scripts
- Notebooks are simple and reproducible

## 🔄 Update Workflow

When you update training code:
1. Push changes to GitHub
2. Re-run the notebook in Colab
3. Notebook automatically clones latest code
4. No notebook updates needed!

## 📊 Training Stages

**Stage 1: Continued Pretraining**
- Embeds Vietnamese legal knowledge into Gemma 4
- Uses pretraining corpus from `data/pretrain/`
- Output: LoRA adapters (~50MB)

**Stage 2: SFT for RAG**
- Fine-tunes on Q&A pairs
- Uses SFT data from `data/sft/`
- Output: LoRA adapters for RAG

## 🎯 Expected Results

**Training time (on T4 GPU):**
- Stage 1: ~4-8 hours depending on dataset size
- Stage 2: ~2-4 hours
- GGUF export: ~30 minutes

**Output files:**
- `gemma4_e2b_tvpl_pretrain_lora/` - LoRA adapters
- `gemma4_e2b_tvpl_gguf/` - GGUF quantized model
- `gemma4_tvpl_model.zip` - Downloadable archive

## 🛠️ Troubleshooting

**Colab OOM:**
- Reduce `MAX_SEQ_LENGTH` to 2048
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8

**Slow training:**
- Use smaller dataset for testing
- Reduce `num_train_epochs` to 0.5
- Check GPU is being used (nvidia-smi)

**Download issues:**
- Models are large (2-5GB zipped)
- Use stable internet connection
- If download fails, copy files to Google Drive first

## 📚 Next Steps

After training:
1. Download model from Colab
2. Extract locally
3. Build RAG pipeline: `python rag/pipeline.py --rebuild`
4. Run Q&A: `python rag/pipeline.py --interactive`

---

**Last updated**: 2026-04-09
**Version**: 1.0.0
