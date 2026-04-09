---
name: notebook-flags
description: Complete reference for Gemma4 Vietnamese Legal Training notebook flags
type: reference
---

# Gemma4 Vietnamese Legal Training Notebook - Complete Flag Reference

## File: `notebooks/Gemma4_Vietnamese_Legal_Train.ipynb`

**Purpose**: Single notebook with flag-controlled pipeline for complete training workflow.

## All Available Flags

### Repository Configuration
```python
GITHUB_REPO = "https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents"
GITHUB_USERNAME = "duyet"
HF_USERNAME = "duyet"
HF_DATASET_NAME = "vietnamese-legal-documents"
HF_MODEL_NAME = "gemma-4-vietnamese-legal-rag"
HF_TOKEN = ""  # Optional: Set token here or use Colab secrets (see below)
```

**🔑 HuggingFace Token Setup** (Required for uploading to HF)

You need a Write token to upload datasets and models. Three options:

**Option A: Colab Secrets (Recommended)**
1. In Colab, click �钥匙 icon in left sidebar
2. Add new secret: Name = `HF_TOKEN`, Value = your token
3. Toggle "Notebook access" → ON
4. Leave `HF_TOKEN = ""` in notebook code

**Option B: Set in Notebook**
```python
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Paste your token here
```

**Option C: CLI Login in Notebook**
```python
# Notebook will call this automatically if HF_TOKEN is empty
!huggingface-cli login --token $HF_TOKEN
```

**How to Get Your Token:**
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `gemma4-vietnamese-legal`
4. Type: **Write** (⚠️ Required for uploading)
5. Copy token (shown only once!)

See [docs/HF_SETUP.md](../../docs/HF_SETUP.md) for complete guide.

### Crawling Flags
```python
CRAWL_ENABLED = False  # Enable crawling fresh data from thuvienphapluat.vn
CRAWL_PAGES = 100  # Number of pages to crawl (if enabled)
```

**When to use**: Set `CRAWL_ENABLED = True` when you need additional data beyond the HF dataset.

### Dataset Flags
```python
DOWNLOAD_HF_DATASET = True  # Download base dataset from HF (~150K docs)
MERGE_DATASETS = True  # Merge crawled + HF data (if both exist)
UPLOAD_DATASET_TO_HF = False  # Upload merged dataset to HuggingFace
```

**Workflow**:
1. If `CRAWL_ENABLED = False`: Uses only HF dataset
2. If `CRAWL_ENABLED = True` and `MERGE_DATASETS = True`: Combines both sources
3. If `UPLOAD_DATASET_TO_HF = True`: Pushes to HF (requires HF_TOKEN)

### Training Flags
```python
RUN_TRAINING = True  # Run fine-tuning
TRAINING_STAGE = "pretrain"  # Options: "pretrain", "sft", "both"
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
```

**Training Stages**:
- `pretrain`: Continued pretraining on legal corpus
- `sft`: Supervised fine-tuning for RAG (requires SFT data)
- `both`: Run both stages sequentially

### Export Flags
```python
EXPORT_TO_GGUF = True  # Export to GGUF format
QUANTIZATION = "q4_k_m"  # Options: q4_k_m, q5_k_m, q8_0
```

**Quantization Guide**:
- `q4_k_m`: 4-bit, fastest inference, ~2.5GB
- `q5_k_m`: 5-bit, better quality, ~3GB
- `q8_0`: 8-bit, best quality, ~5GB

### Upload Flags
```python
UPLOAD_MODEL_TO_HF = False  # Upload model to HuggingFace
```

### Evaluation Flags
```python
GENERATE_SCORES = True  # Generate evaluation scores
RUN_BENCHMARKS = False  # Run detailed benchmarks (slower)
```

**Scores Generated**:
- Test questions with sample answers
- Average score calculation
- Results saved to `training_results.json`

### Git Flags
```python
PUSH_RESULTS_TO_GITHUB = False  # Push training results to GitHub
GIT_COMMIT_MESSAGE = "feat: Complete training run"
```

**What Gets Pushed**:
- `training_results.json` (if generated)
- `outputs_pretrain/` (training checkpoints)
- `gemma4_e2b_tvpl_gguf/` (GGUF files)

## Quick Presets

### Fast Test (30 minutes)
```python
CRAWL_ENABLED = False
DOWNLOAD_HF_DATASET = True
RUN_TRAINING = True
TRAINING_STAGE = "pretrain"
MAX_SEQ_LENGTH = 2048  # Reduce for speed
NUM_EPOCHS = 0.5
EXPORT_TO_GGUF = True
GENERATE_SCORES = True
RUN_BENCHMARKS = False
```

### Full Pipeline (Everything)
```python
CRAWL_ENABLED = True
CRAWL_PAGES = 100
DOWNLOAD_HF_DATASET = True
MERGE_DATASETS = True
UPLOAD_DATASET_TO_HF = True
RUN_TRAINING = True
TRAINING_STAGE = "both"
EXPORT_TO_GGUF = True
UPLOAD_MODEL_TO_HF = True
GENERATE_SCORES = True
RUN_BENCHMARKS = True
PUSH_RESULTS_TO_GITHUB = True
```

### Training Only (Skip Crawling)
```python
CRAWL_ENABLED = False
DOWNLOAD_HF_DATASET = True
RUN_TRAINING = True
TRAINING_STAGE = "pretrain"
EXPORT_TO_GGUF = True
GENERATE_SCORES = True
```

### Dataset Processing Only
```python
CRAWL_ENABLED = False
DOWNLOAD_HF_DATASET = True
RUN_TRAINING = False  # Skip training
```

## How to Use

### Step 1: Open Colab
```bash
./scripts/colab.sh
```

### Step 2: Upload Notebook
- In Colab: File → Upload notebook
- Select: `notebooks/Gemma4_Vietnamese_Legal_Train.ipynb`

### Step 3: Edit Flags
- Open first cell
- Set flags to True/False as needed
- Or use a preset from above

### Step 4: Run
- Runtime → Change runtime type → T4 GPU
- Runtime → Run all (or Cmd/Ctrl+F9)

### Step 5: Download
- Model automatically downloads when complete
- File: `gemma4_tvpl_model.zip`

## Output Files

| File | Description | Size |
|------|-------------|------|
| `gemma4_e2b_tvpl_pretrain_lora/` | LoRA adapters | ~50MB |
| `gemma4_e2b_tvpl_gguf/` | GGUF model | ~2.5-5GB |
| `training_results.json` | Evaluation scores | ~5KB |
| `gemma4_tvpl_model.zip` | Complete package | ~2-5GB |

## Dependencies

The notebook installs these automatically:
- `unsloth[colab-new]` - Fast fine-tuning
- `transformers>=4.46.0` - HuggingFace transformers
- `datasets>=3.0.0` - HuggingFace datasets
- `trl>=0.9.0` - Training library
- `playwright` - Browser automation (if crawling)
- `llama.cpp` - GGUF export
- `pandas`, `beautifulsoup4`, `lxml`, `markdownify` - Data processing

## Troubleshooting

### "nbformat error"
- **Problem**: Invalid notebook format
- **Solution**: Notebook recreated with valid JSON format (nbformat: 4, nbformat_minor: 0)

### "ModuleNotFoundError"
- **Problem**: Missing dependency
- **Solution**: Run cell that installs dependencies (Step 2)

### "Colab OOM"
- **Problem**: Out of memory
- **Solution**: Reduce `MAX_SEQ_LENGTH` to 2048, `BATCH_SIZE` to 1

### "Training stuck"
- **Problem**: Training not progressing
- **Solution**: Check if `RUN_TRAINING = True`, check GPU allocation

### "No model downloaded"
- **Problem**: Download cell didn't run
- **Solution**: Ensure `RUN_TRAINING = True` and `EXPORT_TO_GGUF = True`

## Best Practices

1. **Start Small**: Use preset with `NUM_EPOCHS = 0.5` to test
2. **Monitor Progress**: Watch the training output in Colab
3. **Save Often**: Model saves automatically every 50 steps
4. **Check Flags**: Verify flags are set correctly before running
5. **Use Presets**: Copy-paste presets rather than setting flags manually

## Related Files

- **Helper Script**: `scripts/colab.sh` - Opens Colab with instructions
- **Documentation**: `notebooks/README.md` - Detailed notebook guide
- **CLAUDE.md**: Project context for development
- **README.md**: User-facing project documentation

**Why**: This provides complete reference for the main training notebook, making it easy to customize and debug training runs.
