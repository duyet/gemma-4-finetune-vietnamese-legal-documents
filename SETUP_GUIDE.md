# 🚀 Quick Start - End-to-End Pipeline

## Strategy: Use HF Dataset + Build Cloudflare Bypass Crawler

### Phase 1: Setup (One Time)

```bash
# 1. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Install Playwright (for Cloudflare bypass)
uv pip install playwright scrapy-playwright
playwright install chromium
```

### Phase 2: Prepare for Training

```bash
# Option A: Quick start - Use existing HF dataset (no crawl needed)
# The dataset is ready for training!

# Option B: Maximum dataset - Crawl additional data
# Run Playwright crawler (bypasses Cloudflare)
uv run python crawler/playwright_crawler.py --max-pages 50

# This will:
# - Use headless Chromium to bypass Cloudflare
# - Crawl thuvienphapluat.vn with state persistence
# - Save to data/raw/playwright_documents.jsonl
```

### Phase 3: Fine-Tune on Colab (Auto-Notebook)

**Just upload `notebooks/Auto_Train.ipynb` to Colab!**

**Features:**
- ✅ Auto-clones your GitHub repo
- ✅ Installs all dependencies
- ✅ Downloads dataset (no local upload needed)
- ✅ Runs Stage 1 (pretraining)
- ✅ Exports to GGUF
- ✅ Optional: Push to HF

**Configuration (edit in notebook):**
```python
GITHUB_REPO = "https://github.com/YOUR_USERNAME/tvpl"
HF_USERNAME = "YOUR_USERNAME"
MAX_PAGES = 100  # Set 0 to skip crawling
STAGE = "pretrain"  # or "both" for SFT
```

**Benefits:**
- 🔄 **No notebook updates** - Just push code to GitHub
- 🚀 **Always latest** - Clones fresh code each run
- 📦 **All-in-one** - Data + Training + Export
- ⏱️ **Fast** - Downloads HF dataset directly in Colab

### Phase 4: Local RAG

```bash
# After training, download model from Colab
# Extract and run RAG

uv pip install chromadb langchain langchain-community

# Build vector store
uv run python rag/pipeline.py --rebuild

# Interactive Q&A
uv run python rag/pipeline.py --interactive \
    --model gemga4_e2b_tvpl_gguf/ggml-model-q4_k_m.gguf
```

---

## 📊 Dataset Comparison

| Source | Documents | Source Type | Access Method |
|--------|------------|-------------|---------------|
| **HF Dataset** | ~150K | Official gov portal | Direct download in Colab |
| **Crawled Data** | Variable (depends on MAX_PAGES) | thuvienphapluat.vn | Playwright crawler |
| **Combined** | ~150K+ | Both sources | Merge script |

**Recommendation:** Start with HF dataset (fastest), then crawl additional data as needed.

---

## 🎯 Quick Commands

```bash
# Quick test (1 page crawl)
uv run python crawler/playwright_crawler.py --max-pages 1

# Crawl 100 pages (~30 min)
uv run python crawler/playwright_crawler.py --max-pages 100

# Resume from last state
uv run python crawler/playwright_crawler.py --resume

# Check crawler stats
python -c "
import sqlite3, json
conn = sqlite3.connect('data/raw/.playwright_crawler_state.db')
row = conn.execute('SELECT value FROM stats WHERE key=\"global\"').fetchone()
stats = json.loads(row[0]) if row else {}
print(f'Documents: {stats.get(\"documents_extracted\", 0)}')
print(f'Pages: {stats.get(\"pages_crawled\", 0)}')
"
```

---

## 🔧 Troubleshooting

**Playwright issues:**
```bash
# Reinstall browsers
playwright install chromium --force

# Or use system chromium
export PLAYWRIGHT_BROWSERS_PATH=0
```

**Colab OOM:**
- Reduce `MAX_PAGES` to 10-20 for testing
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8

**Crawler stuck:**
- Check if site structure changed
- Try running with `--max-pages 1` to test
- Check Cloudflare hasn't tightened protection

---

## ✅ Ready to Start!

**To begin training right now:**
1. Upload `notebooks/Auto_Train.ipynb` to Colab
2. Edit configuration at top of notebook
3. Run all cells
4. Download model when done

**To crawl additional data:**
```bash
uv run python crawler/playwright_crawler.py --max-pages 50
```

**To train:**
- Use Auto_Train.ipynb in Colab
- Or run `notebooks/01_pretrain.ipynb` with uploaded data

**To deploy:**
- Run RAG pipeline with GGUF model
- Or deploy to HF as API

The pipeline is complete! 🚀
