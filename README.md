# TVPL - Vietnamese Legal Documents Dataset for RAG

Crawl, process, and prepare Vietnamese legal documents from thuvienphapluat.vn for fine-tuning Gemma 4 E2B for RAG applications.

## Project Structure

```
tvpl/
├── crawler/              # Scrapy crawler for thuvienphapluat.vn
├── data/                 # Raw and processed data
├── scripts/              # Data processing utilities
├── notebooks/            # Colab training notebooks
└── requirements.txt      # Python dependencies
```

## Quick Start

### 1. Install with UV (recommended)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### 2. Run the crawler

```bash
# Test first
uv run tvpl-test

# Run crawler
uv run tvpl-crawl
```

### 3. Process data

```bash
python scripts/process_documents.py
python scripts/build_pretrain.py
python scripts/build_sft.py
```

### 4. Fine-tune on Google Colab

Upload and run `notebooks/01_pretrain.ipynb` and `notebooks/02_sft_rag.ipynb`

## Dataset

After crawling and processing, the dataset includes:
- **Documents**: 50,000–200,000+ Vietnamese legal documents
- **Metadata**: Title, number, type, authority, dates, status
- **Content**: Full text with document structure preserved
- **Q&A Pairs**: 50,000–100,000 instruction pairs for SFT

## License

Legal documents are public domain under Vietnamese Law on Access to Information (No. 104/2016/QH13). The compiled dataset is released under CC BY 4.0.

## Source

Data sourced from [thuvienphapluat.vn](https://thuvienphapluat.vn).

## Citation

```bibtex
@dataset{tvpl2026,
  title={TVPL - Vietnamese Legal Documents Dataset for RAG},
  author={Your Name},
  year={2026},
  publisher={HuggingFace},
  howpublished={\\url{https://huggingface.co/datasets/your-username/tvpl-vi-legal}}
}
```
