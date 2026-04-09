# TVPL Quick Start Guide

## Phase 1: Crawl thuvienphapluat.vn

```bash
# Install dependencies
pip install -r requirements.txt

# Run crawler (polite mode: 2.5s delay between requests)
cd crawler
scrapy crawl tvpl

# Output saved to: data/raw/documents.jsonl
# State saved to: data/raw/.spider_state.json
# Resume capability built-in
```

**Expected crawl time:** 3-7 days (polite rate limiting)

**Monitoring:**
```bash
# Check document count
wc -l data/raw/documents.jsonl

# View sample
head -1 data/raw/documents.jsonl | jq
```

## Phase 2: Process Data

```bash
# Convert to Parquet, extract passages, validate markdown
python scripts/process_documents.py

# Build pretraining corpus
python scripts/build_pretrain.py

# Build SFT Q&A pairs
python scripts/build_sft.py

# Upload to HuggingFace
python scripts/upload_to_hf.py --repo-id YOUR_USERNAME/vietnamese-legal-docs
```

## Phase 3: Fine-tune on Colab

1. Upload `data/pretrain/corpus.txt` to Colab
2. Open `notebooks/01_pretrain.ipynb` in Colab
3. Select T4 GPU runtime
4. Run all cells
5. Download checkpoints

**Or** if dataset is on HuggingFace:
- Just update `YOUR_USERNAME` in notebook and run directly

## Phase 4: RAG Pipeline (Local)

Install dependencies:
```bash
pip install chromadb llama-cpp-python langchain
```

Run RAG:
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

# Vietnamese embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# Load fine-tuned model
llm = LlamaCpp(
    model_path="gemma4_e2b_vi_law_rag_gguf/ggml-model-q4_k_m.gguf",
    n_ctx=4096,
    temperature=0.7,
)

# Vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# RAG query
query = "Điều kiện chuyển nhượng đất nông nghiệp?"
docs = vectorstore.similarity_search(query, k=3)
context = "\n\n".join(d.page_content for d in docs)

answer = llm(f"Dựa vào các văn bản sau:\n\n{context}\n\nCâu hỏi: {query}")
print(answer)
```

## Troubleshooting

**Crawler stuck:**
- Check `data/raw/.spider_state.json` for last page
- Restart with `scrapy crawl tvpl -a resume=True`

**Markdown conversion failed:**
- Install: `pip install markdownify`
- The crawler already handles this, but you can re-run:
  ```python
  from markdownify import markdownify as md
  from bs4 import BeautifulSoup

  soup = BeautifulSoup(html_content, "lxml")
  markdown = md(str(soup))
  ```

**Colab OOM:**
- Reduce `max_seq_length` to 2048
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8

**GGUF export error:**
- Ensure llama.cpp is cloned
- Use smaller quantization: `q4_0` instead of `q4_k_m`

## Next Steps

1. **Start crawling** — it takes time, run in background
2. **Process in parallel** — don't wait for crawl to finish
3. **Train on Colab** — free T4 is sufficient for E2B
4. **Local RAG** — use Ollama or llama.cpp for inference

**Resources:**
- [Unsloth Gemma 4 Guide](https://unsloth.ai/docs/models/gemma-4/train)
- [Vietnamese NLP with underthesea](https://github.com/undertheseanlp)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
