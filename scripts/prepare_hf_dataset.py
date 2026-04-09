#!/usr/bin/env python3
"""
Prepare complete dataset for HuggingFace Hub upload.

Creates a properly formatted dataset with:
- Multiple configs (documents, passages, pretrain, sft)
- Dataset card (README.md)
- Data files (Parquet, JSONL, TXT)
- Metadata and licensing

Output structure:
├── data/
│   ├── documents/
│   │   ├── train-00000-of-00001.parquet
│   │   └── dataset_info.json
│   ├── passages/
│   │   ├── train-00000-of-00001.parquet
│   │   └── dataset_info.json
│   ├── pretrain/
│   │   ├── train-00000-of-00001.txt
│   │   └── dataset_info.json
│   └── sft/
│       ├── train-00000-of-00001.jsonl
│       └── dataset_info.json
└── README.md
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import HfApi


@click.command()
@click.option("--input-dir", "-i", default="data/processed", help="Input directory with processed data")
@click.option("--output-dir", "-o", default="data/hf_dataset", help="Output directory for HF dataset")
@click.option("--repo-id", "-r", required=True, help="Target repo ID (e.g., username/tvpl-vi-legal)")
@click.option("--upload", is_flag=True, help="Upload to HuggingFace Hub after preparing")
def main(input_dir: str, output_dir: str, repo_id: str, upload: bool):
    """Prepare TVPL dataset for HuggingFace Hub."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print("=" * 60)
    print("TVPL Dataset Preparation for HF")
    print("=" * 60)

    # Clean and create output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare each config
    configs = {}

    # 1. Documents config
    docs_file = input_path / "documents.parquet"
    if docs_file.exists():
        print(f"\n[1/4] Processing documents...")
        docs_df = pd.read_parquet(docs_file)

        # Select and rename columns for HF
        docs_columns = {
            "doc_id": "string",
            "title": "string",
            "url": "string",
            "doc_number": "string",
            "doc_type": "string",
            "issuing_authority": "string",
            "signatory": "string",
            "signatory_title": "string",
            "issue_date": "string",
            "effective_date": "string",
            "expiry_date": "string",
            "published_date": "string",
            "status": "string",
            "effect_status": "string",
            "sector": "string",
            "field": "string",
            "scope": "string",
            "content_html": "string",
            "content_text": "string",
            "content_markdown": "string",
            "summary": "string",
            "related_docs": "list[string]",
            "amends_docs": "list[string]",
            "amended_by_docs": "list[string]",
            "repeals_docs": "list[string]",
            "repealed_by_docs": "list[string]",
            "cites_docs": "list[string]",
            "cited_by_docs": "list[string]",
            "tags": "list[string]",
            "language": "string",
            "has_english_version": "bool",
            "original_doc_url": "string",
            "category": "string",
            "sub_category": "string",
            "crawled_at": "string",
            "crawl_source": "string",
        }

        # Keep only available columns
        available_cols = [c for c in docs_columns.keys() if c in docs_df.columns]
        docs_df = docs_df[available_cols]

        # Convert list columns
        list_cols = ["related_docs", "amends_docs", "amended_by_docs", "repeals_docs",
                     "repealed_by_docs", "cites_docs", "cited_by_docs", "tags"]
        for col in list_cols:
            if col in docs_df.columns:
                docs_df[col] = docs_df[col].apply(lambda x: x if isinstance(x, list) else [])

        # Create features
        features_dict = {}
        for col, dtype in docs_columns.items():
            if col not in docs_df.columns:
                continue
            if dtype == "list[string]":
                features_dict[col] = Sequence(Value("string"))
            elif dtype == "bool":
                features_dict[col] = Value("bool")
            else:
                features_dict[col] = Value("string")

        docs_dataset = Dataset.from_pandas(docs_df, features=Features(features_dict))
        docs_split = DatasetDict({"train": docs_dataset})

        docs_output = output_path / "documents"
        docs_split.save_to_disk(str(docs_output))
        configs["documents"] = len(docs_dataset)
        print(f"  ✅ {len(docs_dataset):,} documents")

        # Generate dataset_info.json
        info = {
            "splits": {"train": {"num_examples": len(docs_dataset)}},
            "features": {k: str(v) for k, v in features_dict.items()}
        }
        with open(docs_output / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

    # 2. Passages config
    passages_file = input_path / "passages.parquet"
    if passages_file.exists():
        print(f"\n[2/4] Processing passages...")
        passages_df = pd.read_parquet(passages_file)

        passages_dataset = Dataset.from_pandas(passages_df)
        passages_split = DatasetDict({"train": passages_dataset})

        passages_output = output_path / "passages"
        passages_split.save_to_disk(str(passages_output))
        configs["passages"] = len(passages_dataset)
        print(f"  ✅ {len(passages_dataset):,} passages")

        info = {
            "splits": {"train": {"num_examples": len(passages_dataset)}},
            "features": {k: str(v.dtype) for k, v in passages_df.items()}
        }
        with open(passages_output / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

    # 3. Pretrain config
    pretrain_file = input_path.parent / "pretrain" / "corpus.txt"
    if pretrain_file.exists():
        print(f"\n[3/4] Processing pretrain corpus...")
        with open(pretrain_file, "r", encoding="utf-8") as f:
            corpus_text = f.read()

        # Split by document delimiter
        docs_text = [d.strip() for d in corpus_text.split("<eos>") if d.strip()]

        pretrain_dataset = Dataset.from_dict({"text": docs_text})
        pretrain_split = DatasetDict({"train": pretrain_dataset})

        pretrain_output = output_path / "pretrain"
        pretrain_split.save_to_disk(str(pretrain_output))
        configs["pretrain"] = len(pretrain_dataset)
        print(f"  ✅ {len(pretrain_dataset):,} training documents")

        info = {
            "splits": {"train": {"num_examples": len(pretrain_dataset)}},
            "features": {"text": "large_string"}
        }
        with open(pretrain_output / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

    # 4. SFT config
    sft_file = input_path.parent / "sft" / "train.jsonl"
    if sft_file.exists():
        print(f"\n[4/4] Processing SFT data...")
        sft_data = []
        with open(sft_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sft_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        sft_dataset = Dataset.from_list(sft_data)
        sft_split = DatasetDict({"train": sft_dataset})

        sft_output = output_path / "sft"
        sft_split.save_to_disk(str(sft_output))
        configs["sft"] = len(sft_dataset)
        print(f"  ✅ {len(sft_dataset):,} SFT examples")

        info = {
            "splits": {"train": {"num_examples": len(sft_dataset)}},
            "features": {"conversations": "list[dict]", "source": "string", "metadata": "dict"}
        }
        with open(sft_output / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

    # Create dataset card (README.md)
    print(f"\n[5/5] Creating dataset card...")
    readme = create_dataset_card(repo_id, configs)
    with open(output_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"  ✅ README.md created")

    # Summary
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}")
    print(f"\nConfigs prepared:")
    for config, count in configs.items():
        print(f"  - {config}: {count:,} examples")

    print(f"\nDataset structure:")
    print(f"  {output_path}/")
    for config in configs.keys():
        print(f"    ├── {config}/")
        print(f"    │   ├── dataset_info.json")
        print(f"    │   └── train/")
    print(f"    └── README.md")

    # Upload if requested
    if upload:
        print(f"\nUploading to HuggingFace Hub: {repo_id}")
        try:
            from huggingface_hub import login
            login()

            api = HfApi()
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=False
            )

            # Upload each config
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()

            for config_name in configs.keys():
                config_path = output_path / config_name
                print(f"  Uploading {config_name}...")
                fs.put(
                    f"{config_path}/*",
                    f"datasets/{repo_id}/{config_name}",
                    recursive=True
                )

            # Upload README
            with open(output_path / "README.md", "r") as f:
                readme_content = f.read()
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )

            print(f"\n✅ Dataset uploaded successfully!")
            print(f"   View at: https://huggingface.co/datasets/{repo_id}")

        except Exception as e:
            print(f"\n❌ Upload failed: {e}")
            print(f"\nTo upload manually:")
            print(f"  1. Create repo: huggingface-cli repo create {repo_id} --type dataset")
            print(f"  2. Upload: huggingface-cli upload {output_path} {repo_id}")

    return 0


def create_dataset_card(repo_id: str, configs: dict) -> str:
    """Generate dataset README.md content."""
    return f"""---
language:
- vi
- en
license: cc-by-4.0
pretty_name: Vietnamese Legal Documents for RAG
size_categories:
- n<10K
- 10K<n<100K
task_categories:
- text-generation
- question-answering
- retrieval-augmented-generation
tags:
- legal
- vietnamese
- law
- legislation
- documents
- rag
---

# Vietnamese Legal Documents Dataset for RAG

A comprehensive collection of Vietnamese legal documents sourced from thuvienphapluat.vn, designed for training and evaluating RAG (Retrieval-Augmented Generation) systems with Gemma 4 and other language models.

## 📋 Dataset Overview

| Config | Examples | Description |
|--------|----------|-------------|
| `documents` | {configs.get('documents', 0):,} | Full legal documents with rich metadata |
| `passages` | {configs.get('passages', 0):,} | Chunked passages for retrieval (~512 tokens) |
| `pretrain` | {configs.get('pretrain', 0):,} | Text corpus for continued pretraining |
| `sft` | {configs.get('sft', 0):,} | Q&A pairs for supervised fine-tuning |

## 📚 Dataset Description

### Documents
Complete Vietnamese legal documents including:
- **Luật** (Laws)
- **Nghị quyết** (Resolutions)
- **Nghị định** (Decrees)
- **Thông tư** (Circulars)
- **Quyết định** (Decisions)
- And more...

Each document includes:
- **Identification**: doc_id, title, URL, document number
- **Metadata**: Type, issuing authority, dates (issue/effective/expiry), status
- **Categorization**: Category, sub-category, sector, field
- **Content**: HTML, plain text, and Markdown formats
- **Relationships**: Amendments, repeals, citations (when available)

### Passages
Documents split into ~512 token passages for RAG retrieval, preserving context boundaries.

### Pretrain
Continued pretraining corpus with document headers and delimiters:
```
<bos>LUẬT SỐ 13/2023/QH15 - LUẬT ĐẤT ĐAI

Type: Luật
Authority: Quốc hội
Issue Date: 2023-01-01
Effective Date: 2023-08-01
Status: Còn hiệu lực

Chương I: QUY ĐỊNH CHUNG
<eos>
```

### SFT
Question-answer pairs in ShareGPT format for training RAG capabilities:
```json
{{
  "conversations": [
    {{"role": "system", "content": "Bạn là trợ lý pháp luật..."}},
    {{"role": "user", "content": "Dựa vào văn bản sau...\\n\\nCâu hỏi: ..."}},
    {{"role": "assistant", "content": "Answer..."}}
  ]
}}
```

## 🚀 Usage

### Load Documents
```python
from datasets import load_dataset

docs = load_dataset("{repo_id}", "documents")
print(docs["train"][0])
```

### Load Passages for RAG
```python
passages = load_dataset("{repo_id}", "passages")
for passage in passages["train"]:
    print(passage["text"])
```

### Load Pretrain Corpus
```python
corpus = load_dataset("{repo_id}", "pretrain")
for doc in corpus["train"]:
    print(doc["text"])
```

### Load SFT Data
```python
sft = load_dataset("{repo_id}", "sft")
for example in sft["train"]:
    for msg in example["conversations"]:
        print(f"{{msg['role']}}: {{msg['content'][:100]}}...")
```

## 🏗️ Dataset Structure

### Documents Schema
```python
{{
    "doc_id": str,              # Unique identifier
    "title": str,                # Document title
    "url": str,                  # Source URL
    "doc_number": str,           # Số ký hiệu
    "doc_type": str,             # Loại văn bản
    "issuing_authority": str,    # Cơ quan ban hành
    "issue_date": str,           # Ngày ban hành
    "effective_date": str,       # Ngày có hiệu lực
    "status": str,               # Tình trạng hiệu lực
    "content_text": str,         # Plain text content
    "content_markdown": str,     # Markdown format
    "category": str,             # Category from URL
    "sub_category": str,         # Sub-category
    "related_docs": List[str],   # Related document IDs
    # ... and 15+ more fields
}}
```

## 🔬 Use Cases

1. **RAG Systems**: Build legal Q&A assistants
2. **Legal Research**: Semantic search over Vietnamese laws
3. **Document Analysis**: Extract legal relationships
4. **Model Training**: Fine-tune Gemma 4 for Vietnamese legal domain
5. **Translation**: Parallel Vietnamese-English legal texts

## 📊 Statistics

- **Source**: thuvienphapluat.vn
- **Language**: Vietnamese (some English translations)
- **Time Period**: 1945 – present
- **Document Types**: 15+ types
- **Issuing Authorities**: 500+ organizations

## ⚖️ License

Vietnamese legal documents are **public domain** under:
- Law on Access to Information (No. 104/2016/QH13)
- Law on Promulgation of Legal Documents (No. 64/2025/QH15)

The compiled dataset (schema, processing, curation) is released under **CC BY 4.0**.

## 📝 Citation

```bibtex
@dataset{{vietnamese_legal_docs_2026,
  title={{Vietnamese Legal Documents Dataset for RAG}},
  author={{Your Name}},
  year={{2026}},
  publisher={{HuggingFace}},
  howpublished={{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```

## 🙏 Acknowledgments

Data sourced from [thuvienphapluat.vn](https://thuvienphapluat.vn).

## 📧 Contact

For questions or issues, please open a discussion on the HuggingFace Hub.

---

**Last updated**: {datetime.now().strftime("%Y-%m-%d")}
**Version**: 1.0.0
"""


if __name__ == "__main__":
    main()
