"""
Upload processed dataset to HuggingFace Hub.

Creates a dataset repository with multiple configs:
- documents: Full document corpus with metadata
- passages: Chunked passages for retrieval
- pretrain: Pretraining text corpus
- sft: SFT instruction-tuning pairs
"""

import json
from pathlib import Path

import click
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import pandas as pd


@click.command()
@click.option("--documents", "-d", default="data/processed/documents.parquet", help="Documents Parquet file")
@click.option("--passages", "-p", default="data/processed/passages.parquet", help="Passages Parquet file")
@click.option("--pretrain", "-t", default="data/pretrain/corpus.txt", help="Pretraining corpus file")
@click.option("--sft", "-s", default="data/sft/train.jsonl", help="SFT JSONL file")
@click.option("--repo-id", "-r", required=True, help="HuggingFace repository ID (e.g., username/tvpl-vi-legal)")
@click.option("--private", is_flag=True, help="Make repository private")
def main(documents: str, passages: str, pretrain: str, sft: str, repo_id: str, private: bool):
    """Upload TVPL dataset to HuggingFace Hub."""
    from huggingface_hub import login

    # Check if logged in
    try:
        from huggingface_hub import whoami
        whoami()
    except Exception:
        print("Please login to HuggingFace:")
        print("  huggingface-cli login")
        return

    print(f"Creating dataset repository: {repo_id}")

    datasets = {}

    # Load documents
    if Path(documents).exists():
        print(f"Loading documents from {documents}")
        df_docs = pd.read_parquet(documents)

        # Define features
        features = {
            "doc_id": Value("string"),
            "title": Value("string"),
            "url": Value("string"),
            "doc_number": Value("string"),
            "doc_type": Value("string"),
            "issuing_authority": Value("string"),
            "issue_date": Value("string"),
            "effective_date": Value("string"),
            "status": Value("string"),
            "content_text": Value("string"),
            "content_markdown": Value("string"),
            "category": Value("string"),
            "sub_category": Value("string"),
            "sector": Value("string"),
            "field": Value("string"),
            "tags": Sequence(Value("string")),
            "language": Value("string"),
            "crawled_at": Value("string"),
        }

        datasets["documents"] = Dataset.from_pandas(
            df_docs,
            features=Features(features),
        )
        print(f"  Loaded {len(datasets['documents'])} documents")
    else:
        print(f"Warning: {documents} not found")

    # Load passages
    if Path(passages).exists():
        print(f"Loading passages from {passages}")
        df_passages = pd.read_parquet(passages)

        datasets["passages"] = Dataset.from_pandas(df_passages)
        print(f"  Loaded {len(datasets['passages'])} passages")
    else:
        print(f"Warning: {passages} not found")

    # Load pretrain corpus
    if Path(pretrain).exists():
        print(f"Loading pretrain corpus from {pretrain}")
        with open(pretrain, "r", encoding="utf-8") as f:
            corpus = f.read()

        # Split by document delimiter
        docs_text = [d.strip() for d in corpus.split("<eos>") if d.strip()]

        datasets["pretrain"] = Dataset.from_dict({"text": docs_text})
        print(f"  Loaded {len(datasets['pretrain'])} training documents")
    else:
        print(f"Warning: {pretrain} not found")

    # Load SFT data
    if Path(sft).exists():
        print(f"Loading SFT data from {sft}")
        sft_data = []
        with open(sft, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sft_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        datasets["sft"] = Dataset.from_list(sft_data)
        print(f"  Loaded {len(datasets['sft'])} SFT examples")
    else:
        print(f"Warning: {sft} not found")

    if not datasets:
        print("Error: No datasets found to upload")
        return

    # Create dataset dict
    dataset_dict = DatasetDict(datasets)

    # Create dataset card
    card = """---
language:
- vi
license: cc-by-4.0
pretty_name: Vietnamese Legal Documents
size_categories:
- 10K<n<100K
---

# Vietnamese Legal Documents Dataset

A comprehensive collection of Vietnamese legal documents for training and evaluating language models on legal tasks.

## Dataset Details

- **Source:** thuvienphapluat.vn
- **Language:** Vietnamese (with some English translations)
- **License:** CC BY 4.0
- **Document Types:** Laws (Luật), Decrees (Nghị định), Circulars (Thông tư), Decisions (Quyết định), Resolutions (Nghị quyết), and more

## Dataset Splits

### `documents`
Full document corpus with metadata including:
- Document identification (ID, title, URL)
- Categorization (type, category, sub-category, sector, field)
- Metadata (issuing authority, dates, status)
- Content (plain text and Markdown)
- Relationships (amends, repeals, citations)

### `passages`
Document passages chunked for RAG retrieval (~512 tokens each).

### `pretrain`
Continued pretraining corpus in plain text format with document headers.

### `sft`
Supervised fine-tuning dataset with question-answer pairs for RAG applications.
Format: ShareGPT/chat format with system, user, assistant roles.

## Usage

```python
from datasets import load_dataset

# Load documents
docs = load_dataset("YOUR_USERNAME/vietnamese-legal-docs", "documents")

# Load passages for retrieval
passages = load_dataset("YOUR_USERNAME/vietnamese-legal-docs", "passages")

# Load pretrain corpus
pretrain = load_dataset("YOUR_USERNAME/vietnamese-legal-docs", "pretrain")

# Load SFT dataset
sft = load_dataset("YOUR_USERNAME/vietnamese-legal-docs", "sft")
```

## Citation

```bibtex
@dataset{vietnamese_legal_docs_2026,
  title={Vietnamese Legal Documents Dataset},
  author={Your Name},
  year={2026},
  publisher={HuggingFace},
  howpublished={\\url{https://huggingface.co/datasets/YOUR_USERNAME/vietnamese-legal-docs}}
}
```

## License

Legal documents are public domain under Vietnamese Law on Access to Information (No. 104/2016/QH13). The compiled dataset (schema, processing, curation) is released under CC BY 4.0.

## Acknowledgments

Data sourced from [thuvienphapluat.vn](https://thuvienphapluat.vn).
"""

    # Push to hub
    print(f"\nPushing to HuggingFace Hub: {repo_id}")
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        card_data=card,
        commit_message="Upload Vietnamese legal documents dataset",
    )

    print(f"\n✅ Dataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
