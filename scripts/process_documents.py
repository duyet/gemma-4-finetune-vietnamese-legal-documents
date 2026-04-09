"""
Process crawled legal documents.

This script:
1. Loads raw JSONL documents from crawler
2. Validates and cleans data
3. Enhances metadata
4. Converts HTML to Markdown (already done by crawler, but validates)
5. Splits long documents into passages for RAG
6. Outputs processed data in Parquet format
"""

import json
from pathlib import Path
from typing import Generator
from datetime import datetime
from collections import Counter

import click
from tqdm import tqdm
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import pandas as pd


def load_jsonl(file_path: Path) -> Generator[dict, None, None]:
    """Load JSONL file and yield documents."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue


def validate_document(doc: dict) -> bool:
    """Validate that document has required fields."""
    required = ["doc_id", "title", "url", "content_text"]
    return all(doc.get(field) for field in required)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Remove control characters except newlines
    text = "".join(c for c in text if c.isprintable() or c in "\n\r\t")
    return text.strip()


def extract_passages(doc: dict, max_tokens: int = 512, overlap: int = 50) -> list[dict]:
    """Split document into passages for RAG retrieval.

    Vietnamese average token: ~2.5 characters per token
    So 512 tokens ≈ 1280 characters
    """
    content = doc.get("content_markdown") or doc.get("content_text", "")
    if not content:
        return []

    # Split by double newlines (paragraphs) first
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    passages = []
    current_passage = []
    current_length = 0
    passage_id = 0

    for para in paragraphs:
        para_length = len(para)

        # If single paragraph is too long, split by sentence
        if para_length > max_tokens * 2.5:
            sentences = [s.strip() + "." for s in para.split(".") if s.strip()]
            for sent in sentences:
                if current_length + len(sent) > max_tokens * 2.5:
                    if current_passage:
                        passages.append({
                            "passage_id": f"{doc['doc_id']}_p{passage_id}",
                            "doc_id": doc["doc_id"],
                            "text": "\n\n".join(current_passage),
                            "passage_index": passage_id,
                        })
                        passage_id += 1
                        current_passage = []
                        current_length = 0
                current_passage.append(sent)
                current_length += len(sent)
        else:
            # Add paragraph to current passage
            if current_length + para_length > max_tokens * 2.5:
                if current_passage:
                    passages.append({
                        "passage_id": f"{doc['doc_id']}_p{passage_id}",
                        "doc_id": doc["doc_id"],
                        "text": "\n\n".join(current_passage),
                        "passage_index": passage_id,
                    })
                    passage_id += 1
                    # Keep overlap for context
                    current_passage = current_passage[-2:] if len(current_passage) > 2 else []
                    current_length = sum(len(p) for p in current_passage)
            current_passage.append(para)
            current_length += para_length

    # Add final passage
    if current_passage:
        passages.append({
            "passage_id": f"{doc['doc_id']}_p{passage_id}",
            "doc_id": doc["doc_id"],
            "text": "\n\n".join(current_passage),
            "passage_index": passage_id,
        })

    return passages


@click.command()
@click.option("--input", "-i", default="data/raw/documents.jsonl", help="Input JSONL file")
@click.option("--output", "-o", default="data/processed/documents.parquet", help="Output Parquet file")
@click.option("--passages-output", "-p", default="data/processed/passages.parquet", help="Passages output file")
@click.option("--max-tokens", default=512, help="Max tokens per passage")
def main(input: str, output: str, passages_output: str, max_tokens: int):
    """Process crawled legal documents."""
    input_path = Path(input)
    output_path = Path(output)
    passages_path = Path(passages_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    passages_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from {input_path}")
    documents = []
    all_passages = []
    errors = []

    for doc in tqdm(list(load_jsonl(input_path))):
        if not validate_document(doc):
            errors.append({"doc_id": doc.get("doc_id"), "error": "Validation failed"})
            continue

        # Clean text fields
        for field in ["title", "content_text", "summary"]:
            if doc.get(field):
                doc[field] = clean_text(doc[field])

        # Validate markdown conversion
        if not doc.get("content_markdown") and doc.get("content_html"):
            doc["content_markdown"] = md(doc["content_html"])

        # Extract passages
        passages = extract_passages(doc, max_tokens=max_tokens)
        all_passages.extend(passages)

        # Add passage count to document
        doc["passage_count"] = len(passages)
        documents.append(doc)

    # Convert to DataFrame and save
    df_docs = pd.DataFrame(documents)
    df_passages = pd.DataFrame(all_passages)

    print(f"\nProcessed {len(documents)} documents")
    print(f"Created {len(all_passages)} passages")
    print(f"Errors: {len(errors)}")

    # Save to Parquet
    df_docs.to_parquet(output_path, index=False)
    df_passages.to_parquet(passages_path, index=False)

    print(f"\nSaved documents to {output_path}")
    print(f"Saved passages to {passages_path}")

    # Print statistics
    print("\n=== Document Statistics ===")
    print(f"Total documents: {len(documents)}")
    print(f"Total passages: {len(all_passages)}")
    print(f"Average passages per doc: {len(all_passages) / len(documents):.1f}")

    doc_types = Counter(doc.get("doc_type", "Unknown") for doc in documents)
    print("\n=== Document Types ===")
    for dtype, count in doc_types.most_common(10):
        print(f"  {dtype}: {count}")

    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()
