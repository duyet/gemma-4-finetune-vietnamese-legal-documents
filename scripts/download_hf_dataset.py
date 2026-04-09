#!/usr/bin/env python3
"""
Download and prepare existing HuggingFace dataset.

This script:
1. Downloads th1nhng0/vietnamese-legal-documents from HF
2. Converts to our format
3. Splits into passages
4. Prepares for training
"""

import json
from pathlib import Path
from datetime import datetime

import click
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option("--output", "-o", default="data/hf_downloaded", help="Output directory")
@click.option("--split", "-s", type=click.Choice(["documents", "passages", "pretrain", "sft", "all"]), default="all", help="Which splits to download")
def main(output: str, split: str):
    """Download existing Vietnamese legal dataset from HuggingFace."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading Vietnamese Legal Documents from HuggingFace")
    print("=" * 60)
    print(f"\nSource: th1nhng0/vietnamese-legal-documents")
    print(f"Output: {output_path}")

    # Download metadata and content configs
    if split in ["documents", "all"]:
        print(f"\n[1/4] Downloading metadata and content...")
        try:
            # Download metadata (uses "data" split, not "train")
            metadata = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")
            print(f"✅ Loaded {len(metadata)} metadata records")

            # Download content - handle large_string dtype by downloading parquet directly
            print(f"📥 Loading content (handling large_string dtype)...")
            from huggingface_hub import hf_hub_download

            # Download content.parquet directly to avoid Arrow casting issues
            cache_dir = output_path / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            content_path = hf_hub_download(
                repo_id="th1nhng0/vietnamese-legal-documents",
                filename="data/content.parquet",
                repo_type="dataset",
                local_dir=output_path,
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded content.parquet")

            # Read with pyarrow using large_string support
            import pyarrow.parquet as pq
            import pyarrow as pa

            print("📝 Reading content with proper dtype handling...")
            table = pq.read_table(content_path)
            print(f"✅ Loaded content table with {len(table)} rows")

            # Build content dict
            print("📝 Building content lookup dictionary...")
            content_dict = {}
            for i in tqdm(range(len(table)), desc="Indexing content"):
                row = table.slice(i, 1).to_pydict()
                doc_id = row.get("doc_id", [None])[0]
                if doc_id:
                    content_dict[str(doc_id)] = {
                        "doc_id": str(doc_id),
                        "content_html": row.get("content_html", [""])[0],
                    }

            print(f"✅ Indexed {len(content_dict)} content records")

            # Convert to our format
            our_format = []
            for meta in tqdm(metadata, desc="Merging metadata and content"):
                doc_id = str(meta.get("doc_id", ""))
                content_item = content_dict.get(doc_id, {})

                our_doc = {
                    "url": meta.get("url", ""),
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "doc_number": meta.get("doc_number", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "issuing_authority": meta.get("issuing_authority", ""),
                    "issue_date": meta.get("issue_date", ""),
                    "effective_date": meta.get("effective_date", ""),
                    "status": meta.get("status", ""),
                    "sector": meta.get("sector", ""),
                    "field": meta.get("field", ""),
                    "content_html": content_item.get("content_html", ""),
                    "content_text": "",
                    "content_markdown": "",
                    "category": meta.get("category", ""),
                    "sub_category": meta.get("sub_category", ""),
                    "tags": meta.get("tags", []),
                    "language": "vn",
                    "crawled_at": datetime.now().isoformat(),
                    "crawl_source": "huggingface:th1nhng0/vietnamese-legal-documents",
                }

                # Extract text from HTML
                if our_doc["content_html"]:
                    from bs4 import BeautifulSoup
                    from markdownify import markdownify as md

                    soup = BeautifulSoup(our_doc["content_html"], "lxml")
                    our_doc["content_text"] = soup.get_text(separator="\n", strip=True)
                    our_doc["content_markdown"] = md(our_doc["content_html"])

                our_format.append(our_doc)

            # Save as Parquet
            df = pd.DataFrame(our_format)
            df.to_parquet(output_path / "documents.parquet", index=False)
            print(f"✅ Saved {len(our_format)} documents to {output_path / 'documents.parquet'}")

        except Exception as e:
            print(f"⚠️  Error downloading documents: {e}")

    # Download relationships if available
    if split in ["all"]:
        print(f"\n[2/4] Downloading relationships...")
        try:
            rels = load_dataset("th1nhng0/vietnamese-legal-documents", "relationships", split="data")
            print(f"✅ Loaded {len(rels)} relationships")
            print(f"⚠️  Skipping relationships save - not needed for training")

            # Save as Parquet
            df_rels = pd.DataFrame(rels)
            df_rels.to_parquet(output_path / "relationships.parquet", index=False)
            print(f"✅ Saved relationships to {output_path / 'relationships.parquet'}")

        except Exception as e:
            print(f"⚠️  Error downloading relationships: {e}")

    # Build passages from documents
    if split in ["passages", "all"]:
        print(f"\n[3/4] Building passages...")

        docs_file = output_path / "documents.parquet"
        if not docs_file.exists():
            print("⚠️  Documents not found, skipping passages")
        else:
            df = pd.read_parquet(docs_file)

            passages = []
            passage_id = 0

            for _, doc in tqdm(df.iterrows(), total=len(df), desc="Extracting passages"):
                content = doc.get("content_markdown") or doc.get("content_text", "")
                if not content:
                    continue

                # Split by paragraphs (~512 tokens each)
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

                current_passage = []
                current_length = 0

                for para in paragraphs:
                    if current_length + len(para) > 1500:  # ~512 tokens
                        if current_passage:
                            passages.append({
                                "passage_id": f"{doc['doc_id']}_p{passage_id}",
                                "doc_id": doc["doc_id"],
                                "text": "\n\n".join(current_passage),
                                "passage_index": passage_id,
                                "title": doc.get("title", ""),
                            })
                            passage_id += 1
                            current_passage = []
                            current_length = 0

                    current_passage.append(para)
                    current_length += len(para)

                # Add final passage
                if current_passage:
                    passages.append({
                        "passage_id": f"{doc['doc_id']}_p{passage_id}",
                        "doc_id": doc["doc_id"],
                        "text": "\n\n".join(current_passage),
                        "passage_index": passage_id,
                        "title": doc.get("title", ""),
                    })

            df_passages = pd.DataFrame(passages)
            df_passages.to_parquet(output_path / "passages.parquet", index=False)
            print(f"✅ Created {len(passages)} passages")

    # Build pretrain corpus
    if split in ["pretrain", "all"]:
        print(f"\n[4/4] Building pretrain corpus...")

        docs_file = output_path / "documents.parquet"
        if not docs_file.exists():
            print("⚠️  Documents not found, skipping corpus")
        else:
            df = pd.read_parquet(docs_file)

            corpus_file = output_path / "corpus.txt"
            with open(corpus_file, "w", encoding="utf-8") as f:
                for _, doc in tqdm(df.iterrows(), total=len(df), desc="Building corpus"):
                    content = doc.get("content_markdown") or doc.get("content_text", "")
                    if content:
                        title = doc.get("title", "")
                        doc_number = doc.get("doc_number", "")
                        doc_type = doc.get("doc_type", "")

                        header = f"{doc_type} {doc_number} - {title}"
                        f.write(f"<bos>{header}\n\n{content}\n<eos>\n\n")

            print(f"✅ Created pretrain corpus: {corpus_file}")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    for file in output_path.glob("*"):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.1f} MB")

    print(f"\n✅ Ready for training!")
    print(f"\nTo use with training:")
    print(f"  from datasets import load_dataset")
    print(f"  docs = load_dataset(\"th1nhng0/vietnamese-legal-documents\", \"documents\")")


if __name__ == "__main__":
    main()
