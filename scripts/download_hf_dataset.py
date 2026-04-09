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
from tqdm import tqdm
import polars as pl


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
            print(f"   Sample fields: {list(metadata[0].keys())[:10]}")

            # Download content - read parquet directly with Polars
            print(f"📥 Loading content with Polars...")
            from huggingface_hub import hf_hub_download

            # Download content.parquet
            content_path = hf_hub_download(
                repo_id="th1nhng0/vietnamese-legal-documents",
                filename="data/content.parquet",
                repo_type="dataset",
            )

            # Read with Polars
            content_df = pl.read_parquet(content_path)
            print(f"✅ Loaded content table with {len(content_df)} rows")
            print(f"   Sample fields: {content_df.columns[:10]}")
            print(f"   Sample id: {content_df[0, 'id']}")

            # Build content dict using Polars
            print("📝 Building content lookup dictionary...")
            content_df_filtered = (
                content_df
                .filter(pl.col("id").is_not_null())
                .select([
                    pl.col("id").cast(str),
                    pl.col("content_html")
                ])
            )

            # Create lookup dict - both metadata and content use 'id' field
            content_lookup = dict(zip(
                content_df_filtered["id"].to_list(),
                content_df_filtered["content_html"].to_list()
            ))

            print(f"✅ Indexed {len(content_lookup)} content records")

            # Convert to our format - NOTE: both metadata and content use 'id' field
            our_format = []
            docs_with_content = 0
            docs_without_content = 0

            for meta in tqdm(metadata, desc="Merging metadata and content"):
                # Both metadata and content use 'id' field
                meta_id = str(meta.get("id", ""))
                content_html = content_lookup.get(meta_id, "")

                our_doc = {
                    "url": meta.get("url", ""),
                    "doc_id": meta_id,  # Use 'id' from metadata as doc_id
                    "title": meta.get("title", ""),
                    "doc_number": meta.get("so_ky_hieu", ""),  # Vietnamese field name
                    "doc_type": meta.get("loai_van_ban", ""),  # Vietnamese field name
                    "issuing_authority": "",  # Not in metadata
                    "issue_date": meta.get("ngay_ban_hanh", ""),  # Vietnamese field name
                    "effective_date": meta.get("ngay_co_hieu_luc", ""),  # Vietnamese field name
                    "status": "",  # Will derive from dates
                    "sector": meta.get("nganh", ""),  # Vietnamese field name
                    "field": "",  # Not in metadata
                    "content_html": content_html,
                    "content_text": "",
                    "content_markdown": "",
                    "category": "",
                    "sub_category": "",
                    "tags": [],
                    "language": "vn",
                    "crawled_at": datetime.now().isoformat(),
                    "crawl_source": "huggingface:th1nhng0/vietnamese-legal-documents",
                }

                # Skip text extraction - will be done in build_pretrain stage
                # This makes download ~10x faster
                if our_doc["content_html"]:
                    docs_with_content += 1
                else:
                    docs_without_content += 1

                our_format.append(our_doc)

            print(f"\n📊 Merge statistics:")
            print(f"   Documents with content: {docs_with_content:,}")
            print(f"   Documents without content: {docs_without_content:,}")

            # Save as Parquet using Polars
            df = pl.DataFrame(our_format)
            df.write_parquet(output_path / "documents.parquet")
            print(f"✅ Saved {len(our_format)} documents to {output_path / 'documents.parquet'}")

        except Exception as e:
            print(f"⚠️  Error downloading documents: {e}")

    # Download relationships if available
    if split in ["all"]:
        print(f"\n[2/4] Downloading relationships...")
        try:
            rels = load_dataset("th1nhng0/vietnamese-legal-documents", "relationships", split="data")
            print(f"✅ Loaded {len(rels)} relationships")
            print(f"ℹ️  Skipping relationships save - not needed for training")

        except Exception as e:
            print(f"⚠️  Error downloading relationships: {e}")

    # Build passages from documents
    if split in ["passages", "all"]:
        print(f"\n[3/4] Building passages...")

        docs_file = output_path / "documents.parquet"
        if not docs_file.exists():
            print("⚠️  Documents not found, skipping passages")
        else:
            df = pl.read_parquet(docs_file)

            passages = []
            passage_id = 0

            for doc in tqdm(df.iter_rows(named=True), total=len(df), desc="Extracting passages"):
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

            df_passages = pl.DataFrame(passages)
            df_passages.write_parquet(output_path / "passages.parquet")
            print(f"✅ Created {len(passages)} passages")

    # Build pretrain corpus
    if split in ["pretrain", "all"]:
        print(f"\n[4/4] Building pretrain corpus...")

        docs_file = output_path / "documents.parquet"
        if not docs_file.exists():
            print("⚠️  Documents not found, skipping corpus")
        else:
            df = pl.read_parquet(docs_file)

            corpus_file = output_path / "corpus.txt"
            with open(corpus_file, "w", encoding="utf-8") as f:
                for doc in tqdm(df.iter_rows(named=True), total=len(df), desc="Building corpus"):
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
