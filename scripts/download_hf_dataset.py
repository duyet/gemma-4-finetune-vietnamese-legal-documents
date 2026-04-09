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
import pandas as pd
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
        print(f"\n[1/3] Downloading metadata...")
        metadata = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")
        print(f"✅ Loaded {len(metadata)} metadata records")

        print(f"\n[2/3] Downloading content (HTML only, no extraction)...")
        from huggingface_hub import hf_hub_download

        # Download content.parquet
        content_path = hf_hub_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            filename="data/content.parquet",
            repo_type="dataset",
        )

        # Read with Polars
        content_df = pl.read_parquet(content_path)
        print(f"✅ Loaded {len(content_df)} content records")

        print(f"\n[3/3] Merging and saving (defer text extraction)...")

        # Use pandas for robust merge (handles heterogeneous data better)
        print("📝 Converting to pandas...")
        meta_pd = metadata.to_pandas()

        # Build content lookup (id -> content_html)
        content_dict = dict(zip(
            content_df["id"].cast(str).to_list(),
            content_df["content_html"].to_list()
        ))

        # Merge: add content_html using pandas
        print("📝 Merging content with metadata...")
        meta_pd["content_html"] = meta_pd["id"].astype(str).map(content_dict)

        # Add missing fields for our format
        meta_pd["issuing_authority"] = ""
        meta_pd["status"] = ""
        meta_pd["field"] = ""
        meta_pd["content_text"] = ""
        meta_pd["content_markdown"] = ""
        meta_pd["category"] = ""
        meta_pd["sub_category"] = ""
        meta_pd["tags"] = [[] for _ in range(len(meta_pd))]
        meta_pd["language"] = "vn"
        meta_pd["crawled_at"] = datetime.now().isoformat()
        meta_pd["crawl_source"] = "huggingface:th1nhng0/vietnamese-legal-documents"

        # Rename Vietnamese fields to our schema
        meta_pd = meta_pd.rename(columns={
            "id": "doc_id",
            "so_ky_hieu": "doc_number",
            "loai_van_ban": "doc_type",
            "ngay_ban_hanh": "issue_date",
            "ngay_co_hieu_luc": "effective_date",
            "nganh": "sector",
        })

        # Reorder columns to match expected schema
        result_pd = meta_pd[[
            "url", "doc_id", "title", "doc_number", "doc_type",
            "issuing_authority", "issue_date", "effective_date", "status",
            "sector", "field", "content_html", "content_text", "content_markdown",
            "category", "sub_category", "tags", "language", "crawled_at", "crawl_source"
        ]]

        # Convert to Polars for fast Parquet write
        print("📝 Converting to Polars for save...")
        result_df = pl.from_pandas(result_pd)

        # Save immediately - NO HTML PARSING
        result_df.write_parquet(output_path / "documents.parquet")
        print(f"✅ Saved {len(result_df)} documents to {output_path / 'documents.parquet'}")

        # Statistics
        docs_with_content = result_df["content_html"].apply(lambda x: bool(x and x.strip())).sum()
        print(f"\n📊 Statistics:")
        print(f"   Total documents: {len(result_df):,}")
        print(f"   Documents with HTML: {docs_with_content:,}")
        print(f"   Documents without HTML: {len(result_df) - docs_with_content:,}")
        print(f"\n✅ Download complete!")
        print(f"\nℹ️  Text extraction will be done in build_pretrain stage")
        print(f"   This approach is ~10x faster and produces cleaner text")

    # Skip other stages in fast mode
    print(f"\n⚡ Skipping passages, corpus, SFT (will build from documents.parquet)")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
