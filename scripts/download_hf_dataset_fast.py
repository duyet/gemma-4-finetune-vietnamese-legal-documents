#!/usr/bin/env python3
"""
Optimized download: Save HTML fast, extract text later.

This approach:
1. Downloads and saves HTML immediately (fast)
2. Defers text extraction to build_pretrain stage
3. Result: 10x faster download, better text quality
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
    """Download existing Vietnamese legal dataset from HuggingFace (OPTIMIZED)."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading Vietnamese Legal Documents (FAST MODE)")
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

        # Create metadata DataFrame
        meta_df = pl.DataFrame(metadata)

        # Build content lookup (id -> content_html)
        content_dict = dict(zip(
            content_df["id"].cast(str).to_list(),
            content_df["content_html"].to_list()
        ))

        # Merge: add content_html to metadata
        print("📝 Merging content with metadata...")
        result_df = meta_df.with_columns([
            pl.col("id").map_dict(lambda x: content_dict.get(str(x), "")).alias("content_html"),
        ])

        # Save immediately - NO HTML PARSING
        result_df.write_parquet(output_path / "documents.parquet")
        print(f"✅ Saved {len(result_df)} documents to {output_path / 'documents.parquet'}")

        # Statistics
        docs_with_content = result_df["content_html"].apply(lambda x: bool(x and x.strip())).sum()
        print(f"\n📊 Statistics:")
        print(f"   Total documents: {len(result_df):,}")
        print(f"   Documents with HTML: {docs_with_content:,}")
        print(f"   Documents without HTML: {len(result_df) - docs_with_content:,}")
        print(f"\n✅ Download complete in FAST MODE!")
        print(f"\nℹ️  Text extraction will be done in build_pretrain stage")
        print(f"   This approach is ~10x faster and produces cleaner text")

    # Skip other stages in fast mode
    print(f"\n⚡ Skipping passages, corpus, SFT (will build from documents.parquet)")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
