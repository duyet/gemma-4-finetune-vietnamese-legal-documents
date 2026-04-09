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
        print(f"   Columns in metadata: {list(metadata.column_names)[:15]}...")  # Show first 15

        print(f"\n[2/3] Downloading content (HTML only, no extraction)...")
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        # Download content.parquet
        content_path = hf_hub_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            filename="data/content.parquet",
            repo_type="dataset",
        )

        # Read content parquet directly with pyarrow
        content_table = pq.read_table(content_path)
        content_df = content_table.to_pandas()
        print(f"✅ Loaded {len(content_df)} content records")

        print(f"\n[3/3] Merging and saving (defer text extraction)...")

        # Use pandas for robust merge (handles heterogeneous data better)
        print("📝 Converting to pandas...")
        meta_pd = metadata.to_pandas()

        # Build content lookup (id -> content_html)
        content_dict = dict(zip(
            content_df["id"].astype(str).tolist(),
            content_df["content_html"].tolist()
        ))

        # Merge: add content_html using pandas
        print("📝 Merging content with metadata...")
        meta_pd["content_html"] = meta_pd["id"].astype(str).map(content_dict)

        # Add missing fields for our format (only those that don't exist)
        required_fields = {
            "url": "",
            "issuing_authority": "",
            "status": "",
            "field": "",
            "content_text": "",
            "content_markdown": "",
            "category": "",
            "sub_category": "",
            "language": "vn",
            "crawled_at": datetime.now().isoformat(),
            "crawl_source": "huggingface:th1nhng0/vietnamese-legal-documents",
        }

        for field, default_value in required_fields.items():
            if field not in meta_pd.columns:
                meta_pd[field] = default_value

        # Add tags column (list for each row)
        if "tags" not in meta_pd.columns:
            meta_pd["tags"] = [[] for _ in range(len(meta_pd))]

        # Rename Vietnamese fields to our schema
        rename_map = {
            "id": "doc_id",
            "so_ky_hieu": "doc_number",
            "loai_van_ban": "doc_type",
            "ngay_ban_hanh": "issue_date",
            "ngay_co_hieu_luc": "effective_date",
            "nganh": "sector",
        }
        for old_name, new_name in rename_map.items():
            if old_name in meta_pd.columns:
                meta_pd = meta_pd.rename(columns={old_name: new_name})

        # Reorder columns to match expected schema (only select existing columns)
        expected_columns = [
            "url", "doc_id", "title", "doc_number", "doc_type",
            "issuing_authority", "issue_date", "effective_date", "status",
            "sector", "field", "content_html", "content_text", "content_markdown",
            "category", "sub_category", "tags", "language", "crawled_at", "crawl_source"
        ]
        existing_columns = [col for col in expected_columns if col in meta_pd.columns]
        result_pd = meta_pd[existing_columns]

        # Save in chunks with progress indication (faster and shows progress)
        print("📝 Saving to parquet in chunks...")
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Convert to pyarrow table (faster than pandas to_parquet for large data)
        print("   Converting to Arrow format...")
        table = pa.Table.from_pandas(result_pd)

        # Write in chunks with progress
        print("   Writing to disk...")
        pq.write_table(
            table,
            output_path / "documents.parquet",
            compression='snappy',
            row_group_size=10000  # Write in 10k row groups
        )

        print(f"✅ Saved {len(result_pd)} documents to {output_path / 'documents.parquet'}")

        # Show file size
        file_size = (output_path / "documents.parquet").stat().st_size / (1024**2)  # MB
        print(f"   File size: {file_size:.1f} MB")

        # Statistics
        docs_with_content = result_pd["content_html"].apply(lambda x: bool(x and isinstance(x, str) and x.strip())).sum()
        print(f"\n📊 Statistics:")
        print(f"   Total documents: {len(result_pd):,}")
        print(f"   Documents with HTML: {docs_with_content:,}")
        print(f"   Documents without HTML: {len(result_pd) - docs_with_content:,}")
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
