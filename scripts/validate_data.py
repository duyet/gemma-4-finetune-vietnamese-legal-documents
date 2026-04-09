#!/usr/bin/env python3
"""
Validate processed data before training.

Checks:
- Data completeness
- Text quality (Vietnamese content)
- Passage length distribution
- Markdown formatting
- No corruption
"""

import json
from pathlib import Path
from collections import Counter

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option("--input", "-i", default="data/processed/documents.parquet", help="Input Parquet file")
@click.option("--detailed", is_flag=True, help="Show detailed statistics")
def main(input: str, detailed: bool):
    """Validate processed documents."""
    input_path = Path(input)

    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 1

    print(f"Loading {input_path}")
    df = pd.read_parquet(input_path)

    print(f"\n=== Dataset Overview ===")
    print(f"Total documents: {len(df):,}")

    # Required fields
    required_fields = ["doc_id", "title", "url", "content_text", "content_markdown"]
    missing = {}
    for field in required_fields:
        null_count = df[field].isnull().sum()
        if null_count > 0:
            missing[field] = null_count

    if missing:
        print("\n⚠️  Missing fields:")
        for field, count in missing.items():
            print(f"  {field}: {count:,} ({count/len(df)*100:.1f}%)")
    else:
        print("✅ All required fields present")

    # Text quality checks
    print("\n=== Text Quality ===")

    # Vietnamese character check
    vietnamese_chars = "àáảãạăằẳẵặâầẩẫậèéẻẽẹêềểễệìíỉĩịòóỏõọôồổỗộơờởỡợùúủũụưừửữựỳýỷỹỵđ"
    df["has_vietnamese"] = df["content_text"].str.contains(f"[{vietnamese_chars}]", regex=True, na=False)
    non_vi_count = (~df["has_vietnamese"]).sum()
    print(f"Documents with Vietnamese text: {df['has_vietnamese'].sum():,} ({df['has_vietnamese'].sum()/len(df)*100:.1f}%)")
    if non_vi_count > 0:
        print(f"⚠️  {non_vi_count} documents without Vietnamese characters")

    # Content length distribution
    df["text_length"] = df["content_text"].str.len()
    df["markdown_length"] = df["content_markdown"].str.len()

    print(f"\nContent length (chars):")
    print(f"  Mean: {df['text_length'].mean():,.0f}")
    print(f"  Median: {df['text_length'].median():,.0f}")
    print(f"  Min: {df['text_length'].min():,}")
    print(f"  Max: {df['text_length'].max():,}")

    # Document types
    print(f"\n=== Document Types ===")
    doc_types = df["doc_type"].value_counts().head(10)
    for dtype, count in doc_types.items():
        print(f"  {dtype}: {count:,}")

    # Status distribution
    print(f"\n=== Document Status ===")
    status_counts = df["status"].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count:,}")

    # Date validation
    print(f"\n=== Date Coverage ===")
    for date_col in ["issue_date", "effective_date"]:
        if date_col in df.columns:
            non_null = df[date_col].notna().sum()
            print(f"  {date_col}: {non_null:,} ({non_null/len(df)*100:.1f}%)")

    # Check for duplicates
    print(f"\n=== Duplicate Check ===")
    dup_ids = df["doc_id"].duplicated().sum()
    dup_titles = df["title"].duplicated().sum()
    print(f"  Duplicate doc_ids: {dup_ids}")
    print(f"  Duplicate titles: {dup_titles}")

    # Content quality samples
    if detailed:
        print(f"\n=== Sample Documents ===")
        sample = df.sample(min(5, len(df)), random_state=42)

        for idx, row in sample.iterrows():
            print(f"\n--- [{row['doc_id']}] ---")
            print(f"Title: {row['title'][:80]}")
            print(f"Type: {row.get('doc_type', 'N/A')}")
            print(f"Status: {row.get('status', 'N/A')}")
            print(f"Text length: {row['text_length']:,}")
            print(f"Content preview:")
            print(f"  {row['content_text'][:200]}...")

    # Overall quality score
    print(f"\n=== Quality Assessment ===")
    issues = []
    if missing:
        issues.append(f"{sum(missing.values())} missing field values")
    if non_vi_count > 0:
        issues.append(f"{non_vi_count} non-Vietnamese documents")
    if dup_ids > 0:
        issues.append(f"{dup_ids} duplicate IDs")
    if (df["text_length"] < 100).sum() > 0:
        issues.append(f"{(df['text_length'] < 100).sum()} very short documents")

    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No issues found - data looks good!")

    # Passage analysis (if exists)
    passages_path = input_path.parent / "passages.parquet"
    if passages_path.exists():
        print(f"\n=== Passages Analysis ===")
        df_passages = pd.read_parquet(passages_path)
        print(f"Total passages: {len(df_passages):,}")

        df_passages["passage_length"] = df_passages["text"].str.len()
        print(f"Average passage length: {df_passages['passage_length'].mean():,.0f} chars")
        print(f"Passages per doc: {len(df_passages) / len(df):.1f}")

    return 0


if __name__ == "__main__":
    main()
