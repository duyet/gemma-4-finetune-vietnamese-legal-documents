#!/usr/bin/env python3
"""
Merge HF dataset with crawled data for larger corpus.

Combines:
1. Base dataset: th1nh0/vietnamese-legal-documents (from HF)
2. Crawled data: from thuvienphapluat.vn (Playwright crawler)

Removes duplicates, merges metadata, creates unified dataset.
"""

import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

import click
import pandas as pd
from tqdm import tqdm


def generate_content_hash(content: str) -> str:
    """Generate hash of content for deduplication."""
    if not content:
        return ""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


@click.command()
@click.option("--hf-data", "-h", default="data/hf_downloaded", help="HF dataset directory")
@click.option("--crawled-data", "-c", default="data/raw", help="Crawled data directory")
@click.option("--output", "-o", default="data/merged", help="Output directory")
def main(hf_data: str, crawled_data: str, output: str):
    """Merge HF dataset with crawled data."""
    hf_path = Path(hf_data)
    crawled_path = Path(crawled_data)
    output_path = Path(output)

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Merging Datasets")
    print("=" * 60)

    # Load HF documents
    hf_file = hf_path / "documents.parquet"
    if hf_file.exists():
        print(f"\n[1/4] Loading HF dataset...")
        df_hf = pd.read_parquet(hf_file)
        print(f"✅ Loaded {len(df_hf)} documents from HF")
    else:
        print(f"⚠️  HF dataset not found at {hf_file}")
        df_hf = pd.DataFrame()

    # Load crawled documents (from Playwright crawler)
    crawled_db = crawled_path / ".playwright_crawler_state.db"
    crawled_jsonl = crawled_path / "playwright_documents.jsonl"

    df_crawled = []

    if crawled_db.exists():
        print(f"\n[2/4] Loading crawled data from DB...")
        conn = sqlite3.connect(crawled_db)

        for row in tqdm(conn.execute("SELECT data FROM documents"), desc="Reading DB"):
            doc = json.loads(row[0])
            df_crawled.append(doc)

        conn.close()
        print(f"✅ Loaded {len(df_crawled)} documents from DB")

    elif crawled_jsonl.exists():
        print(f"\n[2/4] Loading crawled data from JSONL...")
        with open(crawled_jsonl, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading JSONL"):
                try:
                    doc = json.loads(line.strip())
                    df_crawled.append(doc)
                except json.JSONDecodeError:
                    continue
        print(f"✅ Loaded {len(df_crawled)} documents from JSONL")
    else:
        print(f"⚠️  No crawled data found")

    if df_crawled:
        df_crawled = pd.DataFrame(df_crawled)

    # Merge and deduplicate
    print(f"\n[3/4] Merging and deduplicating...")

    if not df_hf.empty and not df_crawled.empty:
        # Combine
        df_combined = pd.concat([df_hf, df_crawled], ignore_index=True)
        print(f"Combined: {len(df_combined)} documents (before dedup)")

        # Deduplicate by content hash
        print("Generating content hashes...")
        df_combined["content_hash"] = df_combined["content_text"].apply(
            lambda x: generate_content_hash(x) if pd.notna(x) else ""
        )

        # Remove duplicates (keep first occurrence -优先 HF)
        df_combined = df_combined.drop_duplicates(subset=["content_hash"], keep="first")
        print(f"After dedup: {len(df_combined)} documents")

        # Remove hash column
        df_combined = df_combined.drop(columns=["content_hash"])

        # Add merge metadata
        df_combined["merge_source"] = df_combined["crawl_source"].apply(
            lambda x: "huggingface" if "huggingface" in str(x) else "crawled"
        )
        df_combined["merged_at"] = datetime.now().isoformat()

    elif not df_hf.empty:
        df_combined = df_hf
        df_combined["merge_source"] = "huggingface"
        df_combined["merged_at"] = datetime.now().isoformat()
        print(f"Using HF dataset only: {len(df_combined)} documents")

    elif not df_crawled.empty:
        df_combined = df_crawled
        df_combined["merge_source"] = "crawled"
        df_combined["merged_at"] = datetime.now().isoformat()
        print(f"Using crawled data only: {len(df_combined)} documents")
    else:
        print("❌ No data to merge!")
        return 1

    # Save merged dataset
    print(f"\n[4/4] Saving merged dataset...")

    # Save documents
    df_combined.to_parquet(output_path / "documents.parquet", index=False)
    print(f"✅ Saved {len(df_combined)} documents")

    # Show statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total documents: {len(df_combined):,}")

    if "merge_source" in df_combined.columns:
        source_counts = df_combined["merge_source"].value_counts()
        print(f"\nBy source:")
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} ({count/len(df_combined)*100:.1f}%)")

    if "doc_type" in df_combined.columns:
        type_counts = df_combined["doc_type"].value_counts().head(10)
        print(f"\nTop document types:")
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count:,}")

    # Build passages from merged dataset
    print(f"\nBuilding passages from merged dataset...")
    passages = []
    passage_id = 0

    for _, doc in tqdm(df_combined.iterrows(), total=len(df_combined), desc="Extracting passages"):
        content = doc.get("content_markdown") or doc.get("content_text", "")
        if not content:
            continue

        # Split by paragraphs
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
                        "source": doc.get("merge_source", "unknown"),
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
                "source": doc.get("merge_source", "unknown"),
            })

    df_passages = pd.DataFrame(passages)
    df_passages.to_parquet(output_path / "passages.parquet", index=False)
    print(f"✅ Created {len(passages):,} passages")

    # Build pretrain corpus
    print(f"\nBuilding pretrain corpus...")
    corpus_file = output_path / "corpus.txt"

    with open(corpus_file, "w", encoding="utf-8") as f:
        for _, doc in tqdm(df_combined.iterrows(), total=len(df_combined), desc="Building corpus"):
            content = doc.get("content_markdown") or doc.get("content_text", "")
            if content:
                title = doc.get("title", "")
                doc_number = doc.get("doc_number", "")
                doc_type = doc.get("doc_type", "")

                header = f"{doc_type} {doc_number} - {title}"
                f.write(f"<bos>{header}\n\n{content}\n<eos>\n\n")

    corpus_size_mb = corpus_file.stat().st_size / (1024 * 1024)
    print(f"✅ Created pretrain corpus: {corpus_size_mb:.1f} MB")

    print(f"\n✅ Merge complete!")
    print(f"Output: {output_path}")
    print(f"\nReady for training!")


if __name__ == "__main__":
    main()
