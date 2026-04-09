"""Test metadata and content merge logic."""

import pytest
from bs4 import BeautifulSoup
import polars as pl


def test_merge_metadata_content(sample_metadata, sample_content):
    """Test merging metadata with content using 'id' field."""
    our_format = []
    docs_with_content = 0
    docs_without_content = 0

    for meta in sample_metadata:
        # Both metadata and content use 'id' field
        meta_id = str(meta.get("id", ""))
        content_html = sample_content.get(meta_id, "")

        our_doc = {
            "url": meta.get("url", ""),
            "doc_id": meta_id,
            "title": meta.get("title", ""),
            "doc_number": meta.get("so_ky_hieu", ""),
            "doc_type": meta.get("loai_van_ban", ""),
            "issuing_authority": "",
            "issue_date": meta.get("ngay_ban_hanh", ""),
            "effective_date": meta.get("ngay_co_hieu_luc", ""),
            "content_html": content_html,
            "content_text": "",
        }

        if content_html:
            soup = BeautifulSoup(content_html, "lxml")
            our_doc["content_text"] = soup.get_text(separator="\n", strip=True)
            docs_with_content += 1
        else:
            docs_without_content += 1

        our_format.append(our_doc)

    # Verify merge statistics
    assert len(our_format) == 3
    assert docs_with_content == 2
    assert docs_without_content == 1

    # Verify documents with content
    docs_with_content_list = [d for d in our_format if d["content_text"]]
    assert len(docs_with_content_list) == 2
    assert docs_with_content_list[0]["doc_id"] == "1"
    assert docs_with_content_list[0]["title"] == "Luật Đất đai 2024"
    assert "Full law content" in docs_with_content_list[0]["content_text"]


def test_polars_dataframe_creation(sample_metadata, sample_content):
    """Test creating Polars DataFrame from merged data."""
    our_format = []
    for meta in sample_metadata:
        meta_id = str(meta.get("id", ""))
        content_html = sample_content.get(meta_id, "")

        our_doc = {
            "doc_id": meta_id,
            "title": meta.get("title", ""),
            "doc_number": meta.get("so_ky_hieu", ""),
            "content_html": content_html,
            "content_text": BeautifulSoup(content_html, "lxml").get_text(separator="\n", strip=True) if content_html else "",
        }
        our_format.append(our_doc)

    # Create Polars DataFrame
    df = pl.DataFrame(our_format)

    # Verify DataFrame structure
    assert len(df) == 3
    assert "doc_id" in df.columns
    assert "title" in df.columns
    assert "content_text" in df.columns

    # Verify data types
    assert df.schema["doc_id"] == pl.String
    assert df.schema["title"] == pl.String


def test_polars_write_parquet(sample_metadata, sample_content, output_dir):
    """Test writing and reading Parquet files with Polars."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create test data
    our_format = []
    for meta in sample_metadata:
        meta_id = str(meta.get("id", ""))
        content_html = sample_content.get(meta_id, "")
        our_format.append({
            "doc_id": meta_id,
            "title": meta.get("title", ""),
            "content_html": content_html,
        })

    # Write Parquet
    df = pl.DataFrame(our_format)
    parquet_path = output_dir / "test_documents.parquet"
    df.write_parquet(parquet_path)

    # Verify file exists
    assert parquet_path.exists()

    # Read back and verify
    df_read = pl.read_parquet(parquet_path)
    assert len(df_read) == 3
    assert list(df_read["doc_id"]) == ["1", "2", "3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
