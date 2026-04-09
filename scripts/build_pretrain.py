"""
Build pretraining corpus from processed legal documents.

Creates a text corpus suitable for continued pretraining of Gemma 4 E2B.
Format includes document delimiters and metadata headers.

Output format:
<bos>LUẬT SỐ 13/2023/QH15 - LUẬT ĐẤT ĐAI

Type: Luật
Authority: Quốc hội
Issue Date: 2023-01-01
Effective Date: 2023-08-01
Status: Còn hiệu lực

Chương I: QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Luật này quy định về chế độ sở hữu đất đai...

<eos>
"""

import json
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option("--input", "-i", default="data/processed/documents.parquet", help="Input Parquet file")
@click.option("--output", "-o", default="data/pretrain/corpus.txt", help="Output corpus file")
@click.option("--include-meta", is_flag=True, help="Include metadata headers")
@click.option("--format", type=click.Choice(["plain", "chatml"]), default="plain", help="Output format")
def main(input: str, output: str, include_meta: bool, format: str):
    """Build pretraining corpus from legal documents."""
    input_path = Path(input)
    output_path = Path(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from {input_path}")
    df = pd.read_parquet(input_path)

    print(f"Processing {len(df)} documents")

    with open(output_path, "w", encoding="utf-8") as f:
        for _, doc in tqdm(df.iterrows(), total=len(df)):
            content = doc.get("content_markdown") or doc.get("content_text", "")
            if not content:
                continue

            if format == "plain":
                # Format with document header
                header_parts = []

                # Title with ID
                title = doc.get("title", "")
                doc_number = doc.get("doc_number", "")
                if doc_number:
                    header_parts.append(f"{doc_number} - {title}")
                else:
                    header_parts.append(title)

                # Metadata
                if include_meta:
                    meta_lines = []
                    if doc.get("doc_type"):
                        meta_lines.append(f"Type: {doc['doc_type']}")
                    if doc.get("issuing_authority"):
                        meta_lines.append(f"Authority: {doc['issuing_authority']}")
                    if doc.get("issue_date"):
                        meta_lines.append(f"Issue Date: {doc['issue_date']}")
                    if doc.get("effective_date"):
                        meta_lines.append(f"Effective Date: {doc['effective_date']}")
                    if doc.get("status"):
                        meta_lines.append(f"Status: {doc['status']}")

                    if meta_lines:
                        header_parts.append("\n" + "\n".join(meta_lines))

                header = "\n".join(header_parts)
                f.write(f"<bos>{header}\n\n{content}\n<eos>\n\n")

            elif format == "chatml":
                # ChatML format for dialogue-style training
                system_prompt = f"""Bạn là một chuyên gia pháp luật Việt Nam. Hãy cung cấp thông tin chính xác về văn bản pháp luật sau:

{doc.get('doc_type', '')} {doc.get('doc_number', '')}: {doc.get('title', '')}
Cơ quan ban hành: {doc.get('issuing_authority', '')}
Ngày ban hành: {doc.get('issue_date', '')}
Tình trạng hiệu lực: {doc.get('status', '')}

Nội dung văn bản:
"""
                f.write(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
                f.write(f"<|im_start|>user\nHãy tóm tắt nội dung chính của văn bản này.<|im_end|>\n")
                f.write(f"<|im_start|>assistant\n{content[:2000]}...<|im_end|>\n\n")

    # Calculate statistics
    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
    total_chars = sum(len(doc.get("content_text", "")) for _, doc in df.iterrows())
    estimated_tokens = total_chars // 2.5  # Rough estimate for Vietnamese

    print(f"\n=== Corpus Statistics ===")
    print(f"Documents: {len(df)}")
    print(f"Output size: {output_size:.1f} MB")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {estimated_tokens:,}")
    print(f"\nSaved to: {output_path}")
    print("\n✅ Pretraining corpus built!")


if __name__ == "__main__":
    main()
