"""
Build SFT (Supervised Fine-Tuning) dataset for RAG.

Generates question-answer pairs from legal documents using:
1. Rule-based extraction (article-based Q&A, definitions, penalties)
2. LLM-generated Q&A (optional, using an LLM API)

Output format: ShareGPT / chat format
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Question with context"},
    {"role": "assistant", "content": "Answer"}
  ]
}
"""

import json
import re
from pathlib import Path
from typing import Generator

import click
import pandas as pd
from tqdm import tqdm


def extract_article_qa(doc: dict) -> list[dict]:
    """Extract article-based Q&A pairs from legal documents.

    Pattern: "Điều X. [Title] -> Content"
    """
    content = doc.get("content_markdown") or doc.get("content_text", "")
    if not content:
        return []

    pairs = []

    # Find all articles (Điều)
    article_pattern = r"Điều (\d+[a-z]?)\.?\s+([^\n]+)\n(.*?)(?=Điều \d+|\n\n\n+|$)"
    matches = re.finditer(article_pattern, content, re.DOTALL)

    for match in matches:
        article_num = match.group(1)
        article_title = match.group(2).strip()
        article_content = match.group(3).strip()

        if not article_content:
            continue

        # Create Q&A pairs
        pairs.append({
            "type": "article_content",
            "question": f"{doc.get('doc_type', '')} {doc.get('doc_number', '')}, điều {article_num} quy định về gì?",
            "context": f"{doc.get('title', '')}\nĐiều {article_num}. {article_title}",
            "answer": f"Điều {article_num} quy định về {article_title.lower()}. {article_content[:500]}...",
        })

        pairs.append({
            "type": "article_title",
            "question": f"Nội dung của {article_title} trong {doc.get('title', '')} là gì?",
            "context": f"{doc.get('title', '')}",
            "answer": article_content[:800],
        })

    return pairs[:5]  # Limit per document to avoid bias


def extract_definition_qa(doc: dict) -> list[dict]:
    """Extract definition-based Q&A pairs."""
    content = doc.get("content_markdown") or doc.get("content_text", "")
    if not content:
        return []

    pairs = []

    # Find definitions
    def_pattern = r"(?:Trong|Hiểu rằng|Theo đó)\s+([^\n]+?)(?:được hiểu là|là|gồm|bao gồm)\s+([^\n.]+)\."
    matches = re.finditer(def_pattern, content, re.IGNORECASE)

    for match in matches:
        term = match.group(1).strip()
        definition = match.group(2).strip()

        pairs.append({
            "type": "definition",
            "question": f"Theo {doc.get('title', '')}, '{term}' được hiểu là gì?",
            "context": f"{doc.get('title', '')}\n{match.group(0)}",
            "answer": f"{term} được hiểu là {definition}.",
        })

    return pairs[:3]


def extract_penalty_qa(doc: dict) -> list[dict]:
    """Extract penalty/violation Q&A pairs."""
    content = doc.get("content_markdown") or doc.get("content_text", "")
    if not content:
        return []

    pairs = []

    # Keywords indicating penalties
    penalty_keywords = ["phạt tiền", "chế tài", "xử phạt", "vi phạm", "mức phạt"]

    if any(keyword in content.lower() for keyword in penalty_keywords):
        # Find sentences with penalties
        sentences = content.split(".")

        for sent in sentences[:10]:  # Check first 10 sentences
            if any(keyword in sent.lower() for keyword in penalty_keywords):
                pairs.append({
                    "type": "penalty",
                    "question": f"Hành vi vi phạm nào bị phạt tiền theo {doc.get('title', '')}?",
                    "context": f"{doc.get('title', '')}\n{sent}",
                    "answer": sent.strip(),
                })
                break

    return pairs


def generate_llm_qa(doc: dict, use_llm: bool = False) -> list[dict]:
    """Generate Q&A pairs using LLM (placeholder for future implementation).

    This can be extended to call Gemini/GPT APIs to generate diverse Q&A.
    """
    if not use_llm:
        return []

    # Placeholder for LLM-based generation
    # TODO: Implement API call to generate Q&A pairs
    return []


def format_sharegpt(doc: dict, qa_pairs: list[dict]) -> dict:
    """Format Q&A pairs as ShareGPT/chat format."""
    conversations = []

    # System prompt
    system_msg = {
        "role": "system",
        "content": "Bạn là trợ lý pháp luật Việt Nam chuyên nghiệp. Hãy trả lời câu hỏi dựa trên văn bản pháp luật được cung cấp, trích dẫn cụ thể điều khoản liên quan."
    }
    conversations.append(system_msg)

    # Add Q&A pairs
    for qa in qa_pairs:
        user_msg = {
            "role": "user",
            "content": f"Dựa vào văn bản sau:\n\n{qa.get('context', '')}\n\nCâu hỏi: {qa['question']}"
        }
        assistant_msg = {
            "role": "assistant",
            "content": qa["answer"]
        }
        conversations.append(user_msg)
        conversations.append(assistant_msg)

    return {
        "conversations": conversations,
        "source": doc.get("doc_id"),
        "metadata": {
            "doc_type": doc.get("doc_type"),
            "doc_number": doc.get("doc_number"),
            "title": doc.get("title"),
        }
    }


@click.command()
@click.option("--input", "-i", default="data/processed/documents.parquet", help="Input Parquet file")
@click.option("--output", "-o", default="data/sft/train.jsonl", help="Output JSONL file")
@click.option("--use-llm", is_flag=True, help="Use LLM to generate additional Q&A (requires API key)")
@click.option("--sample-size", type=int, default=1000, help="Number of documents to process (0 for all)")
def main(input: str, output: str, use_llm: bool, sample_size: int):
    """Build SFT dataset from legal documents."""
    input_path = Path(input)
    output_path = Path(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents from {input_path}")
    df = pd.read_parquet(input_path)

    if sample_size > 0:
        df = df.sample(min(sample_size, len(df)), random_state=42)
        print(f"Sampled {len(df)} documents")

    print(f"Generating Q&A pairs for {len(df)} documents")

    all_pairs = []
    type_counts = {}

    with open(output_path, "w", encoding="utf-8") as f:
        for _, doc in tqdm(df.iterrows(), total=len(df)):
            qa_pairs = []

            # Extract rule-based Q&A
            qa_pairs.extend(extract_article_qa(doc))
            qa_pairs.extend(extract_definition_qa(doc))
            qa_pairs.extend(extract_penalty_qa(doc))

            # Optionally add LLM-generated pairs
            if use_llm:
                qa_pairs.extend(generate_llm_qa(doc, use_llm=True))

            if not qa_pairs:
                continue

            # Track types
            for qa in qa_pairs:
                qa_type = qa.get("type", "unknown")
                type_counts[qa_type] = type_counts.get(qa_type, 0) + 1

            # Format as ShareGPT
            formatted = format_sharegpt(doc, qa_pairs)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            all_pairs.append(formatted)

    print(f"\n=== SFT Dataset Statistics ===")
    print(f"Documents processed: {len(df)}")
    print(f"Total Q&A pairs: {len(all_pairs)}")
    print(f"Average pairs per doc: {len(all_pairs) / len(df):.1f}")

    print("\n=== Q&A Type Distribution ===")
    for qa_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {qa_type}: {count}")

    print(f"\nSaved to: {output_path}")
    print("\n✅ SFT dataset built!")


if __name__ == "__main__":
    main()
