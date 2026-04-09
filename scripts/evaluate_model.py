#!/usr/bin/env python3
"""
Evaluate trained model with test questions.

Usage:
    python scripts/evaluate_model.py --model-dir gemma4_e2b_tvpl_pretrain_lora
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from unsloth import FastModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing LoRA adapters (auto-detected if not set)")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--output-file", type=str, default="training_results.json",
                        help="Output file for results")
    return parser.parse_args()


def auto_detect_model_dir():
    """Auto-detect the latest model directory."""
    import glob

    # Look for directories matching the pattern
    pattern = "gemma-*-vi-legal-pretrain"
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No model directory found matching pattern '{pattern}'\n"
            f"Please run training first or specify --model-dir"
        )

    # Sort by modification time, get the latest
    latest = max(matches, key=lambda p: Path(p).stat().st_mtime)
    print(f"🔍 Auto-detected model directory: {latest}")
    return latest


def evaluate_model(args):
    """Evaluate model with test questions."""
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)

    # Check if model exists
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"\n❌ Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    # Load model
    print(f"\n🤖 Loading model from {args.model_dir}")
    model, tokenizer = FastModel.from_pretrained(
        args.model_dir,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print("✅ Model loaded")

    # Test questions for Vietnamese legal domain
    test_questions = [
        "Điều kiện chuyển nhượng quyền sử dụng đất nông nghiệp là gì?",
        "Thời hiệu lực của Luật Đất đai 2023 bắt đầu từ khi nào?",
        "Các trường hợp thu hồi đất theo pháp luật hiện hành?",
        "Quyền sử dụng đất có được thừa kế không?",
        "Thủ tục đăng ký đất đai gồm những bước nào?",
    ]

    print(f"\n📝 Testing with {len(test_questions)} questions\n")
    print("="*60)

    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")

        # Generate answer
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the question from answer if present
        if question in answer:
            answer = answer.replace(question, "").strip()

        print(f"A{i}: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print("-" * 60)

        # Simple quality metrics
        answer_len = len(answer.strip())
        word_count = len(answer.split())

        results.append({
            "question": question,
            "answer": answer,
            "answer_length": answer_len,
            "word_count": word_count,
        })

    # Calculate average metrics
    avg_len = sum(r["answer_length"] for r in results) / len(results)
    avg_words = sum(r["word_count"] for r in results) / len(results)

    # Score based on answer quality (simple heuristic)
    # Good answers should be substantive (>50 chars, >10 words)
    scores = [
        1.0 if r["answer_length"] > 50 and r["word_count"] > 10 else 0.5
        for r in results
    ]
    avg_score = sum(scores) / len(scores)

    print(f"\n📊 Evaluation Summary:")
    print(f"   Average answer length: {avg_len:.0f} chars")
    print(f"   Average word count: {avg_words:.0f} words")
    print(f"   Average score: {avg_score:.2f}/1.00")
    print("="*60)

    # Load training info if available
    training_info = {}
    info_path = model_path / "training_info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            training_info = json.load(f)

    # Prepare results
    output = {
        "average_score": avg_score,
        "average_answer_length": avg_len,
        "average_word_count": avg_words,
        "test_questions_count": len(test_questions),
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "training_info": training_info,
    }

    # Save results
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_path}")

    return output


def main():
    args = parse_args()

    # Auto-detect model directory if not specified
    if args.model_dir is None:
        args.model_dir = auto_detect_model_dir()

    evaluate_model(args)


if __name__ == "__main__":
    main()
