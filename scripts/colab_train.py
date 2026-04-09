#!/usr/bin/env python3
"""
Colab training script for Gemma 4 Vietnamese Legal Documents.

This script handles the complete training pipeline:
- Load dataset
- Fine-tune Gemma 4 E2B with Unsloth
- Save LoRA adapters

Usage:
    python scripts/colab_train.py --stage pretrain --max-seq-length 4096 --batch-size 2
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset, load_dataset
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Gemma 4 for Vietnamese Legal RAG")
    parser.add_argument("--stage", type=str, default="pretrain", choices=["pretrain", "sft", "both"],
                        help="Training stage: pretrain, sft, or both")
    parser.add_argument("--data-dir", type=str, default="data/hf_downloaded",
                        help="Directory containing training data")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for LoRA adapters (auto-generated from base model if not set)")
    parser.add_argument("--base-model", type=str, default="unsloth/gemma-4-E2B-it",
                        help="Base model to fine-tune")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    return parser.parse_args()


def get_output_dir(args) -> str:
    """Generate output directory name from base model."""
    if args.output_dir:
        return args.output_dir

    # Extract model name from path
    # unsloth/gemma-4-E2B-it -> gemma-4-E2B-it-vi-legal-pretrain
    model_name = args.base_model.split('/')[-1]
    return f"{model_name}-vi-legal-pretrain"


def check_gpu():
    """Check if GPU is available and show info."""
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("⚠️  ERROR: NO GPU DETECTED!")
        print("="*60)
        print("\nYou need to enable GPU in Colab:")
        print("1. Runtime → Change runtime type")
        print("2. Select 'T4 GPU'")
        print("3. Click Save")
        print("4. Runtime → Restart session")
        print("\nTraining cannot continue without GPU.")
        print("="*60)
        sys.exit(1)

    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ VRAM: {vram_gb:.1f} GB")

    if vram_gb < 12:
        print(f"\n⚠️  WARNING: Low VRAM detected")
        print(f"   Recommended: --batch-size 1 --max-seq-length 2048")


def load_training_data(data_dir: str) -> Dataset:
    """Load training dataset from HF format or processed data."""
    data_path = Path(data_dir)

    # Try loading from HF dataset format first
    if (data_path / "dataset_info.json").exists():
        print(f"\n📂 Loading dataset from {data_dir}")
        dataset = load_dataset(str(data_dir), split="train")
        print(f"✅ Loaded {len(dataset):,} examples")
        return dataset

    # Try loading from processed parquet
    parquet_files = list(data_path.glob("**/*.parquet"))
    if parquet_files:
        print(f"\n📂 Loading from parquet files")
        import pandas as pd
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(df):,} documents")
        return Dataset.from_pandas(df)

    raise FileNotFoundError(f"No dataset found in {data_dir}")


def build_pretrain_corpus(dataset: Dataset) -> Dataset:
    """Build pretraining corpus from document dataset."""
    print("\n📝 Building pretraining corpus...")

    corpus_text = []
    for doc in dataset:
        content = doc.get('content_html') or doc.get('content_text', '')
        if content and len(content) > 100:  # Skip very short documents
            title = doc.get('title', '')
            doc_number = doc.get('so_ky_hieu', '') or doc.get('doc_number', '')
            header = f"{doc_number} - {title}".strip(" -")
            corpus_text.append(f"<bos>{header}\n\n{content}<eos>")

    print(f"✅ Built corpus with {len(corpus_text):,} documents")

    # Estimate tokens
    total_chars = sum(len(t) for t in corpus_text)
    estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
    print(f"📊 Estimated tokens: {estimated_tokens:,}")

    return Dataset.from_dict({"text": corpus_text})


def train_pretrain(args, dataset: Dataset):
    """Train continued pretraining stage."""
    print("\n" + "="*60)
    print("STAGE 1: CONTINUED PRETRAINING")
    print("="*60)

    # Build corpus
    train_dataset = build_pretrain_corpus(dataset)

    # Load model
    print(f"\n🤖 Loading model: {args.base_model}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print("✅ Model loaded")

    # Configure LoRA
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing="unsloth",
    )
    print(f"✅ LoRA configured (r={args.lora_r}, alpha={args.lora_alpha})")

    # Training config
    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir=args.output_dir,
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    # Train
    print("\n🚀 Starting training...")
    print(f"   Batch size: {args.batch_size} x {args.gradient_accumulation} grad accum")
    print(f"   Seq length: {args.max_seq_length}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.learning_rate}")

    trainer.train()

    # Save
    print(f"\n💾 Saving LoRA adapters to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Show metrics
    log_history = trainer.state.log_history
    if log_history:
        final_loss = log_history[-1].get('loss', 'N/A')
        print(f"\n📊 Training Summary:")
        print(f"   Final loss: {final_loss}")
        print(f"   Steps: {trainer.state.global_step}")

    # Save training info
    info = {
        "stage": "pretrain",
        "base_model": args.base_model,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "steps": trainer.state.global_step,
        "final_loss": log_history[-1].get('loss', None) if log_history else None,
        "timestamp": datetime.now().isoformat(),
    }

    info_path = Path(args.output_dir) / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✅ Training info saved to {info_path}")

    return model, tokenizer


def train_sft(args, dataset: Dataset):
    """Train SFT stage for RAG."""
    print("\n" + "="*60)
    print("STAGE 2: SFT FOR RAG")
    print("="*60)
    print("⚠️  SFT stage not yet implemented")
    print("   This would fine-tune on Q&A pairs with retrieved context")
    return None, None


def main():
    args = parse_args()

    # Generate output directory from base model if not specified
    args.output_dir = get_output_dir(args)

    print("="*60)
    print("🚀 GEMMA 4 VIETNAMESE LEGAL TRAINING")
    print("="*60)
    print(f"Stage: {args.stage}")
    print(f"Base Model: {args.base_model}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Check GPU
    check_gpu()

    # Load dataset
    dataset = load_training_data(args.data_dir)

    # Train based on stage
    model = None
    tokenizer = None

    if args.stage in ["pretrain", "both"]:
        model, tokenizer = train_pretrain(args, dataset)

    if args.stage in ["sft", "both"]:
        # For SFT, we would load the pretrain checkpoint first
        if args.stage == "both":
            # Continue from pretrain
            pass
        model, tokenizer = train_sft(args, dataset)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Output: {args.output_dir}")
    print("\nNext steps:")
    print("1. Export to GGUF: python scripts/export_gguf.py")
    print("2. Upload to HF: python scripts/upload_model.py")
    print("="*60)


if __name__ == "__main__":
    main()
