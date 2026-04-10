#!/usr/bin/env python3
"""
Simplified Unsloth training script for HF Jobs uv run.

This script follows the pattern from Unsloth Jobs blog:
https://huggingface.co/blog/unsloth-jobs

Usage with HF Jobs:
    hf jobs uv run uv_train.py \\
        --flavor t4-medium \\
        --dataset duyet/vietnamese-legal-instruct \\
        --output-repo duyet/gemma-4-E2B-vietnamese-legal \\
        --base-model unsloth/gemma-4-E2B-it
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


def main():
    print("="*60)
    print("🚀 Gemma 4 Vietnamese Legal - Unsloth Training")
    print("="*60)

    # Configuration from command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train with Unsloth on HF Jobs")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--output-repo", type=str, required=True, help="Output model repo")
    parser.add_argument("--base-model", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Base model")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--quantization", type=str, default="q4_k_m", help="GGUF quantization")
    args = parser.parse_args()

    # Parse output repo to get components
    # Format: username/repo-name
    username, repo_name = args.output_repo.split("/")

    # Extract model name for output directory
    model_name = args.base_model.split('/')[-1].replace('-it', '')
    output_dir = f"{model_name}-vietnamese-legal"

    print(f"\n📋 Configuration:")
    print(f"   Base Model: {args.base_model}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Output: {args.output_repo}")
    print(f"   Batch Size: {args.batch_size} x {args.gradient_accumulation} grad accum")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print("="*60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {vram_gb:.1f} GB")
    else:
        print(f"\n⚠️  No GPU detected")

    # Load model
    print(f"\n🤖 Loading model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print("✅ Model loaded")

    # Load and prepare dataset
    print(f"\n📂 Loading dataset: {args.dataset}")
    raw_dataset = load_dataset(args.dataset)
    train_dataset = raw_dataset["train"]

    print(f"✅ Loaded {len(train_dataset):,} examples")

    # Configure chat template
    print("\n📝 Configuring chat template...")
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "model"},
    )

    def format_conversations(examples):
        """Format conversations using Gemma chat template."""
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            ) for convo in convos
        ]
        return {"text": texts}

    # Apply formatting
    print("Formatting conversations...")
    columns_to_keep = ["text"]
    columns_to_remove = [col for col in train_dataset.column_names if col not in columns_to_keep]
    train_dataset = train_dataset.map(
        format_conversations,
        batched=True,
        remove_columns=columns_to_remove
    )

    print(f"✅ Formatted {len(train_dataset):,} examples")

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.print_trainable_parameters()

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train
    print("\n🚀 Starting training...")
    trainer.train()

    # Save LoRA adapters
    print(f"\n💾 Saving LoRA adapters to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Get training metrics
    log_history = trainer.state.log_history
    final_loss = log_history[-1].get("loss", "N/A") if log_history else "N/A"
    global_step = trainer.state.global_step

    print(f"\n📊 Training Summary:")
    print(f"   Steps: {global_step}")
    print(f"   Final Loss: {final_loss}")

    # Save training info
    info = {
        "base_model": args.base_model,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "steps": global_step,
        "final_loss": final_loss,
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
    }

    info_path = Path(output_dir) / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✅ Training info saved to {info_path}")

    # Export to GGUF
    print("\n" + "="*60)
    print("EXPORTING TO GGUF")
    print("="*60)

    gguf_dir = Path(output_dir) / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Exporting to {gguf_dir} ({args.quantization})...")
    print("   This may take 20-40 minutes...")

    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=args.quantization,
    )

    print("✅ Export complete")

    # Show generated files
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if gguf_files:
        print(f"\n📁 Generated GGUF files:")
        for f in gguf_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name} ({size_mb:.1f} MB)")

    # Upload to Hub
    print("\n" + "="*60)
    print("UPLOADING TO HUGGINGFACE HUB")
    print("="*60)
    print(f"📤 Uploading to: {args.output_repo}")

    # Upload LoRA adapters
    print("   Uploading LoRA adapters...")
    model.push_to_hub(args.output_repo)
    tokenizer.push_to_hub(args.output_repo)

    # Upload GGUF
    print("   Uploading GGUF files...")
    model.push_to_hub_gguf(args.output_repo, tokenizer)

    print(f"✅ Uploaded to: https://huggingface.co/{args.output_repo}")

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
