#!/usr/bin/env python3
"""
Local training script for Mac MPS or NVIDIA CUDA.

This script handles training without Unsloth (which is CUDA-only).
Supports Mac MPS acceleration for Apple Silicon.

Usage:
    uv run python scripts/local_train.py --stage pretrain --max-seq-length 2048
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Gemma 4 locally (Mac/CUDA)")
    parser.add_argument("--stage", type=str, default="pretrain", choices=["pretrain", "sft", "both"],
                        help="Training stage")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length (reduce to 2048 for Mac MPS)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")
    parser.add_argument("--base-model", type=str, default="google/gemma-4-2b-it",
                        help="Base model (use official Gemma 4 for local training)")
    parser.add_argument("--use-4bit", action="store_true", default=False,
                        help="Use 4-bit quantization (CUDA only, not MPS)")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    return parser.parse_args()


def get_output_dir(args) -> str:
    """Generate output directory name."""
    if args.output_dir:
        return args.output_dir

    # Extract model name from path
    # google/gemma-4-2b-it -> gemma-4-2b-vietnamese-legal
    model_name = args.base_model.split('/')[-1]
    # Remove -it suffix if present
    model_name = model_name.replace('-it', '')
    return f"{model_name}-vietnamese-legal"


def check_device():
    """Check available device and show info."""
    print("\n" + "="*60)
    print("DEVICE CHECK")
    print("="*60)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ VRAM: {vram_gb:.1f} GB")
        use_4bit = vram_gb < 12
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"✅ MPS (Mac Metal): Available")
        print(f"   Apple Silicon GPU detected")
        use_4bit = False  # MPS doesn't support 4-bit well
        print(f"⚠️  Note: Using float16 for MPS (4-bit not well supported)")
    else:
        device = "cpu"
        print(f"⚠️  No GPU detected - training on CPU will be SLOW")
        use_4bit = False

    return device, use_4bit


def load_training_data() -> Dataset:
    """Load training dataset from HuggingFace."""
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    print(f"\n📂 Loading dataset from HuggingFace...")
    print("   Dataset: duyet/vietnamese-legal-instruct")

    dataset = load_dataset("duyet/vietnamese-legal-instruct", split="train")
    print(f"✅ Loaded {len(dataset):,} examples")

    return dataset


def build_pretrain_corpus(dataset: Dataset) -> Dataset:
    """Build pretraining corpus from instruction dataset."""
    print("\n📝 Building pretraining corpus...")

    corpus_text = []
    for example in dataset:
        content = None

        # Handle conversations format (from duyet/vietnamese-legal-instruct)
        if "conversations" in example:
            conversations = example.get("conversations", [])
            # Extract all messages and format them
            parts = []
            for msg in conversations:
                role = msg.get("role", "")
                text = msg.get("content", "")
                if role and text:
                    # Map role to Gemma format
                    if role == "user":
                        parts.append(f"<start_of_turn>user\n{text}<end_of_turn>")
                    elif role == "assistant":
                        parts.append(f"<start_of_turn>model\n{text}<end_of_turn>")
                    elif role == "system":
                        parts.append(f"System: {text}")

            if parts:
                content = "\n".join(parts)

        # Fallback: handle instruction-output format
        elif "instruction" in example and "output" in example:
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            input_text = example.get("input", "")

            parts = []
            if input_text:
                parts.append(f"Input: {input_text}")
            if instruction:
                parts.append(f"Instruction: {instruction}")
            if output:
                parts.append(f"Output: {output}")

            content = ". ".join(parts) if parts else instruction

        elif "text" in example:
            content = example.get("text", "")

        elif "messages" in example:
            messages = example.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    break
            else:
                continue

        if content and len(content) > 20:
            corpus_text.append(content + "\n")

    print(f"✅ Built corpus with {len(corpus_text):,} examples")

    total_chars = sum(len(t) for t in corpus_text)
    estimated_tokens = total_chars // 4
    print(f"📊 Estimated tokens: {estimated_tokens:,}")

    return Dataset.from_dict({"text": corpus_text})


def train_pretrain(args, dataset, device, use_4bit):
    """Train continued pretraining stage."""
    print("\n" + "="*60)
    print("STAGE 1: CONTINUED PRETRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"4-bit quantization: {use_4bit}")

    # Build corpus
    train_dataset = build_pretrain_corpus(dataset)

    # Load tokenizer
    print(f"\n🤖 Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Add pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"🤖 Loading model: {args.base_model}")

    if use_4bit and device == "cuda":
        # 4-bit quantization for CUDA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # Full precision or float16 for MPS/CPU
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "mps":
            model = model.to("mps")

    print("✅ Model loaded")

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"✅ LoRA configured (r={args.lora_r}, alpha={args.lora_alpha})")

    # Training config
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=device in ["cuda", "mps"],
        use_mps_backend=(device == "mps"),
        report_to="none",
        save_safetensors=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=device in ["cuda", "mps"],
            report_to="none",
            output_dir=args.output_dir,
        ),
    )

    # Train
    print("\n🚀 Starting training...")
    print(f"   Batch size: {args.batch_size} x {args.gradient_accumulation} grad accum")
    print(f"   Seq length: {args.max_seq_length}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Device: {device}")

    if device == "cpu":
        print("\n⚠️  WARNING: Training on CPU will be VERY slow!")
        print("   Consider using Colab with free T4 GPU instead")

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
        "device": device,
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


def main():
    args = parse_args()
    args.output_dir = get_output_dir(args)

    print("="*60)
    print("🚀 GEMMA 4 VIETNAMESE LEGAL - LOCAL TRAINING")
    print("="*60)
    print(f"Stage: {args.stage}")
    print(f"Base Model: {args.base_model}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Check device
    device, use_4bit = check_device()

    # Override 4bit if requested and supported
    if args.use_4bit and device == "cuda":
        use_4bit = True
    elif args.use_4bit and device != "cuda":
        print("\n⚠️  4-bit quantization only works with CUDA, disabling...")
        use_4bit = False

    # Load dataset
    dataset = load_training_data()

    # Train
    model = None
    tokenizer = None

    if args.stage in ["pretrain", "both"]:
        model, tokenizer = train_pretrain(args, dataset, device, use_4bit)

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
