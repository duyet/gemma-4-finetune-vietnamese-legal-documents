#!/usr/bin/env python3
"""
Unsloth native training script (bypassing TRL SFTTrainer).
Uses HuggingFace Trainer directly with Unsloth optimizations.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


def load_config():
    """Load training configuration from environment variables."""
    config = {
        # Model
        "base_model": os.getenv("BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct"),
        "max_seq_length": int(os.getenv("MAX_SEQ_LENGTH", "4096")),
        "load_in_4bit": os.getenv("LOAD_IN_4BIT", "true").lower() == "true",

        # LoRA
        "lora_r": int(os.getenv("LORA_R", "16")),
        "lora_alpha": int(os.getenv("LORA_ALPHA", "16")),
        "lora_dropout": float(os.getenv("LORA_DROPOUT", "0.05")),

        # Training
        "batch_size": int(os.getenv("BATCH_SIZE", "2")),
        "gradient_accumulation": int(os.getenv("GRADIENT_ACCUMULATION", "4")),
        "epochs": int(os.getenv("EPOCHS", "1")),
        "learning_rate": float(os.getenv("LEARNING_RATE", "2e-4")),
        "warmup_ratio": float(os.getenv("WARMUP_RATIO", "0.1")),

        # Output
        "output_dir": os.getenv("OUTPUT_DIR", "./outputs"),
        "hf_username": os.getenv("HF_USERNAME", "duyet"),
        "hf_repo_name": os.getenv("HF_REPO_NAME", "llama-3-2-3b-vietnamese-legal"),
        "push_to_hub": os.getenv("PUSH_TO_HUB", "true").lower() == "true",
        "export_gguf": os.getenv("EXPORT_GGUF", "true").lower() == "true",
        "quantization": os.getenv("QUANTIZATION", "q4_k_m"),

        # Dataset
        "dataset_name": os.getenv("DATASET_NAME", "duyet/vietnamese-legal-instruct"),
        "dataset_split": os.getenv("DATASET_SPLIT", "train"),
    }

    # Generate proper output directory name
    model_name = config["base_model"].split('/')[-1]
    model_name = model_name.replace('-it', '')
    config["output_dir"] = f"{model_name}-vietnamese-legal"

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_name"] = f"run_{timestamp}"

    return config


def format_dataset_to_text(dataset, tokenizer):
    """Format conversations dataset to text for causal LM training."""
    print("Formatting dataset to text format...")

    def format_conversations(examples):
        """Format conversations using chat template."""
        texts = []
        for convo in examples["messages"]:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    # Apply formatting - convert messages to text using chat template
    dataset = dataset.map(
        format_conversations,
        batched=True,
        remove_columns=["messages"]
    )

    # Now tokenize the text for causal LM training
    print("Tokenizing dataset...")

    def tokenize_function(examples):
        """Tokenize text for causal LM training."""
        # Tokenize with truncation and padding
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=4096,
            padding=False,  # Dynamic padding in collator
            return_tensors=None,  # Return lists for dataset.map
        )
        # For causal LM, labels are the same as input_ids
        # Create a proper deep copy
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    # Remove all columns except the tokenized ones
    # Keep only: input_ids, attention_mask, labels
    columns_to_keep = set(["input_ids", "attention_mask", "labels"])
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove  # Remove all non-tokenized columns
    )

    print(f"✅ Tokenized {len(tokenized_dataset):,} examples")
    print(f"✅ Columns: {tokenized_dataset.column_names}")
    return tokenized_dataset


def train(config):
    """Run training with Unsloth + HuggingFace Trainer."""
    print("="*60)
    print("UNSLOTH NATIVE TRAINING (No TRL)")
    print("="*60)
    print(f"Base Model: {config['base_model']}")
    print(f"Max Seq Length: {config['max_seq_length']}")
    print(f"Batch Size: {config['batch_size']} x {config['gradient_accumulation']} grad accum")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"LoRA: r={config['lora_r']}, alpha={config['lora_alpha']}")
    print("="*60)

    # Load model
    print(f"\n🤖 Loading model: {config['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )
    print("✅ Model loaded")

    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load dataset
    print(f"\n📂 Loading dataset: {config['dataset_name']}")
    raw_dataset = load_dataset(config["dataset_name"])
    train_dataset = raw_dataset[config["dataset_split"]]
    print(f"✅ Loaded {len(train_dataset):,} examples")

    # Rename conversations to messages
    train_dataset = train_dataset.rename_column("conversations", "messages")

    # Format to text
    train_dataset = format_dataset_to_text(train_dataset, tokenizer)
    print(f"✅ Formatted {len(train_dataset):,} examples")

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        save_strategy="steps",
        remove_unused_columns=False,  # Keep 'text' column for causal LM
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n🚀 Starting training...")
    trainer.train()

    # Save
    print(f"\n💾 Saving to {config['output_dir']}")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # Get training metrics
    log_history = trainer.state.log_history
    final_loss = log_history[-1].get("loss", "N/A") if log_history else "N/A"
    global_step = trainer.state.global_step

    print(f"\n📊 Training Summary:")
    print(f"   Steps: {global_step}")
    print(f"   Final Loss: {final_loss}")

    # Save training info
    info = {
        "base_model": config["base_model"],
        "max_seq_length": config["max_seq_length"],
        "batch_size": config["batch_size"],
        "gradient_accumulation": config["gradient_accumulation"],
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "steps": global_step,
        "final_loss": final_loss,
        "timestamp": datetime.now().isoformat(),
        "run_name": config["run_name"],
    }

    info_path = Path(config["output_dir"]) / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✅ Training info saved to {info_path}")

    return model, tokenizer, info


def export_gguf(config, model, tokenizer):
    """Export model to GGUF format."""
    if not config["export_gguf"]:
        return

    print("\n" + "="*60)
    print("EXPORTING TO GGUF")
    print("="*60)

    gguf_dir = Path(config["output_dir"]) / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Exporting to {gguf_dir} ({config['quantization']})...")
    print("   This may take 20-40 minutes...")

    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=config["quantization"],
    )

    print("✅ Export complete")

    # Show generated files
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if gguf_files:
        print(f"\n📁 Generated GGUF files:")
        for f in gguf_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name} ({size_mb:.1f} MB)")


def push_to_hub(config):
    """Upload model to HuggingFace Hub."""
    if not config["push_to_hub"]:
        return

    print("\n" + "="*60)
    print("UPLOADING TO HUGGINGFACE HUB")
    print("="*60)

    repo_id = f"{config['hf_username']}/{config['hf_repo_name']}"
    print(f"📤 Uploading to: {repo_id}")

    # Upload LoRA adapters
    print("   Uploading LoRA adapters...")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    # Upload GGUF if exists
    gguf_dir = Path(config["output_dir"]) / "gguf"
    if gguf_dir.exists():
        print("   Uploading GGUF files...")
        model.push_to_hub_gguf(repo_id, tokenizer)

    print(f"✅ Uploaded to: https://huggingface.co/{repo_id}")


def main():
    print("="*60)
    print("🚀 LLAMA-3.2 VIETNAMESE LEGAL - UNSLOTH NATIVE TRAINING")
    print("="*60)

    # Load configuration
    config = load_config()

    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {vram_gb:.1f} GB")
    else:
        device = "cpu"
        print(f"\n⚠️  No GPU detected - training on CPU (SLOW!)")

    # Train
    model, tokenizer, info = train(config)

    # Export to GGUF
    export_gguf(config, model, tokenizer)

    # Upload to Hub
    push_to_hub(config)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Output: {config['output_dir']}")

    if config["push_to_hub"]:
        repo_id = f"{config['hf_username']}/{config['hf_repo_name']}"
        print(f"Model: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
