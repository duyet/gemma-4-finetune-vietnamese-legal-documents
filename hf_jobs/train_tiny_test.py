#!/usr/bin/env python3
"""
Tiny model test to verify HF Jobs pipeline works.
Uses smallest possible model (124M params) to test connectivity.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def load_config():
    """Load training configuration from environment variables."""
    config = {
        "base_model": "sshleifer/tiny-gpt2",  # 124M params - minimal download
        "max_seq_length": 512,  # Short for speed
        "batch_size": int(os.getenv("BATCH_SIZE", "4")),
        "gradient_accumulation": int(os.getenv("GRADIENT_ACCUMULATION", "2")),
        "epochs": int(os.getenv("EPOCHS", "1")),
        "learning_rate": float(os.getenv("LEARNING_RATE", "5e-5")),
        "output_dir": os.getenv("OUTPUT_DIR", "./outputs"),
        "hf_username": os.getenv("HF_USERNAME", "duyet"),
        "hf_repo_name": "tiny-gpt2-vietnamese-legal-test",
        "push_to_hub": os.getenv("PUSH_TO_HUB", "true").lower() == "true",
        "dataset_name": os.getenv("DATASET_NAME", "duyet/vietnamese-legal-instruct"),
        "dataset_split": "train",
    }

    config["output_dir"] = "tiny-gpt2-vietnamese-legal-test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_name"] = f"run_{timestamp}"

    return config


def format_dataset_to_text(dataset, tokenizer):
    """Format dataset to text."""
    print("📝 Formatting dataset...")

    # Simple format for tiny model
    def format_conversations(examples):
        texts = []
        for convo in examples["messages"]:
            text = ""
            for msg in convo:
                if msg['role'] == 'user':
                    text += f"User: {msg['content']}\n"
                elif msg['role'] == 'assistant':
                    text += f"Assistant: {msg['content']}\n"
            texts.append(text + "Assistant:")
        return {"text": texts}

    dataset = dataset.map(
        format_conversations,
        batched=True,
        remove_columns=["messages"]
    )

    print("🔤 Tokenizing...")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
    )

    print(f"✅ Processed {len(tokenized_dataset):,} examples")
    return tokenized_dataset


def train(config):
    """Run training with tiny model."""
    print("="*60)
    print("🧪 TINY MODEL TEST - Verifying HF Jobs Works")
    print("="*60)
    print(f"📦 Model: {config['base_model']}")
    print(f"📏 Seq Length: {config['max_seq_length']}")
    print(f"📊 Batch: {config['batch_size']} x {config['gradient_accumulation']}")
    print("="*60)

    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")

    # Load model
    print("\n🤖 Loading tiny model (124M params)...")
    model = AutoModelForCausalLM.from_pretrained(config["base_model"])
    print("✅ Model loaded")

    # Load small subset of dataset for testing
    print(f"\n📂 Loading dataset: {config['dataset_name']}")
    raw_dataset = load_dataset(config["dataset_name"])
    # Use only 100 examples for speed test
    train_dataset = raw_dataset["train"].select(range(min(100, len(raw_dataset["train"]))))
    print(f"✅ Using {len(train_dataset)} examples for test")

    train_dataset = train_dataset.rename_column("conversations", "messages")
    train_dataset = format_dataset_to_text(train_dataset, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        logging_steps=5,
        save_steps=50,
        fp16=True,
        report_to="none",
        save_strategy="no",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        padding="longest",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n🚀 Starting test training...")
    trainer.train()

    # Save
    print(f"\n💾 Saving model...")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    log_history = trainer.state.log_history
    final_loss = log_history[-1].get("loss", "N/A") if log_history else "N/A"

    print(f"\n📊 Test Results:")
    print(f"   Final Loss: {final_loss}")

    info = {
        "base_model": config["base_model"],
        "test_examples": len(train_dataset),
        "final_loss": final_loss,
        "timestamp": datetime.now().isoformat(),
        "run_name": config["run_name"],
        "status": "SUCCESS - HF Jobs pipeline works!",
    }

    info_path = Path(config["output_dir"]) / "test_results.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✅ Test results saved")

    return model, tokenizer, info


def main():
    print("="*60)
    print("🧪 HF Jobs Pipeline Test")
    print("="*60)
    print(f"⏰ Started: {datetime.now().isoformat()}")

    config = load_config()

    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n⚠️  No GPU")

    try:
        model, tokenizer, info = train(config)

        print("\n" + "="*60)
        print("✅ TEST PASSED - HF Jobs Works!")
        print("="*60)
        print(f"⏰ Finished: {datetime.now().isoformat()}")
        print("\nNext: Scale up to real models (Qwen, Llama)")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
