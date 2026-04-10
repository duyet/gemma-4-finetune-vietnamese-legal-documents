#!/usr/bin/env python3
"""
Unsloth training with explicit timeouts and robust error handling.
Addresses HF Jobs connectivity issues with timeout guards.
"""

import os
import sys
import json
import signal
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import torch


# Timeout handler
class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


@contextmanager
def time_limit(seconds):
    """Context manager to enforce time limit on operations."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def load_config():
    """Load training configuration from environment variables."""
    config = {
        "base_model": os.getenv("BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct"),
        "max_seq_length": int(os.getenv("MAX_SEQ_LENGTH", "4096")),
        "load_in_4bit": os.getenv("LOAD_IN_4BIT", "true").lower() == "true",
        "lora_r": int(os.getenv("LORA_R", "16")),
        "lora_alpha": int(os.getenv("LORA_ALPHA", "16")),
        "lora_dropout": float(os.getenv("LORA_DROPOUT", "0.05")),
        "batch_size": int(os.getenv("BATCH_SIZE", "2")),
        "gradient_accumulation": int(os.getenv("GRADIENT_ACCUMULATION", "4")),
        "epochs": int(os.getenv("EPOCHS", "1")),
        "learning_rate": float(os.getenv("LEARNING_RATE", "2e-4")),
        "warmup_ratio": float(os.getenv("WARMUP_RATIO", "0.1")),
        "output_dir": os.getenv("OUTPUT_DIR", "./outputs"),
        "hf_username": os.getenv("HF_USERNAME", "duyet"),
        "hf_repo_name": os.getenv("HF_REPO_NAME", "llama-3-2-3b-vietnamese-legal"),
        "push_to_hub": os.getenv("PUSH_TO_HUB", "true").lower() == "true",
        "export_gguf": os.getenv("EXPORT_GGUF", "true").lower() == "true",
        "quantization": os.getenv("QUANTIZATION", "q4_k_m"),
        "dataset_name": os.getenv("DATASET_NAME", "duyet/vietnamese-legal-instruct"),
        "dataset_split": os.getenv("DATASET_SPLIT", "train"),
    }

    model_name = config["base_model"].split('/')[-1]
    model_name = model_name.replace('-it', '')
    config["output_dir"] = f"{model_name}-vietnamese-legal"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_name"] = f"run_{timestamp}"

    return config


def format_dataset_to_text(dataset, tokenizer):
    """Format conversations dataset to text for causal LM training."""
    print("📝 Formatting dataset to text format...")

    def format_conversations(examples):
        texts = []
        for convo in examples["messages"]:
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(
        format_conversations,
        batched=True,
        remove_columns=["messages"]
    )

    print("🔤 Tokenizing dataset...")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=4096,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    columns_to_keep = set(["input_ids", "attention_mask", "labels"])
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove
    )

    print(f"✅ Tokenized {len(tokenized_dataset):,} examples")
    return tokenized_dataset


def train(config):
    """Run training with explicit timeouts."""
    print("="*60)
    print("🚀 UNSLOTH TRAINING WITH TIMEOUTS")
    print("="*60)
    print(f"📦 Base Model: {config['base_model']}")
    print(f"📏 Max Seq Length: {config['max_seq_length']}")
    print(f"📊 Batch Size: {config['batch_size']} x {config['gradient_accumulation']} grad accum")
    print(f"🎯 Learning Rate: {config['learning_rate']}")
    print(f"🔧 LoRA: r={config['lora_r']}, alpha={config['lora_alpha']}")
    print("="*60)

    # Import Unsloth with timeout protection
    print("\n📦 Importing Unsloth...")
    try:
        with time_limit(300):  # 5 minute timeout for imports
            from unsloth import FastLanguageModel
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        print("✅ Unsloth imported successfully")
    except TimeoutError:
        print("❌ Import timed out after 5 minutes")
        print("   This suggests Unsloth installation is hanging")
        raise
    except Exception as e:
        print(f"❌ Import failed: {e}")
        raise

    # Enable modelscope fallback
    os.environ['UNSLOTH_USE_MODELSCOPE'] = '1'
    print("🔄 ModelScope fallback enabled")

    # Load model with explicit timeout
    print(f"\n🤖 Loading model: {config['base_model']}")
    print("   ⏱️  Timeout: 10 minutes")

    model = None
    tokenizer = None
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            # Use timeout to prevent infinite hanging
            with time_limit(600):  # 10 minute timeout per attempt
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config["base_model"],
                    max_seq_length=config["max_seq_length"],
                    dtype=None,
                    load_in_4bit=config["load_in_4bit"],
                    token=os.environ.get("HF_TOKEN"),
                )
            print("✅ Model loaded successfully!")
            break
        except TimeoutError:
            print(f"⚠️  Attempt {attempt + 1}/{max_attempts}: Model loading timed out (10 minutes)")
            if attempt < max_attempts - 1:
                print("   🔄 Retrying in 30 seconds...")
                import time
                time.sleep(30)
            else:
                print("❌ All attempts timed out")
                raise Exception("Model loading failed: All attempts timed out")
        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1}/{max_attempts}: {type(e).__name__}")
            print(f"   Error: {str(e)[:200]}")
            if attempt < max_attempts - 1:
                print("   🔄 Retrying in 60 seconds...")
                import time
                time.sleep(60)
            else:
                print(f"❌ Failed to load model after {max_attempts} attempts")
                raise e

    if model is None or tokenizer is None:
        raise Exception("Model loading failed: No model/tokenizer loaded")

    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load dataset
    print(f"\n📂 Loading dataset: {config['dataset_name']}")
    from datasets import load_dataset

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
        remove_unused_columns=False,
    )

    # Data collator with dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
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
    print("📦 EXPORTING TO GGUF")
    print("="*60)

    gguf_dir = Path(config["output_dir"]) / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Exporting to {gguf_dir} ({config['quantization']})...")
    print("   ⏱️  This may take 20-40 minutes...")

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
    print("📤 UPLOADING TO HUGGINGFACE HUB")
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
    print("🚀 VIETNAMESE LEGAL - UNSLOTH TRAINING WITH TIMEOUTS")
    print("="*60)
    print(f"⏰ Started at: {datetime.now().isoformat()}")

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

    try:
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

        print(f"⏰ Finished at: {datetime.now().isoformat()}")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TRAINING FAILED")
        print("="*60)
        print(f"Error: {e}")
        print(f"⏰ Failed at: {datetime.now().isoformat()}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
