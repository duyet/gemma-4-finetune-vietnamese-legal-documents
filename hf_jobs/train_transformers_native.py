#!/usr/bin/env python3
"""
Plain Transformers training (no Unsloth).
Avoids Unsloth import issues on HF Jobs.
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
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_config():
    """Load training configuration from environment variables."""
    config = {
        "base_model": os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        "max_seq_length": int(os.getenv("MAX_SEQ_LENGTH", "2048")),
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
        "hf_repo_name": os.getenv("HF_REPO_NAME", "qwen-2.5-3b-vietnamese-legal"),
        "push_to_hub": os.getenv("PUSH_TO_HUB", "true").lower() == "true",
        "export_gguf": os.getenv("EXPORT_GGUF", "false").lower() == "true",  # GGUF requires Unsloth
        "quantization": os.getenv("QUANTIZATION", "q4_k_m"),
        "dataset_name": os.getenv("DATASET_NAME", "duyet/vietnamese-legal-instruct"),
        "dataset_split": os.getenv("DATASET_SPLIT", "train"),
    }

    model_name = config["base_model"].split('/')[-1]
    model_name = model_name.replace('-Instruct', '').replace('-instruct', '')
    config["output_dir"] = f"{model_name}-vietnamese-legal"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_name"] = f"run_{timestamp}"

    return config


def format_dataset_to_text(dataset, tokenizer):
    """Format conversations dataset to text for causal LM training."""
    print("📝 Formatting dataset to text format...")

    # Apply chat template
    def format_conversations(examples):
        texts = []
        for convo in examples["messages"]:
            # Use apply_chat_template if available
            if hasattr(tokenizer, 'apply_chat_template'):
                text = tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                # Fallback: simple format
                text = ""
                for msg in convo:
                    if msg['role'] == 'system':
                        text += f"System: {msg['content']}\n"
                    elif msg['role'] == 'user':
                        text += f"User: {msg['content']}\n"
                    elif msg['role'] == 'assistant':
                        text += f"Assistant: {msg['content']}\n"
                text += "Assistant:"
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
            max_length=2048,  # Reduced for efficiency
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
    """Run training with plain Transformers."""
    print("="*60)
    print("🚀 PLAIN TRANSFORMERS TRAINING (No Unsloth)")
    print("="*60)
    print(f"📦 Base Model: {config['base_model']}")
    print(f"📏 Max Seq Length: {config['max_seq_length']}")
    print(f"📊 Batch Size: {config['batch_size']} x {config['gradient_accumulation']} grad accum")
    print(f"🎯 Learning Rate: {config['learning_rate']}")
    print(f"🔧 LoRA: r={config['lora_r']}, alpha={config['lora_alpha']}")
    print("="*60)

    # Load tokenizer
    print(f"\n🔤 Loading tokenizer: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )
    print("✅ Tokenizer loaded")

    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure 4-bit loading
    bnb_config = None
    if config["load_in_4bit"]:
        print("🔧 Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load model
    print(f"\n🤖 Loading model: {config['base_model']}")
    print("   ⏱️  This may take 5-10 minutes...")

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )
    print("✅ Model loaded")

    # Prepare model for k-bit training
    if config["load_in_4bit"]:
        print("🔧 Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        fp16=True,
        report_to="none",
        save_strategy="steps",
        remove_unused_columns=False,
        gradient_checkpointing=True,
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
        "framework": "transformers",
    }

    info_path = Path(config["output_dir"]) / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✅ Training info saved to {info_path}")

    return model, tokenizer, info


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

    print(f"✅ Uploaded to: https://huggingface.co/{repo_id}")


def main():
    print("="*60)
    print("🚀 VIETNAMESE LEGAL - TRANSFORMERS TRAINING")
    print("="*60)
    print(f"⏰ Started at: {datetime.now().isoformat()}")

    # Load configuration
    config = load_config()

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {vram_gb:.1f} GB")
    else:
        print(f"\n⚠️  No GPU detected")

    try:
        # Train
        model, tokenizer, info = train(config)

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
