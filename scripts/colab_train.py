#!/usr/bin/env python3
"""
Complete training pipeline for Google Colab.
Run this entire script in a single Colab cell.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# ===========================
# CONFIGURATION
# ===========================

# Get HF token from Colab secrets
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN', '')
    if HF_TOKEN:
        print("✅ HF Token loaded from Colab secrets")
    else:
        print("⚠️  Set HF_TOKEN in Colab secrets!")
        HF_TOKEN = ""
except:
    HF_TOKEN = ""
    print("⚠️  Set HF_TOKEN in Colab secrets!")

# Model and training settings
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
HF_USERNAME = "duyet"
HF_REPO_NAME = "qwen-2.5-3b-vietnamese-legal"

# Set environment
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['BASE_MODEL'] = BASE_MODEL
os.environ['DATASET_NAME'] = "duyet/vietnamese-legal-instruct"
os.environ['MAX_SEQ_LENGTH'] = "2048"
os.environ['BATCH_SIZE'] = "2"
os.environ['GRADIENT_ACCUMULATION'] = "4"
os.environ['EPOCHS'] = "1"
os.environ['LEARNING_RATE'] = "2e-4"
os.environ['HF_USERNAME'] = HF_USERNAME
os.environ['HF_REPO_NAME'] = HF_REPO_NAME
os.environ['PUSH_TO_HUB'] = "true"

print("="*60)
print("🚀 VIETNAMESE LEGAL TRAINING")
print("="*60)
print(f"Model: {BASE_MODEL}")
print(f"Output: {HF_USERNAME}/{HF_REPO_NAME}")
print("="*60)

# ===========================
# INSTALL
# ===========================

print("\n📦 Installing...")
subprocess.run([
    "pip", "install", "-q",
    "datasets", "transformers", "peft", "bitsandbytes", "accelerate"
], check=True)

# ===========================
# CLONE
# ===========================

print("\n📂 Cloning repository...")
subprocess.run([
    "rm", "-rf", "gemma-4-finetune-vietnamese-legal-documents"
], stderr=subprocess.DEVNULL)
subprocess.run([
    "git", "clone", "--depth", "1",
    "https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents.git"
], check=True)

os.chdir("gemma-4-finetune-vietnamese-legal-documents")

# ===========================
# TRAIN
# ===========================

print("\n🚀 Training...")
result = subprocess.run([
    sys.executable, "hf_jobs/train_transformers_native.py"
], env=os.environ)

if result.returncode != 0:
    print("\n❌ Training failed!")
    sys.exit(1)

print("\n✅ Training complete!")

# ===========================
# EVALUATE
# ===========================

print("\n📊 Evaluating...")
subprocess.run([
    sys.executable, "scripts/evaluate_and_report.py"
], env=os.environ)

# ===========================
# DOWNLOAD
# ===========================

print("\n📥 Preparing download...")

# Find output directory
for candidate in Path('.').glob('*-vietnamese-legal'):
    if (candidate / 'training_info.json').exists():
        output_dir = candidate
        break

if output_dir:
    from google.colab import files
    import zipfile

    zip_name = f"{output_dir}.zip"

    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in output_dir.rglob('*'):
            if file.is_file():
                zipf.write(file, file.relative_to(output_dir.parent))

    size_mb = Path(zip_name).stat().st_size / (1024*1024)
    print(f"📥 Downloading: {zip_name} ({size_mb:.1f} MB)")
    files.download(zip_name)

print("\n" + "="*60)
print("✅ COMPLETE")
print("="*60)
print(f"Model: https://huggingface.co/{HF_USERNAME}/{HF_REPO_NAME}")
print("="*60)
