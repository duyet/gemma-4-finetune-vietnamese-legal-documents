#!/bin/bash
# Submit HF Job using hf jobs uv run (simplified approach)
# Usage: bash scripts/submit_hf_job.sh

set -e

# Configuration
DATASET_NAME="duyet/vietnamese-legal-instruct"
BASE_MODEL="unsloth/Llama-3.2-3B-Instruct"
OUTPUT_REPO="duyet/gemma-4-E2B-vietnamese-legal"
HARDWARE="${HARDWARE:-t4-medium}"  # Free tier

echo "🚀 Submitting HF Job via uv run"
echo "================================"
echo "Hardware: $HARDWARE"
echo "Dataset: $DATASET_NAME"
echo "Base Model: $BASE_MODEL"
echo "Output: $OUTPUT_REPO"
echo ""

# Check if logged in
if ! hf auth whoami &> /dev/null; then
    echo "❌ Not logged in"
    echo "Login: hf auth login"
    exit 1
fi

echo "✅ Authenticated"
echo ""

# Submit job using uv run with local script
echo "📤 Submitting job..."

# Use relative path from project root
SCRIPT_PATH="hf_jobs/uv_train.py"

hf jobs uv run "$SCRIPT_PATH" \
    --flavor "$HARDWARE" \
    --dataset "$DATASET_NAME" \
    --output-repo "$OUTPUT_REPO" \
    --base-model "$BASE_MODEL" \
    --max-seq-length 4096 \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --epochs 1 \
    --learning-rate 2e-4 \
    --lora-r 16 \
    --lora-alpha 16 \
    --quantization q4_k_m

echo ""
echo "✅ Job submitted!"
echo ""
echo "Monitor with:"
echo "   hf jobs ps"
echo ""
echo "📦 Results will be at:"
echo "   https://huggingface.co/$OUTPUT_REPO"
