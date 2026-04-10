#!/bin/bash
# Submit HuggingFace Job via CLI
# Usage: bash scripts/hf_jobs_run.sh

set -e

# Configuration
HF_USERNAME="duyet"
HF_REPO_NAME="gemma-4-E2B-vietnamese-legal"
DATASET_NAME="duyet/vietnamese-legal-instruct"
BASE_MODEL="unsloth/gemma-4-E2B-it"
HARDWARE="${HARDWARE:-t4-medium}"  # Free tier

echo "🚀 Submitting HF Job via CLI"
echo "================================"
echo "Hardware: $HARDWARE"
echo "Dataset: $DATASET_NAME"
echo "Base Model: $BASE_MODEL"
echo "Repo: $HF_USERNAME/$HF_REPO_NAME"
echo ""

# Check if logged in
if ! hf auth whoami &> /dev/null; then
    echo "❌ Not logged in"
    echo "Login: hf auth login"
    exit 1
fi

echo "✅ Authenticated"

# Submit job with working directory and compatible versions
echo ""
echo "📤 Submitting job..."

hf jobs run \
    --flavor "$HARDWARE" \
    --env "BASE_MODEL=$BASE_MODEL" \
    --env "MAX_SEQ_LENGTH=4096" \
    --env "BATCH_SIZE=2" \
    --env "GRADIENT_ACCUMULATION=4" \
    --env "EPOCHS=1" \
    --env "LEARNING_RATE=2e-4" \
    --env "LORA_R=16" \
    --env "LORA_ALPHA=16" \
    --env "LORA_DROPOUT=0.05" \
    --env "HF_USERNAME=$HF_USERNAME" \
    --env "HF_REPO_NAME=$HF_REPO_NAME" \
    --env "PUSH_TO_HUB=true" \
    --env "EXPORT_GGUF=true" \
    --env "QUANTIZATION=q4_k_m" \
    --env "DATASET_NAME=$DATASET_NAME" \
    --env "DATASET_SPLIT=train" \
    --env "OUTPUT_DIR=/workspace/gemma-4-E2B-vietnamese-legal" \
    --secrets "HF_TOKEN" \
    --detach \
    unsloth/unsloth \
    bash -c "cd /workspace && git clone --depth 1 https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents.git && cd /workspace/gemma-4-finetune-vietnamese-legal-documents && python /workspace/gemma-4-finetune-vietnamese-legal-documents/hf_jobs/train.py"

echo ""
echo "✅ Job submitted!"
echo ""
echo "Capturing full job ID..."
sleep 2

# Get all jobs to find our submitted job
JOB_INFO=$(hf jobs ps 2>&1 | grep "unsloth/unsloth" | head -1)
if [ -n "$JOB_INFO" ]; then
    JOB_ID=$(echo "$JOB_INFO" | awk '{print $1}')
    echo "Job ID: $JOB_ID"
    echo ""
    echo "📊 Monitor job:"
    echo "   hf jobs inspect $JOB_ID"
    echo ""
    echo "📋 View logs:"
    echo "   hf jobs logs $JOB_ID --follow"
    echo ""
    echo "📦 Results will be at:"
    echo "   https://huggingface.co/$HF_USERNAME/$HF_REPO_NAME"
else
    echo "⚠️  Could not find job in list. Check:"
    echo "   https://huggingface.co/jobs"
fi
