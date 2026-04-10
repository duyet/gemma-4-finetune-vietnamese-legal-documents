#!/bin/bash
# Submit HF Job for Gemma 4 training
# Usage: bash scripts/submit_gemma4_job.sh

set -e

# Configuration
DATASET_NAME="duyet/vietnamese-legal-instruct"
BASE_MODEL="unsloth/gemma-4-E2B-it"
OUTPUT_REPO="duyet/gemma-4-E2B-vietnamese-legal"
HARDWARE="${HARDWARE:-t4-medium}"  # Free tier

echo "🚀 Submitting Gemma 4 Training Job"
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

# Note: This method requires git clone because hf jobs run doesn't support
# local script uploads like hf jobs uv run does. Use submit_hf_job.sh for
# faster submission with uv run (recommended for Llama models).

# Submit job using original train.py with full environment
echo "📤 Submitting job..."

hf jobs run \
    --flavor "$HARDWARE" \
    --env "BASE_MODEL=$BASE_MODEL" \
    --env "DATASET_NAME=$DATASET_NAME" \
    --env "MAX_SEQ_LENGTH=4096" \
    --env "BATCH_SIZE=2" \
    --env "GRADIENT_ACCUMULATION=4" \
    --env "EPOCHS=1" \
    --env "LEARNING_RATE=2e-4" \
    --env "LORA_R=16" \
    --env "LORA_ALPHA=16" \
    --env "HF_USERNAME=duyet" \
    --env "HF_REPO_NAME=gemma-4-E2B-vietnamese-legal" \
    --env "PUSH_TO_HUB=true" \
    --env "EXPORT_GGUF=true" \
    --env "QUANTIZATION=q4_k_m" \
    --secrets "HF_TOKEN" \
    --detach \
    unsloth/unsloth \
    bash -c "cd /workspace && git clone --depth 1 https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents.git && cd gemma-4-finetune-vietnamese-legal-documents && python hf_jobs/train.py"

echo ""
echo "✅ Job submitted!"
echo ""
echo "Waiting for job ID..."
sleep 2

# Get job ID
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
    echo "Or use monitoring script:"
    echo "   bash scripts/monitor_hf_job.sh $JOB_ID"
else
    echo "⚠️  Could not find job in list. Check:"
    echo "   https://huggingface.co/jobs"
fi
