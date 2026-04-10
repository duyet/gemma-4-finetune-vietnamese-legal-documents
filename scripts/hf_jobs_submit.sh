#!/bin/bash
# Submit training job to HuggingFace Jobs
#
# Usage:
#   bash scripts/hf_jobs_submit.sh [--repo REPO_NAME] [--hardware HARDWARE]

set -e

# Default values
REPO_NAME="${HF_REPO_NAME:-gemma-4-E2B-vietnamese-legal}"
HARDWARE="${HARDWARE:-t4.medium}"  # Free tier
DATASET_NAME="${DATASET_NAME:-duyet/vietnamese-legal-instruct}"

echo "🚀 HuggingFace Jobs Training Submission"
echo "======================================"
echo "Repo: $REPO_NAME"
echo "Hardware: $HARDWARE"
echo "Dataset: $DATASET_NAME"
echo ""

# Check if hf CLI is installed
if ! command -v hf &> /dev/null; then
    echo "❌ hf CLI not found"
    echo "Install: pip install -U huggingface_hub"
    exit 1
fi

# Check if logged in
echo "🔐 Checking authentication..."
if ! hf auth whoami &> /dev/null; then
    echo "❌ Not logged in to HuggingFace"
    echo "Login: hf auth login"
    exit 1
fi

HF_USERNAME=$(hf auth whoami 2>&1 | grep "^user=" | sed 's/user=//' | sed 's/ .*//')
echo "✅ Authenticated as: $HF_USERNAME"

# Create job package directory
JOB_DIR=".hf_job_package"
rm -rf "$JOB_DIR"
mkdir -p "$JOB_DIR"

echo ""
echo "📦 Creating job package..."

# Copy training script
cp hf_jobs/train.py "$JOB_DIR/"

# Copy dataset processing code
cp -r crawler "$JOB_DIR/"
cp -r scripts "$JOB_DIR/"

# Create requirements.txt
cat > "$JOB_DIR/requirements.txt" << EOF
# Core dependencies
transformers==4.46.2
trl==0.9.6
accelerate==1.1.0
bitsandbytes==0.43.1
datasets==3.0.1
peft==0.12.0

# Unsloth
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git

# Utilities
huggingface_hub
EOF

# Create README for job repo
cat > "$JOB_DIR/README.md" << EOF
# Gemma 4 Vietnamese Legal - HF Job

Training job for fine-tuning Gemma 4 on Vietnamese legal documents.

**Status:** Training in progress

**Hardware:** $HARDWARE

**Dataset:** $DATASET_NAME
EOF

echo "✅ Job package created: $JOB_DIR/"

# Create or update repo
FULL_REPO_NAME="$HF_USERNAME/$REPO_NAME"
echo ""
echo "📁 Setting up repo: $FULL_REPO_NAME"

# Create repo if it doesn't exist
if ! hf repos info "$FULL_REPO_NAME" &> /dev/null; then
    echo "   Creating new repo..."
    hf repos create "$FULL_REPO_NAME" --type model
else
    echo "   Using existing repo"
fi

# Upload files to repo
echo ""
echo "📤 Uploading to HuggingFace..."
hf upload "$FULL_REPO_NAME" "$JOB_DIR/" --repo-type model

echo "✅ Files uploaded"

# Submit job
echo ""
echo "🚀 Submitting job to HF Jobs..."

# Create job spec
cat > /tmp/job_spec.yaml << EOF
hardware: $HARDWARE
docker_image: ghcr.io/huggingface/unsloth:latest
command:
  - python
  - hf_jobs/train.py
env:
  - BASE_MODEL=unsloth/gemma-4-E2B-it
  - MAX_SEQ_LENGTH=4096
  - BATCH_SIZE=2
  - GRADIENT_ACCUMULATION=4
  - EPOCHS=1
  - LEARNING_RATE=2e-4
  - LORA_R=16
  - LORA_ALPHA=16
  - HF_USERNAME=$HF_USERNAME
  - HF_REPO_NAME=$REPO_NAME
  - PUSH_TO_HUB=true
  - EXPORT_GGUF=true
  - QUANTIZATION=q4_k_m
  - DATASET_NAME=$DATASET_NAME
EOF

# Submit via CLI (if available) or provide manual instructions
echo ""
echo "⚠️  To submit the job, use the web interface:"
echo ""
echo "   1. Go to: https://huggingface.co/jobs"
echo "   2. Click 'New Job'"
echo "   3. Select repo: $FULL_REPO_NAME"
echo "   4. Select hardware: $HARDWARE"
echo "   5. Environment variables (see above)"
echo "   6. Start job"
echo ""
echo "Or use the CLI (if jobs enabled):"
echo "   hf jobs run --hardware $HARDWARE --env hf_jobs/train.py"
echo ""

# Show job monitoring commands
echo "📊 Monitor job:"
echo "   hf jobs list"
echo "   hf jobs logs <job-id> --follow"
echo ""
echo "📦 Results will be at:"
echo "   https://huggingface.co/$FULL_REPO_NAME"
echo ""

# Cleanup
rm -rf "$JOB_DIR"

echo "✅ Setup complete!"
