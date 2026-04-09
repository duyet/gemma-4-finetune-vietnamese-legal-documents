#!/bin/bash
# Configuration helper for Bash scripts
# Source this file to load all configuration variables

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env file if it exists
load_env() {
    local env_file="$PROJECT_ROOT/.env"
    if [ -f "$env_file" ]; then
        set -a
        source "$env_file"
        set +a
    fi
}

# Load environment
load_env

# ============================================
# Git Configuration
# ============================================
export GITHUB_REPO="${GITHUB_REPO:-git@github.com:duyet/gemma-4-finetune-vietnamese-legal-documents.git}"
export GITHUB_USERNAME="${GITHUB_USERNAME:-duyet}"

export HF_USERNAME="${HF_USERNAME:-duyet}"
export HF_REPO_NAME="${HF_REPO_NAME:-gemma-4-finetune-vietnamese-legal-documents}"
export HF_DATASET_NAME="${HF_DATASET_NAME:-vietnamese-legal-documents}"
export HF_MODEL_NAME="${HF_MODEL_NAME:-gemma-4-vietnamese-legal-rag}"

# ============================================
# Crawler Configuration
# ============================================
export CRAWLER_MAX_PAGES="${CRAWLER_MAX_PAGES:-100}"
export CRAWLER_DELAY="${CRAWLER_DELAY:-2.5}"
export CRAWLER_CONCURRENT_REQUESTS="${CRAWLER_CONCURRENT_REQUESTS:-4}"
export CRAWLER_USER_AGENT="${CRAWLER_USER_AGENT:-Mozilla/5.0 (compatible; TVPLBot/1.0)}"

export CRAWLER_START_URL="${CRAWLER_START_URL:-https://thuvienphapluat.vn/}"

# ============================================
# Training Configuration
# ============================================
export BASE_MODEL="${BASE_MODEL:-unsloth/gemma-4-E2B-it}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"

export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs_pretrain}"

# ============================================
# RAG Configuration
# ============================================
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-bkai-foundation-models/vietnamese-bi-encoder}"
export VECTOR_STORE_PATH="${VECTOR_STORE_PATH:-./data/chroma_db}"
export TOP_K_RETRIEVAL="${TOP_K_RETRIEVAL:-3}"

export TEMPERATURE="${TEMPERATURE:-0.7}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# ============================================
# Paths
# ============================================
export DATA_DIR="${DATA_DIR:-./data}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-./data/raw}"
export PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-./data/processed}"
export PRETRAIN_DIR="${PRETRAIN_DIR:-./data/pretrain}"
export SFT_DIR="${SFT_DIR:-./data/sft}"

# ============================================
# Helper Functions
# ============================================

# Print current configuration
print_config() {
    echo "=========================================="
    echo "Current Configuration"
    echo "=========================================="
    echo ""
    echo "[Git Configuration]"
    echo "  GitHub Repo: $GITHUB_REPO"
    echo "  GitHub Username: $GITHUB_USERNAME"
    echo "  HF Username: $HF_USERNAME"
    echo "  HF Dataset: $HF_USERNAME/$HF_DATASET_NAME"
    echo "  HF Model: $HF_USERNAME/$HF_MODEL_NAME"
    echo ""
    echo "[Crawler Configuration]"
    echo "  Max Pages: $CRAWLER_MAX_PAGES"
    echo "  Delay: ${CRAWLER_DELAY}s"
    echo "  Concurrent: $CRAWLER_CONCURRENT_REQUESTS"
    echo ""
    echo "[Training Configuration]"
    echo "  Base Model: $BASE_MODEL"
    echo "  Max Seq Length: $MAX_SEQ_LENGTH"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  LoRA R: $LORA_R"
    echo ""
    echo "[RAG Configuration]"
    echo "  Embedding Model: $EMBEDDING_MODEL"
    echo "  Vector Store: $VECTOR_STORE_PATH"
    echo "  Top K: $TOP_K_RETRIEVAL"
    echo ""
    echo "[Path Configuration]"
    echo "  Data Dir: $DATA_DIR"
    echo ""
}

# Export functions
export -f print_config
