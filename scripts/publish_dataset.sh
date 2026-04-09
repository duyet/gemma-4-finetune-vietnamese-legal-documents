#!/bin/bash
# TVPL Dataset Publishing Script
# Prepares and uploads dataset to HuggingFace Hub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
REPO_ID="${HF_REPO:-your-username/tvpl-vi-legal}"
DATA_DIR="$PROJECT_ROOT/data/hf_dataset"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if logged in to HuggingFace
check_hf_login() {
    if ! command -v huggingface-cli &> /dev/null; then
        log_error "huggingface-cli not found. Install with: pip install huggingface_hub"
        exit 1
    fi

    if ! huggingface-cli whoami &> /dev/null; then
        log_warn "Not logged in to HuggingFace"
        echo "Please login:"
        huggingface-cli login
    fi
}

# Prepare dataset
prepare_dataset() {
    log_info "Preparing dataset..."

    cd "$PROJECT_ROOT"

    if [ ! -d "data/processed" ]; then
        log_error "Processed data not found. Run: uv run tvpl-process"
        exit 1
    fi

    # Prepare using Python script
    uv run python scripts/prepare_hf_dataset.py \
        --input-dir data/processed \
        --output-dir "$DATA_DIR" \
        --repo-id "$REPO_ID"

    log_info "Dataset prepared at: $DATA_DIR"
}

# Upload dataset
upload_dataset() {
    log_info "Uploading to: $REPO_ID"

    # Try XET first (fast for large files)
    if uv run python -c "import huggingface_hub.xet" 2>/dev/null; then
        log_info "Using XET for fast upload..."
        uv run python scripts/upload_with_xet.py \
            --repo-id "$REPO_ID" \
            --data-dir "$DATA_DIR"
    else
        log_info "Using regular upload..."
        huggingface-cli upload "$DATA_DIR" "$REPO_ID" --repo-type dataset
    fi

    log_info "✅ Dataset uploaded!"
    log_info "View at: https://huggingface.co/datasets/$REPO_ID"
}

# Show usage
usage() {
    cat << EOF
TVPL Dataset Publishing Script

Usage: $(basename "$0") [OPTIONS]

Options:
    -r, --repo REPO_ID       HuggingFace repo (default: your-username/tvpl-vi-legal)
    -p, --prepare-only       Only prepare, don't upload
    -u, --upload-only        Only upload (skip preparation)
    -h, --help               Show this help

Environment Variables:
    HF_REPO                  Default repo ID

Examples:
    # Prepare and upload
    $(basename "$0")

    # Custom repo
    $(basename "$0") -r username/tvpl-vi-legal

    # Prepare only
    $(basename "$0") -p

    # Upload only (after preparing)
    $(basename "$0") -u
EOF
}

# Parse arguments
PREPARE_ONLY=false
UPLOAD_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--repo)
            REPO_ID="$2"
            shift 2
            ;;
        -p|--prepare-only)
            PREPARE_ONLY=true
            shift
            ;;
        -u|--upload-only)
            UPLOAD_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main
main() {
    check_hf_login

    if [ "$UPLOAD_ONLY" = false ]; then
        prepare_dataset
    fi

    if [ "$PREPARE_ONLY" = false ]; then
        upload_dataset
    fi
}

main "$@"
