#!/bin/bash
# Setup HuggingFace Repositories for TVPL

set -e

echo "================================"
echo "HuggingFace Repository Setup for TVPL"
echo "================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
HF_USERNAME="${HF_USERNAME:-your-username}"
DATASET_NAME="tvpl-vi-legal"
MODEL_NAME="gemma4-tvpl-legal"

# Check if logged in
check_login() {
    if ! huggingface-cli whoami &> /dev/null; then
        log_warn "Not logged in to HuggingFace"
        echo "Please login:"
        huggingface-cli login
    fi
}

# Create dataset repository
create_dataset_repo() {
    log_info "Creating dataset repository: $HF_USERNAME/$DATASET_NAME"

    huggingface-cli repo create \
        --type dataset \
        --yes \
        "$HF_USERNAME/$DATASET_NAME" || log_warn "Dataset repo may already exist"

    log_info "✅ Dataset repository: https://huggingface.co/datasets/$HF_USERNAME/$DATASET_NAME"
}

# Create model repository
create_model_repo() {
    log_info "Creating model repository: $HF_USERNAME/$MODEL_NAME"

    huggingface-cli repo create \
        --type model \
        --yes \
        "$HF_USERNAME/$MODEL_NAME" || log_warn "Model repo may already exist"

    log_info "✅ Model repository: https://huggingface.co/models/$HF_USERNAME/$MODEL_NAME"
}

# Generate README files
generate_readmes() {
    log_info "Generating README files..."

    python3 -c "
from scripts.create_hf_repos import create_dataset_readme, create_model_readme

# Dataset README
with open('data/DATASET_README.md', 'w', encoding='utf-8') as f:
    f.write(create_dataset_readme())
print('✅ Generated: data/DATASET_README.md')

# Model README
with open('data/MODEL_README.md', 'w', encoding='utf-8') as f:
    f.write(create_model_readme())
print('✅ Generated: data/MODEL_README.md')
"

    log_info "✅ README files generated"
}

# Show structure
show_structure() {
    cat << EOF

Recommended Repository Structure:

1. Dataset Repository
   📁 $HF_USERNAME/$DATASET_NAME
      ├── data/
      │   ├── documents/    (150K+ legal documents)
      │   ├── passages/     (500K+ chunks for RAG)
      │   ├── pretrain/     (Training corpus)
      │   └── sft/          (Q&A pairs)
      └── README.md

2. Model Repository
   📁 $HF_USERNAME/$MODEL_NAME
      ├── adapter_model.safetensors  (LoRA weights)
      ├── tokenizer files
      ├── config.json
      ├── gguf/
      │   ├── q4_k_m.gguf  (Fast inference)
      │   └── q5_k_m.gguf  (Higher quality)
      └── README.md

3. Code Repository (Optional - GitHub is fine)
   📁 YOUR_USERNAME/tvpl
      ├── crawler/
      ├── scripts/
      ├── rag/
      └── notebooks/

EOF
}

# Main
main() {
    echo "Select repositories to create:"
    echo "1) Dataset repository"
    echo "2) Model repository"
    echo "3) Both"
    echo "4) Just show structure"
    echo ""
    read -p "Choice [1-4]: " choice

    case $choice in
        1)
            check_login
            create_dataset_repo
            generate_readmes
            ;;
        2)
            check_login
            create_model_repo
            generate_readmes
            ;;
        3)
            check_login
            create_dataset_repo
            create_model_repo
            generate_readmes
            ;;
        4)
            show_structure
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac

    echo ""
    log_info "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Prepare data: uv run python scripts/download_hf_dataset.py"
    echo "2. Optional crawl: uv run python crawler/playwright_crawler.py --max-pages 50"
    echo "3. Merge data: uv run python scripts/merge_datasets.py"
    echo "4. Upload dataset: uv run python scripts/upload_with_xet.py -r $HF_USERNAME/$DATASET_NAME"
    echo "5. Train on Colab: Upload notebooks/Auto_Train.ipynb"
    echo "6. Upload model to HF: (Auto_Train.ipynb will do this)"
}

main "$@"
