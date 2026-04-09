#!/bin/bash
# Unified Colab helper script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NOTEBOOK_PATH="$PROJECT_ROOT/notebooks/Gemma4_Vietnamese_Legal_Finetune.ipynb"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "=========================================="
echo "Gemma 4 Vietnamese Legal - Colab Helper"
echo "=========================================="
echo ""

# Check if notebook exists
if [ ! -f "$NOTEBOOK_PATH" ]; then
    log_warn "Notebook not found at: $NOTEBOOK_PATH"
    echo "Available notebooks:"
    ls -1 "$PROJECT_ROOT/notebooks/"*.ipynb 2>/dev/null || echo "  No notebooks found"
    exit 1
fi

log_info "Notebook: Gemma4_Vietnamese_Legal_Finetune.ipynb"
log_info "Location: $NOTEBOOK_PATH"
echo ""

echo "This notebook orchestrates scripts from the repository:"
echo "  📦 Clone repository"
echo "  🕷️ Crawl data (optional)"
echo "  📊 Download/prepare datasets"
echo "  🎓 Fine-tune Gemma 4 E2B"
echo "  📦 Export to GGUF"
echo "  📈 Generate evaluation scores"
echo ""

echo "All training logic lives in Python scripts."
echo "Edit the configuration in the first cell:"
echo ""
echo "  CRAWL_ENABLED = False           # Enable crawling"
echo "  DOWNLOAD_HF_DATASET = True     # Download base dataset"
echo "  RUN_TRAINING = True             # Run fine-tuning"
echo "  TRAINING_STAGE = 'pretrain'      # pretrain, sft, or both"
echo "  MAX_SEQ_LENGTH = 4096"
echo "  BATCH_SIZE = 2"
echo "  EXPORT_TO_GGUF = True           # Export to GGUF"
echo "  GENERATE_SCORES = True          # Generate scores"
echo ""

echo "=========================================="
echo "STEPS TO UPLOAD TO COLAB:"
echo "=========================================="
echo ""
echo "1. Google Colab will open in your browser"
echo ""
echo "2. In Colab, click: File → Upload notebook"
echo ""
echo "3. Navigate to and select:"
echo "   $NOTEBOOK_PATH"
echo ""
echo "4. Click: Runtime → Change runtime type → T4 GPU"
echo ""
echo "5. Edit the FLAGS in the first cell (set True/False)"
echo ""
echo "6. Run: Runtime → Run all (or press Cmd/Ctrl+F9)"
echo ""
echo "=========================================="

# Open Colab
log_info "Opening Google Colab..."
open "https://colab.research.google.com/"

echo ""
log_info "✅ Colab opened! Follow the steps above."
echo ""
