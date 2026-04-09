#!/bin/bash
# Helper script to prepare notebook for Colab upload

PROJECT_ROOT="$(pwd)"
NOTEBOOK_PATH="$PROJECT_ROOT/notebooks/Auto_Train.ipynb"

echo "=================================="
echo "Colab Upload Helper"
echo "=================================="
echo ""

# Check if notebook exists
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "❌ Error: Notebook not found at $NOTEBOOK_PATH"
    exit 1
fi

echo "✅ Notebook found: $NOTEBOOK_PATH"
echo ""
echo "Next steps:"
echo ""
echo "1. Open Google Colab:"
echo "   https://colab.research.google.com/"
echo ""
echo "2. Upload the notebook:"
echo "   File → Upload notebook → Select: notebooks/Auto_Train.ipynb"
echo ""
echo "3. Configure (edit the first cell):"
echo ""
echo '   GITHUB_REPO = "https://github.com/duyet/gemma-4-finetune-vietnamese-legal-documents"'
echo '   GITHUB_USERNAME = "duyet"'
echo '   HF_USERNAME = "duyet"'
echo '   MAX_PAGES = 0  # Skip crawling, use HF dataset only'
echo '   STAGE = "pretrain"'
echo ""
echo "4. Connect to GPU:"
echo "   Runtime → Change runtime type → T4 GPU"
echo ""
echo "5. Run all cells:"
echo "   Runtime → Run all (or Cmd/Ctrl+F9)"
echo ""

# Open Colab in browser
echo "Opening Google Colab in browser..."
open "https://colab.research.google.com/"

echo ""
echo "✅ Browser opened! Please follow the steps above."
